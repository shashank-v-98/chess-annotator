"""Positional and tactical features for a single move -- human-style chess ideas,
computed deterministically from the board with python-chess.

features.py supplies the ENGINE facts (evaluations, best move, PV). This module
supplies the IDEAS a human annotator talks about: open and semi-open files,
pawn structure (isolated / doubled / passed / hanging pawns), bishops
(pair, opposite colours, "bad" bishop, pawns fixed on one colour), endgame
type, king safety, restricted pieces, and one-move tactics (checks, forks,
pins, skewers, discovered attacks, threats, mate threats, pieces left en prise).

Everything here is a verifiable fact about the position. The language model
only ever phrases these facts; it is never asked to find them itself.

Design rules:
  * Report what the MOVE CHANGED (a file opened, a pin created, a passed pawn
    appeared) rather than every static feature on the board -- otherwise every
    annotation would recite the same inventory.
  * Tactical claims use conservative definitions (a "fork" needs two targets
    that are each the king, worth more than the forking piece, or undefended).
  * Judgements that depend on the engine (e.g. an opposite-coloured-bishop
    ending being drawish) require the engine evaluation to agree.
"""

from __future__ import annotations

from typing import Optional

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SIDE = {chess.WHITE: "White", chess.BLACK: "Black"}
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
QUEENSIDE = range(0, 4)   # files a-d
KINGSIDE = range(4, 8)    # files e-h
DRAWISH_OCB_CP = 150      # |eval| at or below this, in an opposite-coloured-bishop ending, counts as drawish

MAX_MOVE_INSIGHTS = 5
MAX_CONTEXT = 3


# --------------------------------------------------------------------------- helpers

def _pname(board: chess.Board, sq: int) -> str:
    return chess.piece_name(board.piece_type_at(sq))


def _desc(board: chess.Board, sq: int, with_side: bool = False) -> str:
    p = board.piece_at(sq)
    base = f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
    return f"{SIDE[p.color]}'s {base}" if with_side else base


def _is_light(sq: int) -> bool:
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def _colour_word(light: bool) -> str:
    return "light" if light else "dark"


def _a(noun: str) -> str:
    return ("an " if noun[:1] in "aeiou" else "a ") + noun


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _effective_attackers(board: chess.Board, color: bool, sq: int, relative: bool = False) -> list:
    """Pieces of `color` that can actually capture on `sq`: drops absolutely pinned
    pieces whose pin line doesn't include sq, and a king that would capture into a
    defended square. With relative=True, also drops pieces relatively pinned to
    something more valuable than the piece on `sq` (capturing would lose more)."""
    out = []
    rel = _relative_pins_against(board, color) if relative else {}
    target = board.piece_at(sq)
    for a in board.attackers(color, sq):
        if board.is_pinned(color, a) and not (board.pin(color, a) & chess.BB_SQUARES[sq]):
            continue
        if board.piece_type_at(a) == chess.KING and board.attackers(not color, sq):
            continue
        if a in rel and target is not None:
            by, back = rel[a]
            on_line = chess.between(by, back) | chess.BB_SQUARES[by]
            if not (on_line & chess.BB_SQUARES[sq]) and VALUE[board.piece_type_at(back)] > VALUE[target.piece_type]:
                continue
        out.append(a)
    return out


def _relative_pins_against(board: chess.Board, color: bool) -> dict:
    """{pinned square: (pinner square, square behind)} for `color`'s pieces pinned
    (absolutely or relatively) by the opponent."""
    return {d["front"]: (d["by"], d["back"]) for d in line_tactics(board, not color) if d["type"] == "pin"}


def pinned_would_be_capturers(board: chess.Board, sq: int) -> list:
    """Enemy pieces that attack sq and are cheaper than the piece on it, but can't
    capture because they're pinned to something more valuable. Returns
    (attacker, pinner, back) triples -- the 'uses the pin' idea."""
    piece = board.piece_at(sq)
    if piece is None:
        return []
    enemy = not piece.color
    rel = _relative_pins_against(board, enemy)
    out = []
    for a in board.attackers(enemy, sq):
        if a in rel and VALUE[board.piece_type_at(a)] < VALUE[piece.piece_type]:
            by, back = rel[a]
            on_line = chess.between(by, back) | chess.BB_SQUARES[by]
            if not (on_line & chess.BB_SQUARES[sq]):
                out.append((a, by, back))
    return out


def _vulnerable(board: chess.Board, sq: int) -> bool:
    """Is the piece on sq under a real threat: attacked by something cheaper, or
    attacked and undefended? (The king counts as vulnerable whenever attacked.)"""
    piece = board.piece_at(sq)
    if piece is None:
        return False
    attackers = _effective_attackers(board, not piece.color, sq, relative=piece.piece_type != chess.KING)
    if not attackers:
        return False
    if piece.piece_type == chess.KING:
        return True
    if min(VALUE[board.piece_type_at(a)] for a in attackers) < VALUE[piece.piece_type]:
        return True
    return not _effective_attackers(board, piece.color, sq)


def threatened_pieces(board: chess.Board, color: bool) -> list:
    """Non-king pieces of `color` that are currently en prise / under a real threat."""
    return [sq for sq, p in board.piece_map().items()
            if p.color == color and p.piece_type != chess.KING and _vulnerable(board, sq)]


# --------------------------------------------------------------------------- tactics

def _fork_targets(board: chess.Board, sq: int) -> list:
    """Enemy pieces the piece on sq attacks that are each a real target."""
    piece = board.piece_at(sq)
    if piece is None:
        return []
    targets = []
    for t in board.attacks(sq):
        tp = board.piece_at(t)
        if tp is None or tp.color == piece.color:
            continue
        if (tp.piece_type == chess.KING or VALUE[tp.piece_type] > VALUE[piece.piece_type]
                or not _effective_attackers(board, tp.color, t)):
            targets.append(t)
    return targets


def _ray_pieces(board: chess.Board, sq: int, dirs: list):
    f0, r0 = chess.square_file(sq), chess.square_rank(sq)
    for df, dr in dirs:
        seen = []
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = chess.square(f, r)
            if board.piece_at(s):
                seen.append(s)
                if len(seen) == 2:
                    break
            f, r = f + df, r + dr
        yield seen


def line_tactics(board: chess.Board, color: bool) -> list:
    """Pins and skewers exerted by `color`'s bishops, rooks and queens."""
    out = []
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        dirs = {chess.BISHOP: BISHOP_DIRS, chess.ROOK: ROOK_DIRS, chess.QUEEN: ROOK_DIRS + BISHOP_DIRS}[p.piece_type]
        for seen in _ray_pieces(board, sq, dirs):
            if len(seen) != 2:
                continue
            a, b = seen
            pa, pb = board.piece_at(a), board.piece_at(b)
            if pa.color == color or pb.color == color:
                continue
            va, vb, vs = VALUE[pa.piece_type], VALUE[pb.piece_type], VALUE[p.piece_type]
            # A pawn pinned along its own file can still advance, so that is not a real pin;
            # a pawn pinned to anything less than the queen or king rarely matters.
            if pa.piece_type == chess.PAWN and (chess.square_file(a) == chess.square_file(sq)
                                                 or pb.piece_type not in (chess.QUEEN, chess.KING)):
                continue
            b_undefended = not _effective_attackers(board, pb.color, b)
            if pa.piece_type != chess.KING and (pb.piece_type == chess.KING or (vb > va and (vb > vs or b_undefended))):
                out.append({"type": "pin", "by": sq, "front": a, "back": b,
                            "absolute": pb.piece_type == chess.KING})
            elif pb.piece_type != chess.KING and va > vb and (
                    pa.piece_type == chess.KING or va > vs or not _effective_attackers(board, pa.color, a)
            ) and (b_undefended or vb > vs):
                out.append({"type": "skewer", "by": sq, "front": a, "back": b, "absolute": False})
    return out


def _line_text(board: chess.Board, d: dict) -> str:
    by = _desc(board, d["by"], with_side=True)
    front, back = _desc(board, d["front"]), _desc(board, d["back"])
    if d["type"] == "pin":
        return f"{by} pins the {front} to the {back}" + (" (it cannot legally move)" if d["absolute"] else "")
    return f"{by} skewers the {front}; when it moves, the {back} behind it is exposed"


def mate_threats(board: chess.Board, color: bool) -> list:
    """Moves with which `color` would mate if it were their turn again (SAN)."""
    if board.is_game_over() or board.is_check():
        return []
    b = board.copy(stack=False)
    if b.turn != color:
        b.push(chess.Move.null())
    found = []
    for mv in list(b.legal_moves):
        san = b.san(mv)
        b.push(mv)
        if b.is_checkmate():
            found.append(san)
        b.pop()
    return found


def move_tactics(before: chess.Board, move: chess.Move) -> list:
    """One-move tactical content of `move` played in position `before`.
    Returns (priority, text) pairs; lower priority = more important."""
    mover, opp = before.turn, not before.turn
    after = before.copy(stack=False)
    san = before.san(move)
    captured = before.piece_at(move.to_square) if before.is_capture(move) else None
    if before.is_en_passant(move):
        captured = chess.Piece(chess.PAWN, opp)
    after.push(move)
    to, frm = move.to_square, move.from_square
    out = []

    if after.is_checkmate():
        return [(0, f"{san} is checkmate")]
    if after.is_stalemate():
        return [(0, f"{san} stalemates the opponent")]

    checkers = list(after.checkers())
    if len(checkers) >= 2:
        out.append((1, f"{san} is a double check"))
    elif checkers and checkers[0] != to:
        out.append((1, f"{san} gives a discovered check from the {_desc(after, checkers[0])}"))
    elif checkers:
        out.append((4, f"{san} gives check"))

    if captured is not None:
        out.append((4, f"{san} captures the {chess.piece_name(captured.piece_type)} on {chess.square_name(to)}"))

    moved_type = after.piece_type_at(to)
    pawn_break = []
    if moved_type == chess.PAWN and captured is None:
        pawn_break = [t for t in after.attacks(to) if after.piece_at(t) == chess.Piece(chess.PAWN, opp)]
        if pawn_break:
            out.append((2, f"{san} is a pawn break, striking at the {_join([_desc(after, t) for t in pawn_break])}"))

    for a, by, back in pinned_would_be_capturers(after, to):
        out.append((1, f"the {_desc(after, to)} cannot be taken safely: the {_desc(after, a)} is pinned "
                       f"by the {_desc(after, by)} to the {_desc(after, back)}"))

    fork = _fork_targets(after, to)
    used = set(fork)
    if len(fork) >= 2:
        caveat = " (though the forking piece can itself be captured)" if _vulnerable(after, to) else ""
        out.append((1, f"{_desc(after, to, True)} forks the {_join([_desc(after, s) for s in fork])}{caveat}"))

    before_lines = {(d["type"], d["front"], d["back"]) for d in line_tactics(before, mover)}
    for d in line_tactics(after, mover):
        if (d["type"], d["front"], d["back"]) not in before_lines:
            out.append((1 if d["type"] == "skewer" or d["absolute"] else 2, _line_text(after, d)))
            used.add(d["front"])

    # Discovered attacks: a line piece other than the mover now hits a real target
    # through the square the moving piece just left.
    for sq, p in after.piece_map().items():
        if p.color != mover or sq == to or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        for t in after.attacks(sq):
            tp = after.piece_at(t)
            if (tp and tp.color == opp and tp.piece_type != chess.KING and t not in used
                    and chess.between(sq, t) & chess.BB_SQUARES[frm] and _vulnerable(after, t)):
                out.append((2, f"{san} uncovers a discovered attack by the {_desc(after, sq)} on the {_desc(after, t)}"))
                used.add(t)

    newly = [s for s in threatened_pieces(after, opp) if s not in set(threatened_pieces(before, opp)) and s not in used]
    if newly:
        out.append((3, f"{san} attacks the {_join([_desc(after, s) for s in newly])}, which "
                       f"{'is' if len(newly) == 1 else 'are'} not adequately defended"))

    if not checkers:
        mt = mate_threats(after, mover)
        if mt:
            out.append((1, f"{san} threatens mate with {mt[0]}"))

    was_threatened = set(threatened_pieces(before, mover))
    hanging = []
    for s in threatened_pieces(after, mover):
        if s == to and captured is not None and VALUE[captured.piece_type] >= VALUE[after.piece_type_at(to)]:
            continue  # a capture that can be recaptured is an exchange, not a blunder of the piece
        if s == to and pawn_break:
            continue  # a pawn break offers the pawn by design; described above
        if s not in was_threatened or s == to:
            hanging.append(s)
    if hanging:
        out.append((2, f"after {san}, the {_join([_desc(after, s) for s in hanging])} "
                       f"{'is' if len(hanging) == 1 else 'are'} left en prise"))

    rescued = [s for s in was_threatened if s != frm and s in after.piece_map() and not _vulnerable(after, s)]
    if frm in was_threatened and not _vulnerable(after, to):
        rescued.append(to)
    if rescued:
        out.append((3, f"{san} removes the threat to the {_join([_desc(after, s) for s in rescued])}"))

    return out


# --------------------------------------------------------------------------- static structure

def _file_pawns(board: chess.Board) -> dict:
    counts = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        counts[chess.WHITE][chess.square_file(sq)] += 1
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        counts[chess.BLACK][chess.square_file(sq)] += 1
    return counts


def file_status(board: chess.Board) -> dict:
    fp = _file_pawns(board)
    status = {}
    for f in range(8):
        w, b = fp[chess.WHITE][f], fp[chess.BLACK][f]
        name = chess.FILE_NAMES[f]
        if w == 0 and b == 0:
            status[name] = "open"
        elif w == 0:
            status[name] = "semi-open for White"
        elif b == 0:
            status[name] = "semi-open for Black"
    return status


def pawn_structure(board: chess.Board, color: bool) -> dict:
    fp = _file_pawns(board)
    own, enemy = fp[color], fp[not color]
    pawns = list(board.pieces(chess.PAWN, color))
    isolated, passed = [], []
    for sq in pawns:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if all(own[g] == 0 for g in (f - 1, f + 1) if 0 <= g < 8):
            isolated.append(sq)
        ahead = (lambda rr: rr > r) if color == chess.WHITE else (lambda rr: rr < r)
        if not any(board.piece_at(s) == chess.Piece(chess.PAWN, not color)
                   for s in chess.SQUARES
                   if abs(chess.square_file(s) - f) <= 1 and ahead(chess.square_rank(s))):
            passed.append(sq)
    doubled = [chess.FILE_NAMES[f] for f in range(8) if own[f] >= 2]
    hanging = []
    for f in range(7):
        if own[f] == 1 and own[f + 1] == 1 and enemy[f] == 0 and enemy[f + 1] == 0 \
                and (f == 0 or own[f - 1] == 0) and (f + 2 > 7 or own[f + 2] == 0):
            a = [s for s in pawns if chess.square_file(s) == f][0]
            b = [s for s in pawns if chess.square_file(s) == f + 1][0]
            if chess.square_rank(a) == chess.square_rank(b):
                hanging.append((a, b))
    fixed = []  # pawns blocked head-on by an enemy pawn
    step = 8 if color == chess.WHITE else -8
    for sq in pawns:
        ahead_sq = sq + step
        if 0 <= ahead_sq < 64 and board.piece_at(ahead_sq) == chess.Piece(chess.PAWN, not color):
            fixed.append(sq)
    return {"isolated": isolated, "passed": passed, "doubled": doubled, "hanging": hanging,
            "fixed": fixed, "queenside": sum(own[f] for f in QUEENSIDE), "kingside": sum(own[f] for f in KINGSIDE)}


def _non_pawn(board: chess.Board, color: bool) -> dict:
    return {pt: len(board.pieces(pt, color)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}


def bishop_info(board: chess.Board) -> dict:
    info = {}
    for color in (chess.WHITE, chess.BLACK):
        bs = list(board.pieces(chess.BISHOP, color))
        info[color] = {"squares": bs, "colours": {_is_light(s) for s in bs}}
    w, b = info[chess.WHITE], info[chess.BLACK]
    ocb = len(w["squares"]) == 1 and len(b["squares"]) == 1 and w["colours"] != b["colours"]
    return {"per_side": info, "opposite_coloured": ocb,
            "pair": {c: len(info[c]["colours"]) == 2 for c in (chess.WHITE, chess.BLACK)}}


def endgame_type(board: chess.Board) -> Optional[str]:
    w, b = _non_pawn(board, chess.WHITE), _non_pawn(board, chess.BLACK)
    total = {c: sum(VALUE[pt] * n for pt, n in _non_pawn(board, c).items()) for c in (chess.WHITE, chess.BLACK)}
    if (w[chess.QUEEN] or b[chess.QUEEN]) and max(total.values()) > 13:
        return None
    if max(total.values()) > 13:
        return None
    kinds = {pt for d in (w, b) for pt, n in d.items() if n}
    if not kinds:
        return "king and pawn endgame"
    if kinds == {chess.ROOK}:
        return "rook endgame"
    if kinds == {chess.QUEEN}:
        return "queen endgame"
    if kinds == {chess.BISHOP}:
        if bishop_info(board)["opposite_coloured"]:
            return "opposite-coloured bishop endgame"
        return "bishop endgame"
    if kinds <= {chess.BISHOP, chess.KNIGHT}:
        return "minor-piece endgame"
    if chess.ROOK in kinds and kinds <= {chess.ROOK, chess.BISHOP, chess.KNIGHT}:
        return "rook and minor-piece endgame"
    return "endgame"


def safe_squares(board: chess.Board, sq: int) -> set:
    """Squares the piece on sq can go to that aren't occupied by its own side and
    aren't attacked by an enemy pawn."""
    p = board.piece_at(sq)
    enemy_pawn_attacks = chess.SquareSet()
    for ep in board.pieces(chess.PAWN, not p.color):
        enemy_pawn_attacks |= board.attacks(ep)
    return {s for s in board.attacks(sq)
            if not (board.piece_at(s) and board.piece_at(s).color == p.color) and s not in enemy_pawn_attacks}


def restricted_pieces(board: chess.Board) -> list:
    """Knights with <=1 and bishops with <=2 safe squares."""
    out = []
    for sq, p in board.piece_map().items():
        if p.piece_type not in (chess.KNIGHT, chess.BISHOP):
            continue
        n = len(safe_squares(board, sq))
        if n <= (1 if p.piece_type == chess.KNIGHT else 2):
            out.append((sq, n))
    return out


def king_shelter(board: chess.Board, color: bool) -> Optional[dict]:
    k = board.king(color)
    if k is None:
        return None
    kf, kr = chess.square_file(k), chess.square_rank(k)
    home = 0 if color == chess.WHITE else 7
    if abs(kr - home) > 1 or kf in (3, 4):  # only meaningful for a king tucked away on a wing
        return None
    step = 1 if color == chess.WHITE else -1
    shield = 0
    for f in (kf - 1, kf, kf + 1):
        if not 0 <= f < 8:
            continue
        for dr in (1, 2):
            r = kr + step * dr
            if 0 <= r < 8 and board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
                shield += 1
                break
    fp = _file_pawns(board)
    exposed = [chess.FILE_NAMES[f] for f in (kf - 1, kf, kf + 1)
               if 0 <= f < 8 and fp[color][f] == 0 and fp[not color][f] == 0
               and any(chess.square_file(s) == f for s in board.pieces(chess.ROOK, not color) | board.pieces(chess.QUEEN, not color))]
    return {"shield": shield, "exposed_files": exposed}


# --------------------------------------------------------------------------- per-move analysis

def _structure_snapshot(board: chess.Board) -> dict:
    return {
        "files": file_status(board),
        "pawns": {c: pawn_structure(board, c) for c in (chess.WHITE, chess.BLACK)},
        "bishops": bishop_info(board),
        "endgame": endgame_type(board),
        "restricted": {sq for sq, _ in restricted_pieces(board)},
        "shelter": {c: king_shelter(board, c) for c in (chess.WHITE, chess.BLACK)},
    }


def _structural_changes(before: chess.Board, after: chess.Board, mover: bool, eval_cp, mate) -> list:
    b, a = _structure_snapshot(before), _structure_snapshot(after)
    out = []
    # Files
    for f, st in a["files"].items():
        if b["files"].get(f) != st:
            out.append((2, f"the {f}-file is now {st}"))
    for f, st in b["files"].items():
        if f not in a["files"]:
            out.append((3, f"the {f}-file is no longer {st}"))
    # Pawn structure
    for c in (chess.WHITE, chess.BLACK):
        pb, pa = b["pawns"][c], a["pawns"][c]
        new_passed = [s for s in pa["passed"] if s not in pb["passed"]]
        if new_passed:
            out.append((2, f"{SIDE[c]} now has a passed pawn on {_join([chess.square_name(s) for s in new_passed])}"))
        new_iso = [s for s in pa["isolated"] if s not in pb["isolated"]]
        if new_iso:
            iqp = any(chess.square_file(s) == 3 for s in new_iso)
            what = ("an isolated queen's pawn" if iqp else "an isolated pawn") if len(new_iso) == 1 else "isolated pawns"
            out.append((3, f"{SIDE[c]} is left with {what} on {_join([chess.square_name(s) for s in new_iso])}"))
        new_dbl = [f for f in pa["doubled"] if f not in pb["doubled"]]
        if new_dbl:
            out.append((3, f"{SIDE[c]} now has doubled pawns on the {_join(new_dbl)}-file"))
        if pa["hanging"] and not pb["hanging"]:
            x, y = pa["hanging"][0]
            out.append((2, f"{SIDE[c]} now has hanging pawns on {chess.square_name(x)} and {chess.square_name(y)}"))
    # Bishops
    for c in (chess.WHITE, chess.BLACK):
        if b["bishops"]["pair"][c] and not a["bishops"]["pair"][c]:
            out.append((3, f"{SIDE[c]} gives up the bishop pair"))
    if a["bishops"]["opposite_coloured"] and not b["bishops"]["opposite_coloured"]:
        out.append((2, "the players now have bishops of opposite colours"))
    # Endgame reached / changed
    if a["endgame"] and a["endgame"] != b["endgame"]:
        text = f"the position becomes {_a(a['endgame'])}"
        if a["endgame"] == "opposite-coloured bishop endgame" and mate is None and eval_cp is not None \
                and abs(eval_cp) <= DRAWISH_OCB_CP:
            text += ", which the engine also assesses as close to level: such endings are notoriously drawish"
        out.append((1, text))
    # Restricted pieces, and squares taken away from enemy minor pieces
    for sq in a["restricted"] - b["restricted"]:
        p = after.piece_at(sq)
        if p and p.color != mover and after.fullmove_number > 10:
            out.append((3, f"{_desc(after, sq, True)} is now short of squares"))
    for sq, p in after.piece_map().items():
        if p.color != mover and p.piece_type in (chess.KNIGHT, chess.BISHOP) and before.piece_at(sq) == p \
                and sq not in a["restricted"] and after.fullmove_number > 12:
            lost = safe_squares(before, sq) - safe_squares(after, sq)
            if lost and (len(lost) >= 2 or len(safe_squares(after, sq)) <= 3):
                out.append((3, f"it takes {_join(sorted(chess.square_name(s) for s in lost))} away from {_desc(after, sq, True)}"))
    # Pawns newly fixed (blocked head-on)
    for c in (chess.WHITE, chess.BLACK):
        new_fixed = [s for s in a["pawns"][c]["fixed"] if s not in b["pawns"][c]["fixed"]]
        if new_fixed and c != mover and after.fullmove_number > 12:
            plural = len(new_fixed) > 1
            out.append((2, f"{SIDE[c]}'s pawn{'s' if plural else ''} on "
                           f"{_join([chess.square_name(s) for s in new_fixed])} "
                           f"{'are' if plural else 'is'} now fixed on "
                           f"{'' if plural else 'a '}{_join(sorted({_colour_word(_is_light(s)) for s in new_fixed}))} square{'s' if plural else ''}"))
    # Pawn majority advance
    last = after.peek() if after.move_stack else None
    if last is not None and after.piece_type_at(last.to_square) == chess.PAWN and before.piece_type_at(last.from_square) == chess.PAWN:
        wing = "kingside" if chess.square_file(last.to_square) >= 4 else "queenside"
        own_n, opp_n = a["pawns"][mover][wing], a["pawns"][not mover][wing]
        if own_n > opp_n:
            out.append((2, f"{SIDE[mover]} advances the {wing} pawn majority ({own_n} pawns against {opp_n})"))
    # King shelter
    for c in (chess.WHITE, chess.BLACK):
        sb, sa = b["shelter"][c], a["shelter"][c]
        if sb and sa and sa["shield"] < sb["shield"]:
            out.append((3, f"the pawn cover in front of {SIDE[c]}'s king is weakened"))
        if sa and set(sa["exposed_files"]) - set((sb or {}).get("exposed_files", [])):
            f = _join(sorted(set(sa["exposed_files"]) - set((sb or {}).get("exposed_files", []))))
            out.append((2, f"{SIDE[not c]}'s heavy pieces now bear down the open {f}-file toward {SIDE[c]}'s king"))
    return out


def position_context(board: chess.Board, eval_cp=None, mate=None) -> list:
    """A few standing features of the position after the move (not changes)."""
    out = []
    eg = endgame_type(board)
    if eg:
        text = f"this is {_a(eg)}"
        if eg == "opposite-coloured bishop endgame" and mate is None and eval_cp is not None \
                and abs(eval_cp) <= DRAWISH_OCB_CP:
            text += " and the engine sees it as close to level: a drawish ending"
        out.append((1, text))
    bi = bishop_info(board)
    if bi["opposite_coloured"] and not eg:
        out.append((3, "the bishops are of opposite colours"))
    for c in (chess.WHITE, chess.BLACK):
        if bi["pair"][c] and not bi["pair"][not c]:
            out.append((4, f"{SIDE[c]} has the bishop pair"))
    for c in (chess.WHITE, chess.BLACK):
        ps = pawn_structure(board, c)
        for x, y in ps["hanging"]:
            out.append((2, f"{SIDE[c]} has hanging pawns on {chess.square_name(x)} and {chess.square_name(y)}"))
        adv = [s for s in ps["passed"] if (chess.square_rank(s) >= 4 if c == chess.WHITE else chess.square_rank(s) <= 3)]
        if adv:
            out.append((2, f"{SIDE[c]} has an advanced passed pawn on {_join([chess.square_name(s) for s in adv])}"))
        # Pawns fixed on one colour, and a bishop hemmed in by its own pawns
        fixed_light = sum(_is_light(s) for s in ps["fixed"])
        fixed_dark = len(ps["fixed"]) - fixed_light
        for light, n in ((True, fixed_light), (False, fixed_dark)):
            if n >= 2:
                out.append((3, f"{SIDE[c]}'s pawns are fixed on {_colour_word(light)} squares"))
        own_bishops = bi["per_side"][c]["squares"]
        pawns = list(board.pieces(chess.PAWN, c))
        if len(own_bishops) == 1 and len(pawns) >= 4:
            same = sum(_is_light(p) == _is_light(own_bishops[0]) for p in pawns)
            if same / len(pawns) >= 0.6 and sum(_is_light(p) == _is_light(own_bishops[0]) for p in ps["fixed"]) >= 2:
                out.append((3, f"{SIDE[c]}'s {_colour_word(_is_light(own_bishops[0]))}-squared bishop is hemmed in by its own pawns (a 'bad' bishop)"))
    # Rooks on open / semi-open files
    fs = file_status(board)
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.ROOK:
            fname = chess.FILE_NAMES[chess.square_file(sq)]
            st = fs.get(fname)
            if st == "open":
                out.append((4, f"{SIDE[p.color]}'s rook on {chess.square_name(sq)} controls the open {fname}-file"))
            elif st == f"semi-open for {SIDE[p.color]}":
                out.append((4, f"{SIDE[p.color]}'s rook on {chess.square_name(sq)} uses the {fname}-file, which is semi-open for {SIDE[p.color]}"))
    # Minor-piece imbalance (one minor piece each, of different kinds)
    minors = {c: list(board.pieces(chess.KNIGHT, c) | board.pieces(chess.BISHOP, c)) for c in (chess.WHITE, chess.BLACK)}
    if len(minors[chess.WHITE]) == 1 and len(minors[chess.BLACK]) == 1 and \
            board.piece_type_at(minors[chess.WHITE][0]) != board.piece_type_at(minors[chess.BLACK][0]):
        def _minor(sq):
            return (f"{_colour_word(_is_light(sq))}-squared bishop" if board.piece_type_at(sq) == chess.BISHOP else "knight")
        out.append((3, f"it is White's {_minor(minors[chess.WHITE][0])} against Black's {_minor(minors[chess.BLACK][0])}"))
    # Uncastled king while the opponent has castled (middlegame, queens on)
    if 8 <= board.fullmove_number <= 25 and board.pieces(chess.QUEEN, chess.WHITE) and board.pieces(chess.QUEEN, chess.BLACK):
        for c in (chess.WHITE, chess.BLACK):
            k, ok = board.king(c), board.king(not c)
            if k is not None and ok is not None and chess.square_file(k) in (3, 4) and chess.square_file(ok) in (1, 2, 6, 7):
                out.append((3, f"{SIDE[c]}'s king is still in the centre, while {SIDE[not c]} has castled"))
    # Standing pins
    for c in (chess.WHITE, chess.BLACK):
        for d in line_tactics(board, c):
            if d["type"] == "pin":
                out.append((3, _line_text(board, d)))
    # Restricted minor pieces (after the opening)
    if board.fullmove_number > 10:
        for sq, n in restricted_pieces(board):
            out.append((3, f"{_desc(board, sq, True)} has {'no safe squares' if n == 0 else 'only one safe square' if n == 1 else 'only two safe squares'}"))
    return out


def _top(items: list, n: int) -> list:
    seen, out = set(), []
    for _, text in sorted(items, key=lambda x: x[0]):
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) == n:
            break
    return out


def analyze_move(before: chess.Board, move: chess.Move, best_move: Optional[chess.Move] = None,
                 reply_move: Optional[chess.Move] = None, eval_after_cp=None, mate_after=None) -> dict:
    """Human-style insights for one move.

    before       position before the move (not modified)
    move         the move played
    best_move    engine's preferred move in `before` (to explain why it was better)
    reply_move   engine's best reply in the position after the move (first PV move)
    eval_after_cp / mate_after   engine evaluation after the move, White's POV
    """
    mover = before.turn
    after = before.copy(stack=True)
    after.push(move)

    struct = _structural_changes(before, after, mover, eval_after_cp, mate_after)
    # Mid-exchange positions produce misleading structure (e.g. "doubled pawns" that
    # vanish on the recapture). If the engine's reply recaptures on the same square,
    # describe the structure after the recapture instead.
    if reply_move is not None and reply_move in after.legal_moves and after.is_capture(reply_move) \
            and reply_move.to_square == move.to_square:
        after_reply = after.copy(stack=True)
        rsan = after_reply.san(reply_move)
        after_reply.push(reply_move)
        struct = [(pr, f"after the recapture {rsan}, {t}")
                  for pr, t in _structural_changes(before, after_reply, mover, eval_after_cp, mate_after)]
        if before.is_capture(move):
            struct.append((2, f"{before.san(move)} starts an exchange: the expected recapture {rsan} "
                              f"restores the material balance, so no material is won"))
    insights = move_tactics(before, move) + struct
    changed = _top(insights, MAX_MOVE_INSIGHTS)
    context = [t for t in _top(position_context(after, eval_after_cp, mate_after), MAX_CONTEXT + len(changed))
               if not any(t.split(" (")[0] in c for c in changed)][:MAX_CONTEXT]

    result = {"move_ideas": changed, "position_context": context, "best_move_ideas": [], "opponent_reply_ideas": []}
    if best_move is not None and best_move != move and best_move in before.legal_moves:
        result["best_move_ideas"] = _top(move_tactics(before, best_move), 3)
    if reply_move is not None and reply_move in after.legal_moves:
        result["opponent_reply_ideas"] = _top(move_tactics(after, reply_move), 3)
    return result


def insights_for_move_record(m: dict) -> dict:
    """analyze_move() for one per-move dict produced by features.py (needs
    fen_before, uci, best_move_san, principal_variation_san, eval_after_cp, mate_after)."""
    before = chess.Board(m["fen_before"])
    move = chess.Move.from_uci(m["uci"])
    best = None
    if m.get("best_move_san"):
        try:
            best = before.parse_san(m["best_move_san"])
        except ValueError:
            pass
    after = before.copy()
    after.push(move)
    reply = None
    if m.get("principal_variation_san"):
        try:
            reply = after.parse_san(m["principal_variation_san"][0])
        except ValueError:
            pass
    return analyze_move(before, move, best, reply, m.get("eval_after_cp"), m.get("mate_after"))

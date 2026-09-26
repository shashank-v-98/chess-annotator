"""Structured feature extraction for chess games.

This module turns a PGN into a list of per-move feature dictionaries using
python-chess plus a position-evaluation source. No language model is
involved here -- everything produced is a deterministic, verifiable fact
about the position. The LLM layer (prompts.py / llm_backends.py) only ever
*explains* these facts; it never has to derive them.

Two evaluation sources are supported (engine_mode):
  - "local"  (default): a local Stockfish (or any UCI engine) binary via
             python-chess's chess.engine module. Fast, unlimited, full
             multi-move principal variations. Needs a Stockfish binary
             installed and on PATH (or pass engine_path explicitly).
  - "online": no local install at all -- queries the free community
             Stockfish server at chess-api.com over HTTPS for each
             position. Useful as a zero-install fallback while getting the
             pipeline wired up, or if a local Stockfish install hits
             trouble. It's a shared free server, so only use it for light,
             interactive use (a handful of games) -- switch to "local" for
             batches. PV is limited to the single best move in this mode.

Requires:
    pip install python-chess
    (local mode) a Stockfish binary on PATH, or pass engine_path
    (online mode) pip install requests (already in requirements.txt)
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import chess
import chess.engine
import chess.pgn

# Centipawn-loss thresholds for move-quality labels, from the mover's point
# of view. These mirror the conventions used by lichess/chess.com review
# tools so the labels are already familiar to any chess player.
BLUNDER_CP = 200
MISTAKE_CP = 100
INACCURACY_CP = 50
MATE_SCORE = 100000  # internal sentinel for "mate in N", sign-adjusted

ONLINE_API_URL = "https://chess-api.com/v1"


@dataclass
class MoveFeatures:
    ply: int
    move_number: int
    side: str  # "white" | "black"
    san: str
    uci: str
    fen_before: str
    fen_after: str
    phase: str  # "opening" | "middlegame" | "endgame"
    is_capture: bool
    is_check: bool
    is_castle: bool
    material_before: int  # positive = white ahead, in pawns
    material_after: int
    eval_before_cp: Optional[int]  # from white's POV, None if mate score dominates
    eval_after_cp: Optional[int]
    mate_before: Optional[int]
    mate_after: Optional[int]
    cp_loss: int  # centipawn loss from the mover's perspective, >=0
    quality_label: str  # "best" | "excellent" | "good" | "inaccuracy" | "mistake" | "blunder"
    nag_symbol: str  # "!!" "!" "" "?!" "?" "??"
    is_best_move: bool
    best_move_san: Optional[str]
    principal_variation_san: list = field(default_factory=list)
    gives_up_mate: bool = False
    missed_mate: bool = False
    tags: list = field(default_factory=list)  # lightweight heuristic tags
    captured_piece: Optional[str] = None  # e.g. "knight", "pawn" -- None if not a capture
    engine_ok: bool = True  # False if the engine returned no evaluation for the position before or after
                            # this move; cp_loss / quality_label are then meaningless and callers should skip it


def _material_balance(board: chess.Board) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for piece_type, val in values.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * val
        total -= len(board.pieces(piece_type, chess.BLACK)) * val
    return total


def _phase(board: chess.Board, move_number: int) -> str:
    piece_count = len(board.piece_map())
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    if move_number <= 10 and piece_count >= 28:
        return "opening"
    if queens == 0 or piece_count <= 12:
        return "endgame"
    return "middlegame"


def _quality_label(cp_loss: int, was_forced_mate_missed: bool) -> tuple:
    if was_forced_mate_missed:
        return "mistake", "?"
    if cp_loss <= 0:
        return "best", "!!" if cp_loss == 0 else ""
    if cp_loss < 10:
        return "excellent", ""
    if cp_loss < INACCURACY_CP:
        return "good", ""
    if cp_loss < MISTAKE_CP:
        return "inaccuracy", "?!"
    if cp_loss < BLUNDER_CP:
        return "mistake", "?"
    return "blunder", "??"


def _captured_piece_name(board: chess.Board, move: chess.Move) -> Optional[str]:
    """Name of the piece a move captures, or None if it's not a capture.
    Must be called BEFORE board.push(move) -- for a normal capture the
    piece is still sitting on the destination square; en passant is a
    special case where the captured pawn is on a different square than
    the move's destination."""
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        return "pawn"  # en passant only ever captures a pawn
    piece = board.piece_at(move.to_square)
    return chess.piece_name(piece.piece_type) if piece else None


def _simple_fork_tag(board_after: chess.Board, moved_to: chess.Square, mover_is_white: bool) -> bool:
    """Very lightweight fork heuristic: does the piece that just moved attack
    two or more enemy pieces worth more than a pawn? Not a full tactic
    detector -- a lightweight, explainable signal that's cheap to compute
    and cheap for the LLM to describe."""
    piece = board_after.piece_at(moved_to)
    if piece is None:
        return False
    attacked = board_after.attacks(moved_to)
    valuable_targets = 0
    for sq in attacked:
        target = board_after.piece_at(sq)
        if target and target.color != piece.color and target.piece_type != chess.PAWN:
            valuable_targets += 1
    return valuable_targets >= 2


class _PositionAnalyser:
    """Uniform interface over a local UCI engine or the online API.

    analyse(board) -> dict with keys:
        cp (white POV, or None if mate), mate (white POV, or None),
        best_move (chess.Move or None), pv (list[chess.Move])
    """

    def analyse(self, board: chess.Board) -> dict:
        raise NotImplementedError

    def close(self):
        pass


class _LocalEngineAnalyser(_PositionAnalyser):
    def __init__(self, engine_path: str, depth: int, multipv: int = 1):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth
        self.multipv = multipv

    def analyse(self, board: chess.Board) -> dict:
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multipv)
        info = info[0] if isinstance(info, list) else info
        white_score = info["score"].white()
        cp = None if white_score.is_mate() else white_score.score()
        mate = white_score.mate() if white_score.is_mate() else None
        pv = info.get("pv", [])
        best_move = pv[0] if pv else None
        return {"cp": cp, "mate": mate, "best_move": best_move, "pv": pv[:4]}

    def close(self):
        self.engine.close()


class _OnlineAPIAnalyser(_PositionAnalyser):
    """Queries the free chess-api.com community Stockfish server.

    No local Stockfish binary needed. Be a good citizen of this shared free
    service: this class adds a short delay between requests and is meant
    for light/interactive use, not bulk analysis.
    """

    def __init__(self, depth: int = 12, min_interval_s: float = 1.0, timeout: int = 20):
        import requests
        self._requests = requests
        self.depth = depth
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    MAX_ATTEMPTS = 5
    _warned = 0

    def analyse(self, board: chess.Board) -> dict:
        # The free server sometimes answers without an evaluation (rate limiting or
        # overload) instead of returning an HTTP error. Without a retry those positions
        # would silently score as 0 and every affected move would be labelled "best".
        if board.is_game_over():
            return self._analyse_once(board)  # no eval exists for a finished game; don't retry
        result = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                result = self._analyse_once(board)
            except Exception as e:  # network error / HTTP error
                result = {"cp": None, "mate": None, "best_move": None, "pv": [], "_error": str(e)}
            if result["cp"] is not None or result["mate"] is not None:
                return result
            time.sleep(2 * (2 ** attempt))  # 2, 4, 8, 16, 32s
        if _OnlineAPIAnalyser._warned < 5:
            _OnlineAPIAnalyser._warned += 1
            print(f"    [engine] no evaluation after {self.MAX_ATTEMPTS} attempts; "
                  f"last response: {result.get('_error') or result.get('_raw', '')[:200]}")
        return result

    def _analyse_once(self, board: chess.Board) -> dict:
        self._throttle()
        resp = self._requests.post(
            ONLINE_API_URL,
            json={"fen": board.fen(), "depth": self.depth},
            timeout=self.timeout,
        )
        self._last_call = time.time()
        resp.raise_for_status()
        data = resp.json()

        mate = data.get("mate")
        cp = data.get("centipawns")
        if mate is not None:
            try:
                mate = int(mate)
            except (TypeError, ValueError):
                mate = None  # unparseable -- treat as "no mate info" rather than crash downstream
        if mate is not None:
            cp = None  # mate takes priority, matches local-engine shape
        elif cp is not None:
            try:
                cp = int(cp)
            except (TypeError, ValueError):
                cp = None

        best_move = None
        pv = []
        best_uci = data.get("move")
        if best_uci:
            try:
                best_move = chess.Move.from_uci(best_uci)
                pv = [best_move]  # the free API only gives the top move, not a full PV
            except ValueError:
                pass

        return {"cp": cp, "mate": mate, "best_move": best_move, "pv": pv, "_raw": json.dumps(data)}

    def close(self):
        pass


def _make_analyser(engine_mode: str, engine_path: str, depth: int, multipv: int) -> _PositionAnalyser:
    if engine_mode == "local":
        return _LocalEngineAnalyser(engine_path, depth, multipv)
    if engine_mode == "online":
        return _OnlineAPIAnalyser(depth=min(depth, 15))  # community server: keep depth modest
    raise ValueError(f"Unknown engine_mode: {engine_mode!r} (expected 'local' or 'online')")


def analyze_game(
    pgn_text: str,
    engine_mode: str = "local",
    engine_path: str = "stockfish",
    depth: int = 16,
    multipv: int = 1,
    max_plies: Optional[int] = None,
) -> dict:
    """Analyze a single-game PGN text move by move.

    engine_mode: "local" (default, needs a Stockfish binary) or "online"
                 (zero-install, uses the free chess-api.com server; see
                 module docstring for etiquette notes).

    Returns a dict:
        {
          "headers": {...pgn headers...},
          "moves": [MoveFeatures-as-dict, ...],
        }
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN text")

    analyser = _make_analyser(engine_mode, engine_path, depth, multipv)
    try:
        return _analyze_parsed_game(game, analyser, max_plies)
    finally:
        analyser.close()


def iter_games_in_pgn(pgn_text: str):
    """Yield each chess.pgn.Game found in a (possibly multi-game) PGN text,
    e.g. a bulk export from the Lichess API."""
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game


def analyze_games_in_pgn(
    pgn_text: str,
    engine_mode: str = "local",
    engine_path: str = "stockfish",
    depth: int = 16,
    multipv: int = 1,
    max_plies: Optional[int] = None,
):
    """Like analyze_game, but for a PGN text containing multiple games (e.g.
    a bulk Lichess export). Reuses a single analyser (engine process, or
    online-API throttle state) across all games for efficiency. Yields one
    analysis dict per game, in order. Games that fail to analyze (rare
    parsing edge cases, e.g. games with no moves) are skipped with a
    printed warning rather than aborting the whole batch."""
    analyser = _make_analyser(engine_mode, engine_path, depth, multipv)
    try:
        for i, game in enumerate(iter_games_in_pgn(pgn_text)):
            try:
                yield _analyze_parsed_game(game, analyser, max_plies)
            except Exception as e:
                white = game.headers.get("White", "?")
                black = game.headers.get("Black", "?")
                print(f"  [skip] game {i} ({white} vs {black}): {e}")
    finally:
        analyser.close()


def _analyze_parsed_game(game: "chess.pgn.Game", analyser: _PositionAnalyser, max_plies: Optional[int] = None,
                         on_move=None) -> dict:
    """Core analysis loop, shared by analyze_game and analyze_games_in_pgn.
    Does NOT own the analyser's lifecycle -- callers are responsible for
    closing it (so it can be reused across many games)."""
    headers = dict(game.headers)
    board = game.board()

    moves_out = []
    prev_info = analyser.analyse(board)
    cp_before, mate_before, best_before = prev_info["cp"], prev_info["mate"], prev_info["best_move"]

    for ply, move in enumerate(game.mainline_moves(), start=1):
        if max_plies and ply > max_plies:
            break

        side_white = board.turn == chess.WHITE
        move_number = board.fullmove_number
        san = board.san(move)
        uci = move.uci()
        fen_before = board.fen()
        is_capture = board.is_capture(move)
        captured_piece = _captured_piece_name(board, move)  # must compute before push()
        best_move_san = board.san(best_before) if best_before else None
        is_best_move = (move == best_before)

        material_before = _material_balance(board)
        board.push(move)
        material_after = _material_balance(board)
        is_check = board.is_check()
        is_castle = board.is_castling(move) if hasattr(board, "is_castling") else False
        fen_after = board.fen()
        phase = _phase(board, move_number)

        info_after = analyser.analyse(board)
        cp_after, mate_after, best_after = info_after["cp"], info_after["mate"], info_after["best_move"]

        pv_san = []
        tmp_board = board.copy()
        for pv_move in info_after["pv"]:
            try:
                pv_san.append(tmp_board.san(pv_move))
                tmp_board.push(pv_move)
            except Exception:
                break

        # cp loss from the mover's perspective (mover = side_white before this move)
        def pov_cp(cp, mate, is_white_pov):
            if mate is not None:
                val = MATE_SCORE - abs(mate) * 100
                val = val if mate > 0 else -val
                return val if is_white_pov else -val
            if cp is None:
                return 0
            return cp if is_white_pov else -cp

        # A checkmate/stalemate on the board legitimately has no engine score; anything
        # else missing means the engine call failed, and cp_loss below would be fake.
        engine_ok = ((cp_before is not None or mate_before is not None)
                     and (cp_after is not None or mate_after is not None or board.is_game_over()))

        score_before_mover = pov_cp(cp_before, mate_before, side_white)
        score_after_mover = pov_cp(cp_after, mate_after, side_white)
        cp_loss = max(0, score_before_mover - score_after_mover)

        missed_mate = bool(mate_before and side_white == (mate_before > 0) and not (mate_after and side_white == (mate_after > 0)))
        gives_up_mate = missed_mate

        quality_label, nag_symbol = _quality_label(cp_loss, missed_mate)

        tags = []
        if is_capture:
            tags.append("capture")
        if is_check:
            tags.append("check")
        if is_castle:
            tags.append("castle")
        if _simple_fork_tag(board, move.to_square, side_white):
            tags.append("possible_fork")

        moves_out.append(asdict(MoveFeatures(
            ply=ply,
            move_number=move_number,
            side="white" if side_white else "black",
            san=san,
            uci=uci,
            fen_before=fen_before,
            fen_after=fen_after,
            phase=phase,
            is_capture=is_capture,
            is_check=is_check,
            is_castle=bool(is_castle),
            material_before=material_before,
            material_after=material_after,
            eval_before_cp=cp_before,
            eval_after_cp=cp_after,
            mate_before=mate_before,
            mate_after=mate_after,
            cp_loss=cp_loss,
            quality_label=quality_label,
            nag_symbol=nag_symbol,
            is_best_move=is_best_move,
            best_move_san=best_move_san,
            principal_variation_san=pv_san,
            gives_up_mate=gives_up_mate,
            missed_mate=missed_mate,
            tags=tags,
            captured_piece=captured_piece,
            engine_ok=engine_ok,
        )))

        if on_move is not None:  # streaming hook (used by the web app); may raise to abort
            on_move(moves_out[-1])

        # roll forward for next iteration
        cp_before, mate_before, best_before = cp_after, mate_after, best_after

    return {"headers": headers, "moves": moves_out}

"""Automatic fact-consistency checks for a move annotation.

check(facts, text) compares an annotation against the facts JSON it was
written from (the record built by chess_annotator.app_prompts) and returns a
list of violations -- claims the facts don't support. Used to:
  * filter teacher labels before training (drop annotations with violations),
  * score a student model (share of annotations with zero violations).

The checks are deliberately conservative (few false alarms) rather than
exhaustive: they catch wrong advantage sizes, wrong side, wrong direction,
tactics/structures not in the insights, captures and checks that didn't
happen, and pieces named on squares where they don't stand. Sentences about
hypothetical engine lines ("would", "could", the best move) are skipped for
the direction/capture/check rules.

Pure Python (no python-chess), so it also runs inside the Kaggle notebook.
"""

from __future__ import annotations

import re

PIECES = ("king", "queen", "rook", "bishop", "knight", "pawn")

# phrase regex -> substring that must appear in the insights text for the phrase to be allowed
TACTIC_TERMS = [
    (r"\bpin(s|ned|ning)?\b", ("pin",)),
    (r"\bfork(s|ed|ing)?\b", ("fork",)),
    (r"\bskewer", ("skewer",)),
    (r"\bdiscovered\b", ("discover",)),
    (r"\bdouble check\b", ("double check",)),
    (r"threat(ens|ening)? (of )?mate|mating threat|mate threat", ("mate",)),
    (r"\bsemi-?open\b|\bhalf-?open\b", ("semi-open",)),
    (r"\bopen [a-h]-file\b|\bopen file\b", ("open",)),
    (r"\bhanging pawns?\b", ("hanging pawns",)),
    (r"\bisolated\b|\bisolani\b", ("isolated",)),
    (r"\bpassed pawn|\bpasser\b", ("passed pawn",)),
    (r"\bdoubled pawns?\b", ("doubled pawns",)),
    (r"\bbishop pair\b|\btwo bishops\b", ("bishop pair",)),
    (r"\bopposite[- ]colou?red\b", ("opposite",)),
    (r"\bbad bishop\b", ("'bad' bishop",)),
    (r"\bpawn majority\b", ("majority",)),
    (r"\ben prise\b|\bundefended\b|\bhangs\b|\bleft hanging\b", ("en prise", "not adequately defended", "can itself be captured")),
    (r"\boutpost\b|\bweak square\b|\bhole\b", ()),  # never produced by the detectors
]

WIN_WORDS = re.compile(r"\b(decisive|winning(?! (?:a|the|that|this|his|her|back|an?)\b)|crushing|overwhelming|won position)\b", re.I)
EQUAL_WORDS = re.compile(r"\b(roughly|essentially|about|dead) (equal|level|balanced)\b|\bequal position\b", re.I)
SIDE_BETTER = re.compile(
    r"\b(white|black)(?:['’]s)? (?:is|remains|stands|stays|was) (?:still |now |clearly |much |slightly |comfortably )?"
    r"(?:better|ahead|winning|on top)\b|\badvantage (?:for|to) (white|black)\b", re.I)
WORSEN = re.compile(r"\b(worsen(s|ed|ing)?|costs?|drops?|loses?|slips?|gives? away|squanders?)\b", re.I)
IMPROVE = re.compile(r"\b(improv(es|ed|ing)|gains?|increas(es|ed|ing)|raises?|boosts?|strengthens?)\b", re.I)
HYPOTHETICAL = re.compile(r"\b(would|could|can|cannot|can't|might|instead|should|if|not|without)\b", re.I)
CAPTURE = re.compile(r"\b(captures?|capturing|takes|grabs?|snaps? (off|up)|wins (a|the) (pawn|knight|bishop|rook|queen))\b", re.I)
CHECK = re.compile(r"\b(gives? check|with check|delivers? check|delivering check|checks the king)\b", re.I)
PIECE_ON = re.compile(r"\b(king|queen|rook|bishop|knight|pawn)s? on ([a-h][1-8])\b", re.I)
MATERIAL_WIN = re.compile(r"\b(wins?|winning|gains?|nets?) (a |the )?(pawn|piece|material|exchange)\b|\bmaterial gain\b", re.I)
PLURAL = re.compile(r"\b(both|two|coordinat\w*|connect\w*) (?:the |his |her |white['’]s |black['’]s )?(bishops|knights|rooks)\b", re.I)
FORBIDDEN = re.compile(r"\bcentipawns?\b|\bcp\b", re.I)


def normalize(text: str) -> str:
    """Models often write non-breaking or typographic hyphens/dashes/quotes; normalise them."""
    for ch in "\u2010\u2011\u2012\u2013\u2212":
        text = text.replace(ch, "-")
    return text.replace("\u2019", "'").replace("\u00a0", " ")


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _insights_text(facts: dict) -> str:
    parts = []
    for k in ("move_ideas", "position_context", "best_move_ideas", "opponent_reply_ideas"):
        parts.extend(facts.get(k) or [])
    return " ".join(parts).lower()


def _abs_eval(block: dict):
    if not block:
        return None
    if block.get("mate_in") is not None:
        return 10000
    return abs(block["cp"]) if block.get("cp") is not None else None


def _sign(block: dict):
    if not block:
        return 0
    if block.get("mate_in") is not None:
        return 1 if block["mate_in"] > 0 else -1
    cp = block.get("cp")
    if cp is None or abs(cp) < 50:
        return 0
    return 1 if cp > 0 else -1


def _pieces(facts: dict) -> set:
    out = set()
    for side, desc in (facts.get("pieces_after") or {}).items():
        for part in desc.split(";"):
            if ":" not in part:
                continue
            name, squares = part.split(":", 1)
            name = name.strip().rstrip("s")
            for sq in squares.split(","):
                out.add((name, sq.strip()))
    return out


def check(facts: dict, text: str) -> list:
    """Return a list of human-readable violations (empty list = consistent)."""
    v = []
    if not text or not text.strip():
        return ["empty annotation"]
    text = normalize(text)
    if not text.rstrip().endswith((".", "!", "?", '"', "'", ")")):
        v.append("annotation is cut off mid-sentence")
    low = text.lower()
    ins = _insights_text(facts)

    if FORBIDDEN.search(text):
        v.append("uses 'centipawn'/'cp'")

    # Size of advantage
    before, after = _abs_eval(facts.get("eval_before")), _abs_eval(facts.get("eval_after"))
    if WIN_WORDS.search(text) and not any(x is not None and x > 300 for x in (before, after)):
        v.append(f"calls the position decisive/winning at {facts.get('eval_after', {}).get('pawns')}")
    if EQUAL_WORDS.search(text) and after is not None and after > 100 and (before is None or before > 100):
        v.append(f"calls the position equal at {facts.get('eval_after', {}).get('pawns')}")

    # Which side is better
    signs = {_sign(facts.get("eval_before")), _sign(facts.get("eval_after"))} - {0}
    for m in SIDE_BETTER.finditer(text):
        if re.match(r"\s*(in|on) material", text[m.end():m.end() + 14], re.I):
            continue  # "ahead in material" is about material, not the evaluation
        side = (m.group(1) or m.group(2)).lower()
        claimed = 1 if side == "white" else -1
        if signs and claimed not in signs:
            v.append(f"says {side.title()} is better, but the engine favours the other side")

    # Direction of the move's effect, captures, checks -- only in factual sentences
    change = (facts.get("eval_change_for_mover") or {}).get("cp")
    best = (facts.get("best_move_san") or "").lower()
    is_capture = bool(facts.get("captured_piece")) or "capture" in (facts.get("tags") or [])
    gives_check = "check" in (facts.get("tags") or [])
    for s in _sentences(text):
        sl = s.lower()
        if HYPOTHETICAL.search(s) or (best and best != facts.get("san", "").lower() and best in sl):
            continue
        if change is not None:
            if change >= 40 and WORSEN.search(s) and not IMPROVE.search(s):
                v.append(f"says the move worsened the position, but it gained {change / 100:.2f}")
            if change <= -40 and IMPROVE.search(s) and not WORSEN.search(s) and "evaluation" in sl:
                v.append(f"says the move improved the evaluation, but it lost {-change / 100:.2f}")
        if CAPTURE.search(s) and not is_capture and "exchange" not in ins and "captures" not in ins:
            v.append("describes a capture that did not happen")
        if CHECK.search(s) and not gives_check:
            v.append("describes a check that did not happen")

    # Material won in what is really an exchange
    if "no material is won" in ins and any(MATERIAL_WIN.search(x) and not HYPOTHETICAL.search(x) for x in _sentences(text)):
        v.append("claims material is won, but the capture is an exchange")

    # Plural pieces the mover doesn't have ("coordinate the bishops" with one bishop)
    pieces_desc = (facts.get("pieces_after") or {}).get("White" if facts.get("side") == "white" else "Black", "")
    if pieces_desc:
        for m in PLURAL.finditer(text):
            kind = m.group(2).lower()[:-1]
            n = next((len(part.split(":", 1)[1].split(",")) for part in pieces_desc.split(";")
                      if ":" in part and part.split(":", 1)[0].strip().rstrip("s") == kind), 0)
            if n < 2:
                v.append(f"mentions two {kind}s, but the mover has {n}")

    # Tactical / structural terms must be backed by the insights
    for pattern, needs in TACTIC_TERMS:
        m = re.search(pattern, low)
        if m and not any(n in ins for n in needs):
            v.append(f"mentions '{m.group(0)}' which is not in the insights")

    # Pieces named on squares must exist there (after the move) or be named in the insights
    pieces = _pieces(facts)
    if pieces:
        from_sq = (facts.get("uci") or "")[:2]
        for m in PIECE_ON.finditer(text):
            name, sq = m.group(1).lower(), m.group(2).lower()
            if (name, sq) in pieces or f"{name} on {sq}" in ins or sq == from_sq:
                continue
            v.append(f"names a {name} on {sq}, but there is none")
    return sorted(set(v))

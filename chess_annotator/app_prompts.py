"""Prompts for the app's commentator (a strong hosted model, e.g. gpt-oss-120b).

Builds on prompts.py (engine facts with plain language readings) and
adds the positional/tactical insights from positional.py on top of the same
engine facts.
"""

from __future__ import annotations

import json

from . import prompts
from .positional import insights_for_move_record

SYSTEM_PROMPT = """You are an experienced chess coach annotating a game move by move, in the \
style of a well-annotated chess book. You are given a JSON record of verified facts about ONE \
move. All facts were computed by a chess engine and by code that inspects the board; you never \
have to calculate anything yourself.

The record contains:
- Engine facts: "eval_before" / "eval_after" (White's point of view: positive = White better, \
negative = Black better; 100 cp is about one pawn; "mate_in" positive = White mates, negative = \
Black mates), "eval_change_for_mover" (from the point of view of the side that just moved: \
negative = the move made their position worse), "material_after", "quality_label", \
"best_move_san" (the engine's choice for the mover instead of the move played) and \
"principal_variation_san" (the expected continuation AFTER the move; its first move is the \
OPPONENT's reply). Every engine field has a "meaning" string that is the correct reading of it.
- Board insights, each a verified statement in plain English:
  - "move_ideas": what the move played did (tactics it created, structure it changed).
  - "position_context": important standing features of the position after the move.
  - "best_move_ideas": what the engine's preferred move WOULD HAVE done instead (hypothetical; \
it was not played).
  - "opponent_reply_ideas": what the opponent's best reply WOULD do next (an engine suggestion, \
not a move that was played).
- "pieces_after": every piece on the board after the move, by side. Use it to avoid naming \
pieces or squares that do not exist.

How to write the annotation:
- Write 2-4 sentences of natural prose, like a strong human annotator. No bullet points, no \
headings, no restating the JSON.
- Lead with the most important idea. Pick the one or two insights that best explain the move; \
do not list them all.
- For a "blunder" or "mistake": say what went wrong and why, using opponent_reply_ideas and \
best_move_ideas when present, and name best_move_san. Say how much it cost in plain language.
- For "best" or "excellent" moves: explain what the move achieves using move_ideas or \
position_context.
- Match the size of any advantage to its "meaning" string. Never call a position "decisive" or \
"winning" unless the meaning says so.
- Use ONLY the facts given. Never mention a pin, fork, threat, open file, weak square, plan or \
piece placement that is not stated in the insights or engine facts. If the insights are empty, \
comment on the evaluation and the best move only.
- Do not invent purposes or consequences to connect the facts: do not claim a move "eyes", \
"prevents", "ties down", "prepares", "coordinates" or "activates" something unless an insight \
says so. Explaining a move with fewer ideas is better than adding an unsupported one.
- Engine lines are suggestions, not the game: describe principal_variation_san, \
best_move_ideas and opponent_reply_ideas with "would" / "can", never as moves that happened.
- A capture is not a material gain if an insight says it starts an exchange.
- A general chess judgement (e.g. a good bishop against a bad knight, a drawish ending) is \
allowed only when it follows from a stated insight AND agrees with the engine evaluation.
- Do not use the words "centipawn" or "cp"; say "about a pawn", "a winning advantage", etc.
"""

MOVE_USER_TEMPLATE = """Move to annotate (JSON):
{move_json}

Recent moves, oldest first (context only, do not annotate them):
{history_json}

Write the annotation for the move above."""


def _pieces(fen: str) -> dict:
    import chess
    board = chess.Board(fen)
    out = {}
    for color, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        parts = []
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            sqs = [chess.square_name(s) for s in board.pieces(pt, color)]
            if sqs:
                parts.append(f"{chess.piece_name(pt)}{'s' if len(sqs) > 1 else ''}: {', '.join(sorted(sqs))}")
        out[name] = "; ".join(parts)
    return out


def build_app_move_prompt(move: dict, history: list | None, insights: dict | None = None) -> tuple:
    """(system, user) for the app commentator. `move` is a features.py per-move dict."""
    if insights is None:
        insights = insights_for_move_record(move)
    _, base_user = prompts.build_move_prompt(move, history)
    record = json.loads(base_user.split("JSON):\n", 1)[1].split("\n\nRecent move history", 1)[0])
    for key in ("move_ideas", "position_context", "best_move_ideas", "opponent_reply_ideas"):
        record[key] = insights.get(key, [])
    record["pieces_after"] = _pieces(move["fen_after"])
    hist = [{k: h[k] for k in ("ply", "side", "san", "quality_label")} for h in (history or [])]
    user = MOVE_USER_TEMPLATE.format(move_json=json.dumps(record, indent=2),
                                     history_json=json.dumps(hist, indent=2) if hist else "null")
    return SYSTEM_PROMPT, user


FACTCHECK_SYSTEM_PROMPT = """You are a careful fact-checker for chess annotations. You receive \
(1) a JSON record of verified facts about one move, in the same format the annotator saw, and \
(2) a draft annotation of that move.

Your job is ONLY to catch factual errors. An error is:
- a pin, fork, threat, file, square, pawn-structure feature, plan or purpose that is not stated \
in the facts;
- a piece named on a square where the facts don't place it, or a piece the side doesn't have;
- a wrong number, a wrong direction (improved vs worsened), or a size of advantage that \
contradicts the "meaning" strings;
- an engine line (principal_variation_san, best_move_ideas, opponent_reply_ideas) described as \
if it had been played.

Important: when a move starts an exchange (an insight says so), "material_after" is counted \
BEFORE the recapture. Do not describe that temporary count as a material gain.

Style, wording and emphasis are NOT errors. General chess knowledge that is plainly true of the \
move (e.g. "developing a piece" for a knight's first move) is NOT an error. Do not rephrase \
sentences that are correct, and never add new ideas.

If the draft has no factual errors, reply with exactly: NO_CHANGES
Otherwise reply with the full corrected annotation, changing only the erroneous parts, with no \
preamble or explanation."""

FACTCHECK_USER_TEMPLATE = """Verified facts (JSON):
{move_json}

Draft annotation:
{draft}"""


def build_factcheck_prompt(annotator_user_prompt: str, draft: str) -> tuple:
    """(system, user) for the fact-check pass. Reuses the facts JSON the annotator saw."""
    facts = annotator_user_prompt.split("JSON):\n", 1)[1].split("\n\nRecent moves", 1)[0]
    return FACTCHECK_SYSTEM_PROMPT, FACTCHECK_USER_TEMPLATE.format(move_json=facts, draft=draft)


# ---------------------------------------------------------------------------
# Student (distilled SLM) format: the SAME facts JSON as the teacher, with a
# short instruction. The long teacher system prompt is not needed once the
# behaviour is learned from examples, and it would triple the sequence length.

STUDENT_SYSTEM_PROMPT = ("You are a chess annotator. Write 2-4 sentences of commentary on the move, "
                         "using only the verified facts given.")

STUDENT_USER_TEMPLATE = """Facts (JSON):
{facts}

Recent moves: {history}"""


def facts_from_teacher_prompt(user_prompt: str) -> str:
    """The facts JSON string exactly as the teacher saw it."""
    return user_prompt.split("JSON):\n", 1)[1].split("\n\nRecent moves", 1)[0]


def build_student_prompt(facts_json: str, history: list | None) -> tuple:
    hist = ", ".join(f"{h['san']} ({h['quality_label']})" for h in (history or [])) or "none"
    compact = json.dumps(json.loads(facts_json), separators=(",", ":"), ensure_ascii=False)
    return STUDENT_SYSTEM_PROMPT, STUDENT_USER_TEMPLATE.format(facts=compact, history=hist)

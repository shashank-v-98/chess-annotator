"""Prompt templates that condition the SLM on structured move features.

The model is deliberately never asked to evaluate the position itself -- all
numeric judgment (centipawn loss, best move, mate detection) already happened
in features.py. The model's only task is controlled natural-language
generation: turn a small structured record into 1-3 fluent sentences a human
commentator might write.

Engine numbers are passed in two forms side by side: the exact value (in
centipawns and pawns) and a deterministic plain-language reading of it
("White is clearly better"). Language models tend to misread raw centipawns -- calling +29 -> +40 "a decisive edge", or getting
the sign backwards on Black's moves -- so the interpretation is computed
here, in code, and the model only has to put it into words.

This file is also embedded verbatim into distill/train_on_kaggle.ipynb by
distill/_build_notebook.py, so it must stay self-contained (stdlib only).
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You are an experienced chess coach writing concise annotations for a game, \
in the style of a well-annotated chess book (e.g. My 60 Memorable Games). \
You will be given a JSON record of facts about ONE move: the move played, \
the engine evaluation before and after the move, how much the move changed \
the evaluation for the side that played it, the engine's preferred move and \
short principal variation, material balance, game phase, a computed quality \
label, what piece (if any) was captured, and a few tags.

How to read the engine numbers:
- "eval_before" and "eval_after" are from WHITE's point of view. A positive \
"cp" means White is better, a negative "cp" means Black is better, and 0 \
means the position is equal. 100 cp is about the value of one pawn, so \
+40 is a small edge for White, -150 is Black about a pawn and a half better, \
and anything beyond +300 or -300 is usually a winning advantage. "pawns" is \
the same number divided by 100.
- If "mate_in" is set instead of "cp", the engine sees a forced mate: \
positive means White mates in that many moves, negative means Black does.
- "eval_change_for_mover" is from the point of view of the side that JUST \
MOVED, not White. A negative "cp" means the move made that side's position \
worse by that amount; a positive "cp" means it improved it. Near zero means \
the move kept the evaluation where it was.
- "material_after" is the material count after the move, in pawns, from \
White's point of view (pawn=1, knight/bishop=3, rook=5, queen=9).
- "best_move_san" is the engine's choice for the side that just moved, in \
the position BEFORE the move. "principal_variation_san" is the engine's \
expected continuation AFTER the move: its first move is the OPPONENT's best \
reply, and the moves then alternate sides. Do not describe the PV as the \
mover's own plan.
- Every one of these fields has a "meaning" string. It is the correct \
reading of the numbers. Whatever you say about who stands better, by how \
much, and whether the move helped or hurt must agree with it.

Rules:
- Use ONLY the facts given. Never invent tactics, threats, plans, open \
files, weak squares, or defenders that are not present in the JSON. You may \
mention moves from principal_variation_san by name.
- Match the size of the advantage to its "meaning": do not call an edge \
"decisive" or "winning" unless the meaning says so, and do not call a small \
change a "swing".
- If captured_piece is set, you may say what was captured -- do not guess \
or invent a piece type if captured_piece is null, even if "tags" contains "capture".
- Write 1-3 sentences, natural and specific, not generic ("a solid move").
- If quality_label is "blunder" or "mistake", say what the better move was \
(best_move_san) and how much the played move cost, using eval_change_for_mover.
- If quality_label is "best" or "excellent", say briefly why the move is \
strong, referencing tags (capture/check/possible_fork) or the PV if useful.
- Do not restate the raw JSON or use bullet points. Plain prose only.
- Do not use the words "centipawn" or "cp" -- translate numbers into plain \
chess language (e.g. "roughly a pawn ahead", "a winning advantage").
"""

MOVE_USER_TEMPLATE = """Move to annotate (JSON):
{move_json}

Recent move history, oldest first (for continuity only, do not re-annotate these):
{history_json}

Write the annotation for the move above."""

# How many preceding moves to include as history context in build_move_prompt.
# Callers that assemble the history list (annotate.py, generate_dataset.py)
# should keep it to this many entries.
HISTORY_LEN = 2


# --- Deterministic readings of engine numbers --------------------------------
# Thresholds are in centipawns. They are deliberately coarse; the exact number
# always travels alongside the words, so nothing is lost by rounding here.

def _pawns(cp: int) -> float:
    return round(cp / 100, 2)


def _signed_pawns(cp: int) -> str:
    return f"{_pawns(cp):+.2f}"


def _advantage_meaning(cp: int) -> str:
    """cp from White's POV -> who stands better and by roughly how much."""
    side = "White" if cp > 0 else "Black"
    a = abs(cp)
    if a <= 20:
        return f"The position is roughly equal ({_signed_pawns(cp)})."
    if a <= 60:
        return f"{side} is slightly better ({_signed_pawns(cp)}, less than a pawn)."
    if a <= 150:
        return f"{side} is clearly better ({_signed_pawns(cp)}, about a pawn's worth)."
    if a <= 300:
        return f"{side} is much better ({_signed_pawns(cp)}, well over a pawn)."
    return f"{side} is winning ({_signed_pawns(cp)}, a decisive advantage)."


def _eval_block(cp, mate) -> dict:
    if mate is not None:
        winner = "White" if mate > 0 else "Black"
        return {"cp": None, "pawns": None, "mate_in": mate,
                "meaning": f"{winner} has a forced mate in {abs(mate)}."}
    if cp is None:
        return {"cp": None, "pawns": None, "mate_in": None,
                "meaning": "No engine evaluation available."}
    return {"cp": cp, "pawns": _pawns(cp), "mate_in": None, "meaning": _advantage_meaning(cp)}


def _mover_state(cp, mate, mover: str) -> str:
    """Describe one evaluation from the mover's POV (used when mates are involved)."""
    sign = 1 if mover == "White" else -1
    if mate is not None:
        m = mate * sign
        return f"{mover} mates in {abs(m)}" if m > 0 else f"{mover} gets mated in {abs(m)}"
    if cp is None:
        return "unknown"
    return f"{mover} at {_signed_pawns(cp * sign)} pawns"


def _change_block(move: dict) -> dict:
    mover = "White" if move["side"] == "white" else "Black"
    sign = 1 if mover == "White" else -1
    cb, ca = move.get("eval_before_cp"), move.get("eval_after_cp")
    mb, ma = move.get("mate_before"), move.get("mate_after")

    if cb is not None and ca is not None and mb is None and ma is None:
        d = (ca - cb) * sign
        a = abs(d)
        if a <= 10:
            meaning = f"The move kept {mover}'s evaluation essentially unchanged ({_signed_pawns(d)} pawns)."
        else:
            size = ("slightly" if a <= 30 else "moderately" if a <= 80
                    else "significantly" if a <= 200 else "drastically")
            direction = "improved" if d > 0 else "worsened"
            meaning = (f"The move {size} {direction} {mover}'s position: "
                       f"{_signed_pawns(d)} pawns from {mover}'s point of view.")
        return {"cp": d, "pawns": _pawns(d), "meaning": meaning}

    # A forced mate is involved on at least one side: no meaningful cp delta.
    before = _mover_state(cb, mb, mover)
    after = _mover_state(ca, ma, mover)
    return {"cp": None, "pawns": None,
            "meaning": f"From {mover}'s point of view: before the move, {before}; after it, {after}."}


def _material_block(pawns: int) -> dict:
    if pawns == 0:
        meaning = "Material is level."
    else:
        side = "White" if pawns > 0 else "Black"
        meaning = f"{side} is up {abs(pawns)} point{'s' if abs(pawns) != 1 else ''} of material."
    return {"pawns": pawns, "meaning": meaning}


def build_move_prompt(move: dict, history: list[dict] | None) -> tuple:
    """Returns (system, user) strings for annotating a single move.

    history: up to HISTORY_LEN preceding moves' feature dicts, oldest first
    (e.g. [move_at_ply_n-2, move_at_ply_n-1]). Pass None or [] for the game's
    first move(s). Kept as a list rather than a single "previous move" so the
    model gets short-term continuity (the last couple of moves' quality
    labels), not just the immediately preceding one.
    """
    slim = {k: move[k] for k in ("ply", "move_number", "side", "san", "phase")}
    slim["eval_before"] = _eval_block(move.get("eval_before_cp"), move.get("mate_before"))
    slim["eval_after"] = _eval_block(move.get("eval_after_cp"), move.get("mate_after"))
    slim["eval_change_for_mover"] = _change_block(move)
    slim["material_after"] = _material_block(move["material_after"])
    for k in ("quality_label", "nag_symbol", "is_best_move", "best_move_san",
              "principal_variation_san", "tags", "captured_piece"):
        slim[k] = move.get(k)

    hist_slim = None
    if history:
        hist_slim = [
            {k: h[k] for k in ("ply", "side", "san", "quality_label")}
            for h in history
        ]
    user = MOVE_USER_TEMPLATE.format(
        move_json=json.dumps(slim, indent=2),
        history_json=json.dumps(hist_slim, indent=2) if hist_slim else "null",
    )
    return SYSTEM_PROMPT, user


GAME_SUMMARY_SYSTEM_PROMPT = """You are a chess coach writing a short (4-6 \
sentence) narrative summary of an entire game, given a JSON list of its key \
moments (opening info, blunders, best moves, and the result). Use only the \
facts given. Mention the opening if named, the critical turning point(s), \
and how the result came about. Plain prose, no bullet points."""

GAME_SUMMARY_USER_TEMPLATE = """Game headers:
{headers_json}

Key moments (JSON list, each with ply, san, quality_label, cp_loss, side):
{key_moments_json}

Write the game summary."""


def build_game_summary_prompt(headers: dict, moves: list) -> tuple:
    key_moments = [
        {
            "ply": m["ply"], "san": m["san"], "side": m["side"],
            "quality_label": m["quality_label"], "cp_loss": m["cp_loss"],
        }
        for m in moves
        if m["quality_label"] in ("blunder", "mistake", "best") or m["ply"] in (1, len(moves))
    ]
    user = GAME_SUMMARY_USER_TEMPLATE.format(
        headers_json=json.dumps(headers, indent=2),
        key_moments_json=json.dumps(key_moments, indent=2),
    )
    return GAME_SUMMARY_SYSTEM_PROMPT, user

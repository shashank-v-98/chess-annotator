"""Build the v3 distillation dataset with the APP's teacher pipeline.

For every game in the given PGN folders:
  1. Stockfish analysis (local engine) + positional insights for every move
  2. choose which moves to label (all blunders/mistakes, moves with notable
     ideas, a sample of the rest)
  3. teacher = app commentator (gpt-oss-120b) + fact-check pass
  4. automatic consistency check of the final annotation
  5. one JSON record per labelled move, appended to --out (resumable)

Each record keeps the teacher's draft AND the fact-checked version, so the
pairs where the fact-checker changed something can later be used for
preference tuning (DPO).

    python -m distill.generate_app_dataset \
        --pgn-dirs examples/distill_games,examples/master_games \
        --engine ./stockfish/stockfish/stockfish-windows-x86-64-avx2.exe \
        --out distill/data/raw_dataset_v3app.jsonl --max-cost 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_annotator import app_prompts, features, positional, prompts  # noqa: E402
from distill import consistency  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PRICES = {"openai/gpt-oss-120b": (0.15, 0.75), "openai/gpt-oss-20b": (0.075, 0.30)}  # USD per M tokens
NOTABLE = ("fork", "pin", "skewer", "discover", "mate", "en prise", "cannot be taken", "pawn break",
           "passed pawn", "hanging pawns", "isolated", "doubled", "opposite", "endgame", "majority",
           "fixed", "short of squares", "takes", "bishop pair", "open", "double check")


class AuthError(RuntimeError):
    pass


def groq_chat(api_key: str, model: str, system: str, user: str, effort: str, max_tokens: int = 1500) -> tuple:
    import requests
    payload = {"model": model, "temperature": 0.4, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = effort
    last = None
    for attempt in range(6):
        try:
            r = requests.post(GROQ_URL, json=payload, timeout=90, headers={"Authorization": f"Bearer {api_key}"})
        except requests.RequestException as e:
            last = e
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after") or 5 * (attempt + 1))
            if wait > 120:
                raise RuntimeError(f"rate limited for {wait:.0f}s -- quota exhausted?")
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            last = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (401, 403):
            raise AuthError(f"Groq rejected the API key (HTTP {r.status_code}). Check GROQ_API_KEY.")
        r.raise_for_status()
        data = r.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if text:
            return text, data.get("usage") or {}
        last = RuntimeError("empty response")
    raise RuntimeError(f"teacher failed: {last}")


def cost_of(model: str, usage: dict) -> float:
    pin, pout = PRICES.get(model, (0.5, 1.5))
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


def select(moves: list, insights: list, args, rng: random.Random) -> list:
    chosen = []
    for m, ins in zip(moves, insights):
        if m["quality_label"] in ("blunder", "mistake"):
            chosen.append(m["ply"])
            continue
        text = " ".join(ins["move_ideas"]).lower()
        p = args.p_notable if any(k in text for k in NOTABLE) else (args.p_best if m["quality_label"] == "best" else args.p_other)
        if rng.random() < p:
            chosen.append(m["ply"])
    return chosen


def label_move(m, history, ins, args, api_key) -> dict:
    sys_p, user_p = app_prompts.build_app_move_prompt(m, history, ins)
    draft, u1 = groq_chat(api_key, args.model, sys_p, user_p, args.reasoning)
    final, u2 = draft, {}
    facts_json = app_prompts.facts_from_teacher_prompt(user_p)
    facts = json.loads(facts_json)
    facts["uci"] = m["uci"]
    if not args.no_factcheck:
        fsys, fuser = app_prompts.build_factcheck_prompt(user_p, draft)
        checked, u2 = groq_chat(api_key, args.model, fsys, fuser, args.reasoning)
        if checked.strip() and checked.strip() != "NO_CHANGES" \
                and len(checked) >= 0.6 * len(draft) \
                and len(consistency.check(facts, checked)) <= len(consistency.check(facts, draft)):
            final = checked  # accept the correction only if it isn't worse by the automatic check
    return {
        "facts": facts_json,
        "teacher_draft": draft,
        "teacher_annotation": final,
        "factcheck_changed": final.strip() != draft.strip(),
        "violations": consistency.check(facts, final),
        "draft_violations": consistency.check(facts, draft),
        "cost": cost_of(args.model, u1) + cost_of(args.model, u2),
    }


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dirs", default="examples/distill_games,examples/master_games")
    ap.add_argument("--out", default="distill/data/raw_dataset_v3app.jsonl")
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--no-factcheck", action="store_true")
    ap.add_argument("--max-games", type=int, default=None, help="Total game limit (for pilots)")
    ap.add_argument("--max-games-per-file", type=int, default=None,
                    help="Use at most this many (valid) games from each PGN file, e.g. 30 per player")
    ap.add_argument("--skip-sources", default="", help="Comma-separated PGN file stems to skip, e.g. anishgiri")
    ap.add_argument("--min-plies", type=int, default=20, help="Skip very short games")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--p-notable", type=float, default=0.7, help="Label prob. for moves with notable insights")
    ap.add_argument("--p-best", type=float, default=0.35)
    ap.add_argument("--p-other", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4, help="Parallel teacher calls")
    ap.add_argument("--max-cost", type=float, default=40.0, help="Stop once the teacher spend reaches this (USD)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, spent = set(), 0.0
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["game_id"], r["move"]["ply"]))
                spent += r.get("cost", 0.0)
        print(f"Resuming: {len(done)} records, ${spent:.2f} spent so far")

    rng = random.Random(args.seed)
    analyser = features._make_analyser("local", args.engine, args.depth, 1)
    games_seen, written = 0, 0
    try:
        with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
            for d in [x.strip() for x in args.pgn_dirs.split(",") if x.strip()]:
                skip = {x.strip() for x in args.skip_sources.split(",") if x.strip()}
                for pgn_file in sorted(Path(d).glob("*.pgn")):
                    if pgn_file.stem in skip:
                        print(f"  skipping {pgn_file.name}")
                        continue
                    per_file = 0
                    for idx, game in enumerate(features.iter_games_in_pgn(pgn_file.read_text(encoding="utf-8"))):
                        if args.max_games and games_seen >= args.max_games:
                            return
                        if spent >= args.max_cost:
                            print(f"\nReached --max-cost ${args.max_cost:.2f}; stopping.")
                            return
                        game_id = f"{pgn_file.stem}_{idx}"
                        n = sum(1 for _ in game.mainline_moves())
                        if n < args.min_plies or n > args.max_plies or game.errors:
                            continue
                        if args.max_games_per_file and per_file >= args.max_games_per_file:
                            break
                        per_file += 1
                        games_seen += 1
                        # Selection must not depend on resume state, so always analyse + select.
                        analysis = features._analyze_parsed_game(game, analyser, None)
                        moves = analysis["moves"]
                        insights = []
                        for m in moves:
                            try:
                                insights.append(positional.insights_for_move_record(m))
                            except Exception:  # noqa: BLE001
                                insights.append({"move_ideas": [], "position_context": [],
                                                 "best_move_ideas": [], "opponent_reply_ideas": []})
                        # Per-game seed: which moves get labelled doesn't depend on which files ran before.
                        chosen = [p for p in select(moves, insights, args, random.Random(f"{args.seed}-{game_id}"))
                                  if (game_id, p) not in done and moves[p - 1].get("engine_ok", True)]
                        jobs = []
                        for p in chosen:
                            i = p - 1
                            hist = moves[max(0, i - prompts.HISTORY_LEN):i]
                            jobs.append((moves[i], hist, insights[i],
                                         pool.submit(label_move, moves[i], hist, insights[i], args, api_key)))
                        ok = 0
                        for m, hist, ins, fut in jobs:
                            try:
                                lab = fut.result()
                            except AuthError as e:
                                raise SystemExit(str(e))
                            except Exception as e:  # noqa: BLE001
                                print(f"    [skip] {game_id} ply {m['ply']}: {e}")
                                continue
                            rec = {"game_id": game_id, "source": pgn_file.stem,
                                   "headers": {k: analysis["headers"].get(k) for k in ("White", "Black", "WhiteElo", "BlackElo", "Result", "ECO", "TimeControl")},
                                   "move": m, "history": hist, "insights": ins,
                                   "teacher_model": args.model, **lab}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            spent += lab["cost"]
                            written += 1
                            ok += 1
                        print(f"  {game_id}: {n} plies, {len(chosen)} to label, {ok} written | total {written} new, ${spent:.2f}")
    finally:
        analyser.close()
        print(f"\nWrote {written} new records; total spend ${spent:.2f} -> {out}")


if __name__ == "__main__":
    main()

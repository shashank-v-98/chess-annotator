"""Run the v3 student (Qwen2.5-1.5B + LoRA) on a game, using exactly the student prompt
format it was trained on, and score every comment with the consistency checker.

    python -m distill.run_student_v3 --pgn examples/sample_game.pgn
    python -m distill.run_student_v3 --pgn path/to/game.pgn --plies 1-30          # first 30 half moves
    python -m distill.run_student_v3 --pgn games.pgn --game-index 3               # 4th game in the file
    python -m distill.run_student_v3 --game-json saved_analysis.json              # skip Stockfish

Stockfish analyses the PGN first. Runs on CPU (slowly: roughly 10-30 s per move) or on a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_annotator import app_prompts, positional, prompts  # noqa: E402
from chess_annotator.llm_backends import LocalHFBackend  # noqa: E402
from distill import consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-json", default=None,
                    help="Saved engine analysis (annotate.py --json-out). Ignored if --pgn is given.")
    ap.add_argument("--pgn", default=None, help="Any PGN file: analyse it with Stockfish first, then annotate")
    ap.add_argument("--game-index", type=int, default=0,
                    help="Which game to use when the PGN file holds several (0 = first)")
    ap.add_argument("--engine", default="./stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--human", default=None, help="Optional JSON of human comments keyed by ply, shown alongside")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="chess-annotator-lora-v3")
    ap.add_argument("--plies", default="all", help="'all', 'human' or e.g. '1-20'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.pgn:
        from chess_annotator import features
        print(f"Analysing {args.pgn} with Stockfish (depth {args.depth})...")
        pgn_text = Path(args.pgn).read_text(encoding="utf-8")
        chosen = None
        for idx, g in enumerate(features.iter_games_in_pgn(pgn_text)):
            if idx == args.game_index:
                chosen = g
                break
        if chosen is None:
            raise SystemExit(f"{args.pgn} has no game with index {args.game_index}")
        game = features.analyze_game(str(chosen), engine_mode="local", engine_path=args.engine, depth=args.depth)
        human = {}
        if args.plies == "human":
            args.plies = "all"
        stem = Path(args.pgn).stem + (f"_{args.game_index}" if args.game_index else "")
    elif args.game_json:
        game = json.loads(Path(args.game_json).read_text(encoding="utf-8"))
        human = json.loads(Path(args.human).read_text(encoding="utf-8")) if args.human and Path(args.human).exists() else {}
        stem = Path(args.game_json).stem
    else:
        raise SystemExit("Give a game with --pgn or --game-json.")
    args.out = args.out or f"examples/{stem}_student_v3.md"
    args.json_out = args.json_out or f"examples/{stem}_student_v3.json"
    moves = game["moves"]
    if args.plies == "human":
        wanted = {int(p) for p in human}
    elif args.plies == "all":
        wanted = {m["ply"] for m in moves}
    else:
        a, b = args.plies.split("-")
        wanted = set(range(int(a), int(b) + 1))

    print("Loading model + adapter...")
    student = LocalHFBackend(model_name=args.model, adapter_path=args.adapter, temperature=0.0)

    h = game.get("headers", {})
    title = f"{h.get('White', '?')} vs {h.get('Black', '?')} ({h.get('Result', '?')})"
    results, lines, clean = [], [f"# {title} -- v3 student commentary\n"], 0
    for i, m in enumerate(moves):
        if m["ply"] not in wanted:
            continue
        history = moves[max(0, i - prompts.HISTORY_LEN):i]
        insights = positional.insights_for_move_record(m)
        _, teacher_user = app_prompts.build_app_move_prompt(m, history, insights)
        facts_json = app_prompts.facts_from_teacher_prompt(teacher_user)
        sys_p, user_p = app_prompts.build_student_prompt(facts_json, history)
        t0 = time.time()
        text = student.generate(sys_p, user_p, max_tokens=260)
        facts = json.loads(facts_json)
        facts["uci"] = m["uci"]
        viol = consistency.check(facts, text)
        clean += not viol
        num = f"{m['move_number']}{'.' if m['side'] == 'white' else '...'}"
        print(f"{num}{m['san']} ({time.time() - t0:.0f}s){' VIOLATIONS: ' + str(viol) if viol else ''}\n  {text}\n")
        results.append({"ply": m["ply"], "san": m["san"], "quality_label": m["quality_label"],
                        "insights": insights, "student": text, "violations": viol,
                        "human": human.get(str(m["ply"]))})
        lines.append(f"## {num}{m['san']} ({m['quality_label']})")
        lines.append(f"**Student:** {text}\n")
        if viol:
            lines.append(f"*Checker: {'; '.join(viol)}*\n")
        if human.get(str(m["ply"])):
            lines.append(f"**Human:** {human[str(m['ply'])].strip()}\n")

    h = game.get("headers", {})
    lines.insert(1, f"{h.get('White', '?')} vs {h.get('Black', '?')}, {h.get('Event', '')} {h.get('Date', '')} ({h.get('Result', '?')})\n")
    lines.insert(2, f"Consistency check: {clean}/{len(results)} comments with no violations.\n")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{clean}/{len(results)} clean. Wrote {args.out}")


if __name__ == "__main__":
    main()

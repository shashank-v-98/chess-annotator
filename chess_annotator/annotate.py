"""End-to-end CLI: PGN -> structured features -> SLM annotations -> report.

Usage:
    python -m chess_annotator.annotate \\
        --pgn examples/sample_game.pgn \\
        --backend groq --model openai/gpt-oss-120b \\
        --out examples/sample_output.md

    # No API key / no GPU yet? Wire the pipeline first:
    python -m chess_annotator.annotate --pgn examples/sample_game.pgn --backend mock --out /tmp/out.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import features as feat_mod
from . import prompts
from .llm_backends import get_backend


def annotate_game(pgn_text: str, backend, engine_mode: str = "local", engine_path: str = "stockfish", depth: int = 16, max_plies=None) -> dict:
    analysis = feat_mod.analyze_game(pgn_text, engine_mode=engine_mode, engine_path=engine_path, depth=depth, max_plies=max_plies)
    moves = analysis["moves"]

    annotations = []
    history: list = []
    for m in moves:
        sys_p, user_p = prompts.build_move_prompt(m, history)
        text = backend.generate(sys_p, user_p, max_tokens=200)
        annotations.append(text)
        history = (history + [m])[-prompts.HISTORY_LEN:]

    sys_p, user_p = prompts.build_game_summary_prompt(analysis["headers"], moves)
    summary = backend.generate(sys_p, user_p, max_tokens=300)

    return {"headers": analysis["headers"], "moves": moves, "annotations": annotations, "summary": summary}


def render_markdown(result: dict) -> str:
    h = result["headers"]
    lines = []
    title = f"{h.get('White', '?')} vs {h.get('Black', '?')}, {h.get('Event', '')} {h.get('Date', '')}".strip()
    lines.append(f"# {title}\n")
    lines.append(f"**Result:** {h.get('Result', '?')}\n")
    lines.append("## Summary\n")
    lines.append(result["summary"] + "\n")
    lines.append("## Move-by-move annotations\n")
    for m, ann in zip(result["moves"], result["annotations"]):
        prefix = f"{m['move_number']}." + ("" if m["side"] == "white" else "..")
        lines.append(f"**{prefix} {m['san']}{m['nag_symbol']}**  ({m['quality_label']})")
        lines.append(f"> {ann}\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to a .pgn file")
    ap.add_argument("--backend", default="mock", choices=["groq", "local", "mock"])
    ap.add_argument("--model", default=None, help="Model name/id for groq or local backend")
    ap.add_argument("--adapter", default=None, help="Path to a LoRA adapter (local backend only) -- "
                                                      "the distilled student trained via distill/train_on_kaggle.ipynb")
    ap.add_argument("--engine-mode", default="local", choices=["local", "online"],
                     help="'local' uses a Stockfish binary; 'online' uses the free "
                          "chess-api.com server (no install needed, light use only)")
    ap.add_argument("--engine", default="stockfish", help="Path to Stockfish binary (local mode only)")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output .md path (default: stdout)")
    ap.add_argument("--json-out", default=None, help="Optional raw JSON output path")
    args = ap.parse_args()

    pgn_text = Path(args.pgn).read_text(encoding="utf-8")

    backend_kwargs = {}
    if args.model:
        key = "model" if args.backend == "groq" else "model_name"
        backend_kwargs[key] = args.model
    if args.adapter:
        if args.backend != "local":
            raise SystemExit("--adapter is only supported with --backend local")
        backend_kwargs["adapter_path"] = args.adapter
    backend = get_backend(args.backend, **backend_kwargs)

    result = annotate_game(pgn_text, backend, engine_mode=args.engine_mode, engine_path=args.engine, depth=args.depth, max_plies=args.max_plies)
    md = render_markdown(result)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

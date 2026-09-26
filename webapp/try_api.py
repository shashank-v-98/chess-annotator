"""Send a PGN to a running server and print the event stream.

    python -m webapp.try_api examples/sample_game.pgn
    python -m webapp.try_api examples/sample_game.pgn --no-factcheck --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--no-factcheck", action="store_true")
    args = ap.parse_args()

    body = {"pgn": Path(args.pgn).read_text(encoding="utf-8"), "factcheck": not args.no_factcheck}
    t0 = time.time()
    first_comment = None
    with httpx.stream("POST", f"{args.url}/api/annotate", json=body, timeout=None) as r:
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.read().decode()}")
            sys.exit(1)
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            t = time.time() - t0
            if ev["type"] == "meta":
                h = ev["headers"]
                print(f"[{t:5.1f}s] {h.get('White')} vs {h.get('Black')}: {ev['plies']} plies (cached={ev['cached']})")
            elif ev["type"] == "move":
                print(f"[{t:5.1f}s] move {ev['ply']:>3} {ev['san']:<7} {ev['quality_label']:<10} eval {ev['eval_after_cp']}")
            elif ev["type"] == "comment":
                first_comment = first_comment or t
                print(f"[{t:5.1f}s] comment {ev['ply']:>3}: {ev.get('text') or 'ERROR ' + ev.get('error', '')}")
            else:
                print(f"[{t:5.1f}s] {ev}")
    if first_comment:
        print(f"\nfirst comment after {first_comment:.1f}s, total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

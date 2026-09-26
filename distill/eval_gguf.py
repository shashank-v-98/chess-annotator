"""Check a quantized (GGUF) student against the original before using it in the app.

Runs a sample of the v3 test set through a running llama-server, scores every comment
with the same consistency checker, and compares with the Hugging Face student's outputs
on the same records (chess-annotator-lora-v3/test_outputs.json). Also reports speed.

    llama-server -m student-v3-q8_0.gguf --jinja -c 4096 --port 8081      # other terminal
    python -m distill.eval_gguf --n 150
    python -m distill.eval_gguf --n 150 --out distill/data/v3/gguf_q4_eval.json   # after switching models
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill import consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--test-jsonl", default="distill/data/v3/test.jsonl")
    ap.add_argument("--hf-outputs", default="chess-annotator-lora-v3/test_outputs.json")
    ap.add_argument("--n", type=int, default=150, help="records to sample (0 = all 1,145)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=260)
    ap.add_argument("--out", default="distill/data/v3/gguf_eval.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.test_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    hf = {(r["game_id"], r["ply"]): r for r in json.loads(Path(args.hf_outputs).read_text(encoding="utf-8"))}
    if args.n and args.n < len(recs):
        recs = random.Random(args.seed).sample(recs, args.n)

    results, t_all, tok_all = [], 0.0, 0
    for i, r in enumerate(recs, 1):
        t0 = time.time()
        resp = requests.post(f"{args.url}/v1/chat/completions", timeout=600, json={
            "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            "temperature": 0, "max_tokens": args.max_tokens, "cache_prompt": True})
        resp.raise_for_status()
        data = resp.json()
        dt = time.time() - t0
        text = (data["choices"][0]["message"].get("content") or "").strip()
        timings = data.get("timings", {})
        facts = json.loads(r["facts"]); facts["uci"] = r["uci"]
        viol = consistency.check(facts, text)
        ref = hf.get((r["game_id"], r["ply"]), {})
        results.append({"game_id": r["game_id"], "ply": r["ply"], "san": r["san"], "quality_label": r["quality_label"],
                        "gguf": text, "gguf_violations": viol, "hf": ref.get("student"),
                        "hf_violations": ref.get("violations"), "identical": text == ref.get("student"),
                        "seconds": round(dt, 2), "prompt_tps": timings.get("prompt_per_second"),
                        "gen_tps": timings.get("predicted_per_second")})
        t_all += dt
        tok_all += (data.get("usage") or {}).get("completion_tokens", 0)
        if i % 10 == 0 or i == len(recs):
            clean = sum(not x["gguf_violations"] for x in results)
            print(f"{i}/{len(recs)}  clean {clean}/{i}  avg {t_all / i:.1f}s per comment")

    n = len(results)
    g_clean = sum(not x["gguf_violations"] for x in results)
    both = [x for x in results if x["hf_violations"] is not None]
    h_clean = sum(not x["hf_violations"] for x in both)
    ident = sum(x["identical"] for x in both)
    pps = [x["prompt_tps"] for x in results if x["prompt_tps"]]
    gps = [x["gen_tps"] for x in results if x["gen_tps"]]
    summary = {
        "n": n,
        "gguf_clean": f"{g_clean}/{n} ({100 * g_clean / n:.1f}%)",
        "hf_clean_same_records": f"{h_clean}/{len(both)} ({100 * h_clean / max(1, len(both)):.1f}%)",
        "identical_to_hf": f"{ident}/{len(both)}",
        "avg_seconds_per_comment": round(t_all / n, 2),
        "avg_completion_tokens": round(tok_all / n, 1),
        "prompt_tokens_per_s": round(sum(pps) / len(pps), 1) if pps else None,
        "gen_tokens_per_s": round(sum(gps) / len(gps), 1) if gps else None,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Filter, split and format the v3 app-teacher dataset for student training.

  * drops labels that fail the automatic consistency check (distill/consistency.py)
  * splits BY GAME within each source (player), so no game leaks across splits
  * renders the student prompt (short instruction + the teacher's facts JSON) so the
    Kaggle notebook needs no prompt code: each line has system / user / completion
  * writes DPO pairs where the fact-checker fixed a draft that had violations
    (rejected = draft, chosen = fact-checked final)

    python -m distill.build_v3_split --in distill/data/raw_dataset_v3app.jsonl --out-dir distill/data/v3
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_annotator import app_prompts  # noqa: E402
from distill import consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="distill/data/raw_dataset_v3app.jsonl")
    ap.add_argument("--out-dir", default="distill/data/v3")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped, dpo = [], collections.Counter(), []
    for r in recs:
        facts = json.loads(r["facts"])
        facts["uci"] = r["move"]["uci"]
        final_v = consistency.check(facts, r["teacher_annotation"])
        if final_v:
            for v in final_v:
                dropped[v.split(",")[0][:55]] += 1
            continue
        kept.append(r)
        if r.get("factcheck_changed") and consistency.check(facts, r["teacher_draft"]):
            dpo.append(r)

    # Game-level split within each source
    rng = random.Random(args.seed)
    by_source = collections.defaultdict(list)
    for r in kept:
        by_source[r["source"]].append(r["game_id"])
    split_of = {}
    for src, gids in by_source.items():
        games = sorted(set(gids))
        rng.shuffle(games)
        n_val = max(1, round(len(games) * args.val_frac))
        n_test = max(1, round(len(games) * args.test_frac))
        for g in games[:n_test]:
            split_of[g] = "test"
        for g in games[n_test:n_test + n_val]:
            split_of[g] = "val"
        for g in games[n_test + n_val:]:
            split_of[g] = "train"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    counts = collections.Counter()
    labels = collections.defaultdict(collections.Counter)
    for r in kept:
        s = split_of[r["game_id"]]
        sys_p, user_p = app_prompts.build_student_prompt(r["facts"], r["history"])
        line = {"system": sys_p, "user": user_p, "completion": r["teacher_annotation"],
                "facts": r["facts"], "uci": r["move"]["uci"],
                "game_id": r["game_id"], "source": r["source"], "ply": r["move"]["ply"],
                "san": r["move"]["san"], "quality_label": r["move"]["quality_label"],
                "best_move_san": r["move"].get("best_move_san")}
        files[s].write(json.dumps(line, ensure_ascii=False) + "\n")
        counts[s] += 1
        labels[s][r["move"]["quality_label"]] += 1
    for f in files.values():
        f.close()

    with (out / "dpo_pairs.jsonl").open("w", encoding="utf-8") as f:
        n_dpo = 0
        for r in dpo:
            if split_of[r["game_id"]] != "train":
                continue
            sys_p, user_p = app_prompts.build_student_prompt(r["facts"], r["history"])
            f.write(json.dumps({"system": sys_p, "user": user_p, "chosen": r["teacher_annotation"],
                                "rejected": r["teacher_draft"]}, ensure_ascii=False) + "\n")
            n_dpo += 1

    print(f"Input {len(recs)} records; dropped {len(recs) - len(kept)} failing the consistency check; kept {len(kept)}")
    for k, v in dropped.most_common(8):
        print(f"    {v:4}  {k}")
    games = collections.Counter(split_of.values())
    for s in ("train", "val", "test"):
        dist = ", ".join(f"{k} {100 * v / counts[s]:.0f}%" for k, v in sorted(labels[s].items()))
        print(f"{s:5}: {games[s]:3} games, {counts[s]:5} records | {dist}")
    print(f"DPO pairs (train games only): {n_dpo}")
    print(f"Wrote {out}/train.jsonl, val.jsonl, test.jsonl, dpo_pairs.jsonl")


if __name__ == "__main__":
    main()

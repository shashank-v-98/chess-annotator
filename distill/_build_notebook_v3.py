"""Generates distill/train_v3_on_kaggle.ipynb (v3 student: Qwen2.5-1.5B on the app-teacher data).

The data files from distill/build_v3_split.py already contain the rendered prompts
(system / user / completion), so the notebook has no prompt-building code to drift.
distill/consistency.py is embedded verbatim for the test-set fact check.

    python distill/_build_notebook_v3.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSISTENCY_SRC = (HERE / "consistency.py").read_text(encoding="utf-8")


def code(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def md(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


cells = []
cells.append(md("""# Chess commentary student v3 -- Qwen2.5-1.5B LoRA (Kaggle)

Trains a small model to write move commentary from verified engine + board facts, imitating the app's
teacher (gpt-oss-120b with a fact-check pass). Data: `distill/build_v3_split.py` output
(`train.jsonl`, `val.jsonl`, `test.jsonl`), already filtered by the automatic consistency check.

**Before running:** upload the three files as a Kaggle Dataset, attach it, set `DATA_DIR`,
choose **GPU T4 x2**, turn **Internet on**, then Run All (or Save & Run All to run in the background)."""))

cells.append(code("""!pip install -q -U peft trl accelerate datasets torchao"""))

cells.append(code("""import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import json, random, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)"""))

cells.append(code("""# ---- Config: edit these ----
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "/kaggle/input/<your-v3-dataset>"      # paste the path from the Input sidebar
OUTPUT_DIR = "/kaggle/working/chess-annotator-lora-v3"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8            # effective batch 16
MAX_SEQ_LEN = 1024              # v3 prompts are ~650 tokens; the check below fails loudly if any are longer
MAX_TRAIN = None                # e.g. 3000 for a quick trial run
TEST_GEN_BATCH = 16             # batched greedy generation on the test set
MAX_NEW_TOKENS = 260
# ----------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)"""))

cells.append(md("## 1. Data"))
cells.append(code("""def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train_records = load_jsonl(f"{DATA_DIR}/train.jsonl")
val_records = load_jsonl(f"{DATA_DIR}/val.jsonl")
test_records = load_jsonl(f"{DATA_DIR}/test.jsonl")
if MAX_TRAIN:
    random.shuffle(train_records)
    train_records = train_records[:MAX_TRAIN]
print(f"train={len(train_records)}  val={len(val_records)}  test={len(test_records)}")
print(train_records[0]["user"][:600], "...\\n->", train_records[0]["completion"][:300])"""))

cells.append(code("""tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def render_prompt(r):
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def build_dataset(records):
    return Dataset.from_dict({"prompt": [render_prompt(r) for r in records],
                              "completion": [r["completion"] for r in records]})

train_ds = build_dataset(train_records)
val_ds = build_dataset(val_records)

lengths = sorted(len(tokenizer(p + c)["input_ids"]) for p, c in zip(train_ds["prompt"], train_ds["completion"]))
print(f"train sequence length: median {lengths[len(lengths)//2]}, max {lengths[-1]} (MAX_SEQ_LEN={MAX_SEQ_LEN})")
assert lengths[-1] <= MAX_SEQ_LEN, "Some examples exceed MAX_SEQ_LEN -- raise it or they will be truncated\""""))

cells.append(md("## 2. Model + LoRA"))
cells.append(code("""model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()"""))

cells.append(md("## 3. Train (validation loss every half epoch)"))
cells.append(code("""steps_per_epoch = max(1, len(train_ds) // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
eval_every = max(20, steps_per_epoch // 2)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_SEQ_LEN,
    completion_only_loss=True,
    loss_type="nll",                       # trl's default chunked loss misbehaves with PEFT models
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    disable_tqdm=True,                     # print losses as plain text, so background (commit) runs show progress in the log
    eval_strategy="steps", eval_steps=eval_every,
    save_strategy="steps", save_steps=eval_every, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer)
t0 = time.time()
trainer.train()
print(f"training took {(time.time() - t0) / 60:.0f} min")"""))

cells.append(code("""hist = trainer.state.log_history
tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
plt.figure(figsize=(7, 4))
if tr: plt.plot(*zip(*tr), label="train loss")
if ev: plt.plot(*zip(*ev), marker="o", label="val loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("v3 student: train vs val loss")
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=120, bbox_inches="tight"); plt.show()
print("val loss by eval:", [round(x[1], 4) for x in ev])

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)"""))

cells.append(md("## 4. Test set: generate, then check every annotation against its facts\n\nThe checker below is `distill/consistency.py`, embedded verbatim."))
cells.append(code(CONSISTENCY_SRC))
cells.append(code("""model.eval()
tokenizer.padding_side = "left"

def generate_batch(records):
    prompts = [render_prompt(r) for r in records]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

t0 = time.time()
student = []
for i in range(0, len(test_records), TEST_GEN_BATCH):
    student += generate_batch(test_records[i:i + TEST_GEN_BATCH])
    if (i // TEST_GEN_BATCH) % 10 == 0:
        print(f"  {len(student)}/{len(test_records)}  ({time.time() - t0:.0f}s)")

results = []
for r, s in zip(test_records, student):
    facts = json.loads(r["facts"]); facts["uci"] = r["uci"]
    results.append({**{k: r[k] for k in ("game_id", "source", "ply", "san", "quality_label", "best_move_san")},
                    "teacher": r["completion"], "student": s, "violations": check(facts, s)})
Path(f"{OUTPUT_DIR}/test_outputs.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

n = len(results)
clean = sum(not x["violations"] for x in results)
print(f"\\nStudent consistency: {clean}/{n} ({100 * clean / n:.1f}%) annotations with no violations "
      f"(teacher: 100% by construction -- the data was filtered with the same check)")
from collections import Counter
print(Counter(v.split(",")[0][:55] for x in results for v in x["violations"]).most_common(10))
g = [x for x in results if x["quality_label"] in ("blunder", "mistake") and x["best_move_san"]]
hits = sum(x["best_move_san"] in x["student"] for x in g)
print(f"Blunders/mistakes naming the engine's best move: {hits}/{len(g)} ({100 * hits / max(1, len(g)):.0f}%)")
by_src = {}
for x in results:
    by_src.setdefault("own games" if x["source"] == "shanky8991" else "master games", []).append(not x["violations"])
for k, v in by_src.items():
    print(f"  {k}: {100 * sum(v) / len(v):.1f}% clean ({len(v)} examples)")
for x in random.sample(results, 6):
    print("=" * 70); print(f"{x['san']} [{x['quality_label']}] violations={x['violations']}")
    print("TEACHER:", x["teacher"]); print("STUDENT:", x["student"])"""))

cells.append(code("""import shutil
shutil.make_archive("/kaggle/working/chess-annotator-lora-v3", "zip", OUTPUT_DIR)
print("Download: /kaggle/working/chess-annotator-lora-v3.zip (adapter + loss curve + test_outputs.json)")"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "train_v3_on_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out}")

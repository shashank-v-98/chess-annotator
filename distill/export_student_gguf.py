"""Merge the v3 LoRA adapter into Qwen2.5-1.5B-Instruct and save a plain Hugging Face
model folder that llama.cpp can convert to GGUF.

    python -m distill.export_student_gguf                 # -> student-v3-merged/

The merge is done directly on the safetensors files, one tensor at a time:
    W' = W + (lora_alpha / r) * B @ A
It deliberately does not go through transformers' from_pretrained: transformers 5.x loads
weights on several threads, and that crashed with an access violation on Windows.
The result is exactly what PeftModel.merge_and_unload() would produce.

Then (see webapp/README.md, "Faster student"):
    python llama.cpp/convert_hf_to_gguf.py student-v3-merged --outtype bf16 --outfile student-v3-bf16.gguf
    llama.cpp/bin/llama-quantize.exe student-v3-bf16.gguf student-v3-q8_0.gguf Q8_0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt")
ADAPTER_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                           "special_tokens_map.json", "added_tokens.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="chess-annotator-lora-v3")
    ap.add_argument("--out", default="student-v3-merged")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter = Path(args.adapter)
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("modules_to_save") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("This adapter uses LoRA features the simple merge doesn't handle.")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)

    print(f"Locating {args.model} (uses the local Hugging Face cache if already downloaded)...")
    base = Path(snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json", "*.txt"]))
    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors files in {base}")

    # LoRA pairs, keyed by the base weight name they modify
    lora = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if ".lora_A." in k or ".lora_B." in k:
                target = k.replace("base_model.model.", "", 1).split(".lora_")[0] + ".weight"
                lora.setdefault(target, {})["A" if ".lora_A." in k else "B"] = f.get_tensor(k).float()
            else:
                raise SystemExit(f"Unexpected adapter tensor {k}; refusing to guess how to merge it.")
    print(f"Adapter: {len(lora)} LoRA-modified weights, r={r}, alpha={alpha}, scale={scale:g}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged, index = 0, {}
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                w = f.get_tensor(k)
                if k in lora:
                    A, B = lora[k]["A"], lora[k]["B"]
                    delta = (B @ A) * scale
                    if fan_in_fan_out:
                        delta = delta.T
                    if delta.shape != w.shape:
                        raise SystemExit(f"Shape mismatch for {k}: {tuple(delta.shape)} vs {tuple(w.shape)}")
                    w = (w.float() + delta).to(w.dtype)
                    merged += 1
                tensors[k] = w.contiguous()
                index[k] = shard.name
        save_file(tensors, str(out / shard.name), metadata={"format": "pt"})
        print(f"  wrote {shard.name} ({len(tensors)} tensors)")
        del tensors
    if merged != len(lora):
        missing = sorted(set(lora) - set(index))[:5]
        raise SystemExit(f"Merged {merged}/{len(lora)} LoRA weights; not found in base model: {missing}")

    for name in BASE_FILES + ("model.safetensors.index.json",):
        if (base / name).exists():
            shutil.copy(base / name, out / name)
    for name in ADAPTER_TOKENIZER_FILES:  # tokenizer + chat template exactly as used in training
        if (adapter / name).exists():
            shutil.copy(adapter / name, out / name)
    print(f"Merged {merged} weights. Saved to {out}/")
    print("Next: python llama.cpp/convert_hf_to_gguf.py", out, "--outtype bf16 --outfile student-v3-bf16.gguf")


if __name__ == "__main__":
    main()

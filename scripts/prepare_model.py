#!/usr/bin/env python3
"""Prepare pinned upstream weights, then enable group-128 INT4 activations."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path


def w4a4_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    groups = result["quantization_config"]["config_groups"]
    changed = 0
    for group in groups.values():
        weight = group.get("weights") or {}
        if weight.get("type") == "int" and weight.get("num_bits") == 4:
            group["input_activations"] = {
                "actorder": None, "block_structure": None, "dynamic": True,
                "group_size": 128, "num_bits": 4, "observer": "memoryless_minmax",
                "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                "symmetric": True, "type": "int", "zp_dtype": None,
            }
            changed += 1
    if not changed:
        raise ValueError("No INT4 weight groups found; unexpected checkpoint configuration")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperqwen", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    root = args.models.resolve()
    raw, prepared, fast, final = [root / name for name in
                                  ("raw", "prepared", "prepared-fast", "prepared-fast-w4a4g")]
    if any(path.exists() for path in (raw, prepared, fast, final)):
        raise ValueError("Model output already exists; use a fresh WORK_DIR to avoid overwriting weights")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.repo, revision=args.revision, local_dir=str(raw),
                      allow_patterns=["*.safetensors", "*.json", "*.jinja"])
    shutil.copytree(raw, prepared, ignore=shutil.ignore_patterns(".cache"))
    for name in ("quant_lm_head.py", "quant_embed.py", "quant_mtp.py"):
        subprocess.run([sys.executable, str(args.hyperqwen / "prepare" / name), str(prepared)], check=True)
    subprocess.run([sys.executable, str(args.hyperqwen / "prepare/build_draft_vocab.py"), str(prepared),
                    "--ids", str(args.hyperqwen / "prepare/draft_vocab_ids.json")], check=True)
    subprocess.run([sys.executable, str(args.hyperqwen / "prepare/fetch_fast_variant.py"),
                    str(prepared), str(fast)], check=True)
    final.mkdir()
    for path in fast.iterdir():
        if path.is_file() and path.name != "config.json":
            (final / path.name).symlink_to(path.resolve())
    config = json.loads((fast / "config.json").read_text())
    (final / "config.json").write_text(json.dumps(w4a4_config(config), ensure_ascii=False, indent=2) + "\n")
    print(f"Model ready: {final}")


if __name__ == "__main__":
    main()

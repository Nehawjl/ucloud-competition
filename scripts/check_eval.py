#!/usr/bin/env python3
"""Stage-specific dependency checks; NLTK download errors never prompt for input."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys


def official_runner():
    path = Path(__file__).resolve().parents[1] / "third_party/official-eval/runner.py"
    spec = importlib.util.spec_from_file_location("official_eval_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_nltk(directory: Path, resources: dict[str, str]) -> None:
    import nltk

    directory.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(directory))
    socket.setdefaulttimeout(60)
    for name, location in resources.items():
        try:
            nltk.data.find(location)
            continue
        except LookupError:
            pass
        if not nltk.download(name, download_dir=str(directory),
                             halt_on_error=True, raise_on_error=True):
            raise RuntimeError(f"NLTK resource download failed: {name}")
        nltk.data.find(location)


def check_stage(runner, directory: Path, stage: str) -> dict:
    return runner.inspect_tool(runner.tool_python(directory, stage), stage,
                               directory / "nltk_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("perf", "accuracy"), required=True)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--prepare-nltk", action="store_true")
    args = parser.parse_args()
    if args.prepare_nltk and args.mode != "accuracy":
        parser.error("--prepare-nltk is only valid for accuracy")
    runner = official_runner()
    directory = args.env_dir.resolve()
    try:
        if args.prepare_nltk:
            prepare_nltk(directory / "nltk_data", runner.NLTK_RESOURCES)
        status = check_stage(runner, directory, args.mode)
    except (OSError, ValueError, RuntimeError, LookupError) as error:
        print(f"{args.mode} dependency preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": args.mode, "dependency": status}, ensure_ascii=False, indent=2))
    return 0 if status["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

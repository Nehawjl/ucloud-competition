#!/usr/bin/env python3
"""Select one C1-C14 cell, then execute the competition runner's main workflow."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_official():
    runpy.run_path(str(ROOT / "scripts/verify_official_eval.py"))["verify"]()
    spec = importlib.util.spec_from_file_location("competition_performance", ROOT / "third_party/official-eval/runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(concurrency: int, arguments: list[str]) -> int:
    if concurrency not in range(1, 15):
        raise ValueError("Concurrency must be between 1 and 14")
    if any(value == "--mode" or value.startswith("--mode=") for value in arguments):
        raise ValueError("This entry runs performance only; do not override --mode")
    official = load_official()
    # Only the concurrency list changes. Dataset, requests, parsing and reports
    # continue through the unmodified competition implementation.
    official.CONCURRENCIES = (concurrency,)
    try:
        return official.main(["--mode", "perf", *arguments])
    except (official.EvaluationError, OSError, ValueError) as error:
        print("错误：" + official.redact(str(error), os.environ.get("OPENAI_API_KEY", "")), file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Other options are passed to the original runner: --base-url, --model, --env-dir, --dry-run, etc.")
    parser.add_argument("--concurrency", type=int, choices=range(1, 15), default=8)
    args, official_arguments = parser.parse_known_args()
    try:
        return run(args.concurrency, official_arguments)
    except (OSError, ValueError) as error:
        print(f"Evaluation entry failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    def interrupted_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted_signal)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

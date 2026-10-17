#!/usr/bin/env python3
"""Run one official AIPerf concurrency cell with the competition runner's helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import uuid
from pathlib import Path
from types import ModuleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-runner", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--perf-requests", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_official(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_eval_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(root: Path, summary: dict) -> None:
    row = summary["performance"][0]
    metrics = row["metrics"]
    lines = [
        "# 定向性能评测",
        "",
        "本报告只运行一个并发档，用于单变量筛选；不是完整 C1/C2/C4/C8 正式报告。",
        "",
        "| 并发 | 状态 | 成功/期望 | 输出 token/s | TTFT P95 s | TPOT P95 ms/token | E2E P95 s |",
        "|---:|---|---:|---:|---:|---:|---:|",
        f"| {row['concurrency']} | {row['status']} | {row['successful_requests']}/{row['expected_requests']} | "
        f"{row['output_token_throughput']:.4f} | {metrics['time_to_first_token']['p95']/1000:.4f} | "
        f"{metrics['inter_token_latency']['p95']:.4f} | {metrics['request_latency']['p95']/1000:.4f} |",
        "",
        f"数据集 SHA-256：`{summary['configuration']['dataset_sha256']}`。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.perf_requests < 1 or args.warmup_requests < 0:
        raise ValueError("concurrency and request counts must be valid")
    if args.warmup_requests != 8:
        raise ValueError("this comparison requires the official eight warmup requests")
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    official = load_official(args.official_runner)
    if args.warmup_requests != official.WARMUP:
        raise ValueError("warmup count differs from official runner")

    key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if not key or "\n" in key or "\r" in key:
        raise ValueError("invalid API key environment")
    root = args.output_dir.resolve()
    performance = root / "performance"
    destination = performance / f"c{args.concurrency}"
    destination.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    dataset = performance / "dataset.jsonl"
    dataset_meta = performance / "dataset.meta.json"
    shutil.copy2(args.dataset, dataset)
    shutil.copy2(args.dataset_meta, dataset_meta)
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    metadata = json.loads(dataset_meta.read_text(encoding="utf-8"))
    if metadata.get("sha256") != dataset_sha:
        raise ValueError("dataset metadata SHA-256 mismatch")
    if sum(1 for line in dataset.open(encoding="utf-8") if line.strip()) < args.perf_requests + args.warmup_requests + 1:
        raise ValueError("dataset does not contain enough records")

    runner_args = argparse.Namespace(
        base_url=args.base_url,
        model=args.model,
        tokenizer="builtin",
        tokenizer_revision=None,
        env_dir=args.env_dir.resolve(),
        perf_requests=args.perf_requests,
        request_timeout=args.request_timeout,
        seed=args.seed,
    )
    configuration = {
        "mode": "targeted-perf",
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": "builtin",
        "output_dir": str(root),
        "env_dir": str(args.env_dir.resolve()),
        "perf_requests": args.perf_requests,
        "request_timeout": args.request_timeout,
        "seed": args.seed,
        "isl": official.ISL,
        "osl": official.OSL,
        "concurrencies": [args.concurrency],
        "warmup_requests": args.warmup_requests,
        "dataset_sha256": dataset_sha,
        "official_runner_sha256": hashlib.sha256(args.official_runner.read_bytes()).hexdigest(),
    }
    official.write_json(root / "configuration.json", configuration)
    official.write_json(root / "environment.json", {"platform": platform.platform(), "created_at": official.utc_now()})
    model_info = official.check_model(args.base_url, args.model, key)
    official.write_json(root / "model_info.json", model_info, key)
    official.probe_performance(runner_args, dataset, key, uuid.uuid4().hex + ":probe")
    command = official.aiperf_command(
        runner_args, args.concurrency, dataset, destination, uuid.uuid4().hex + f":c{args.concurrency}", key
    )
    official.write_json(destination / "command.json", command, key)
    started_at = official.utc_now()
    failure = None
    try:
        official.execute(
            command,
            destination / "console.log",
            key,
            official.child_environment(args.env_dir.resolve() / "nltk_data"),
        )
    except official.EvaluationError as error:
        failure = str(error)
    finally:
        official.sanitize_files(destination, key)
    try:
        result = official.parse_performance(
            destination,
            args.perf_requests,
            args.concurrency,
            warmup_expected=args.warmup_requests,
            strict_isl=False,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "status": "异常",
            "issues": [f"结果解析失败：{error}"],
            "concurrency": args.concurrency,
            "artifacts": str(destination),
        }
    if failure:
        result["status"] = "异常"
        result.setdefault("issues", []).append(failure)
    official.write_json(destination / "summary.json", result, key)
    official.write_json(performance / "summary.json", [result], key)
    summary = {
        "started_at": started_at,
        "configuration": configuration,
        "status": "完整" if result["status"] == "完整" else "异常",
        "issues": [f"并发 {args.concurrency}：{issue}" for issue in result.get("issues", [])],
        "performance": [result],
        "ended_at": official.utc_now(),
    }
    official.write_json(root / "summary.json", summary, key)
    write_report(root, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "完整" else 1


if __name__ == "__main__":
    raise SystemExit(main())

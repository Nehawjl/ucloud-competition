#!/usr/bin/env python3
"""Check archived file hashes and recompute metrics from native request records."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> None:
    manifest = load(EVIDENCE / "manifest.json")
    for relative, meta in manifest["files"].items():
        path = EVIDENCE / relative
        assert path.resolve().is_relative_to(EVIDENCE.resolve()), relative
        data = path.read_bytes()
        assert len(data) == meta["bytes"], relative
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], relative
    runner = runpy.run_path(str(ROOT / "third_party/official-eval/runner.py"))
    perf_count = accuracy_count = 0
    for entry in manifest["entries"]:
        folder = EVIDENCE / entry["directory"]
        summary = load(folder / "summary.json")
        if entry["kind"] == "performance":
            recalculated = runner["parse_performance"](folder, 100, summary["concurrency"], 8)
            assert recalculated["status"] == "完整", (entry["id"], recalculated["issues"])
            for metric in ("time_to_first_token", "inter_token_latency", "request_latency"):
                assert math.isclose(recalculated["metrics"][metric]["p95"],
                                    summary["metrics"][metric]["p95"], abs_tol=1e-5), entry["id"]
            assert math.isclose(recalculated["output_token_throughput"],
                                summary["output_token_throughput"], rel_tol=1e-8), entry["id"]
            perf_count += 1
        else:
            metrics = summary.get("accuracy", summary)["metrics"]
            assert all(0 <= value["score"] <= 1 for value in metrics.values()), entry["id"]
            if (folder / "reviews.jsonl").exists():
                reviews = [json.loads(line) for line in (folder / "reviews.jsonl").read_text().splitlines() if line]
                assert len(reviews) == 100, entry["id"]
                assert len({r["index"] for r in reviews}) == 100, entry["id"]
                native = load(folder / "ifbench.json")
                for metric in native["metrics"]:
                    name = metric["identity"]["name"]
                    numerator = denominator = 0.0
                    for row in reviews:
                        score = row["sample_score"]["score"]
                        assert score["status"] == "success", (entry["id"], row["index"])
                        weight = score.get("metadata", {}).get("metric_weights", {}).get(name, 1)
                        numerator += score["value"][name] * weight
                        denominator += weight
                    assert round(numerator / denominator, 4) == round(metric["score"], 4), (entry["id"], name)
                    assert math.isclose(metrics[name]["score"], metric["score"], abs_tol=1e-8)
                accuracy_count += 1
    print(f"OK: {len(manifest['files'])} file hashes; {perf_count} performance cases; {accuracy_count} native IFBench100 reviews; all indexed summaries readable.")


if __name__ == "__main__":
    verify()

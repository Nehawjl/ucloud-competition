#!/usr/bin/env python3
"""CUDA Graph coverage for two MTP drafts plus one verification token."""
import argparse
import json


def graph_config(concurrency: int) -> dict:
    if not 1 <= concurrency <= 15:
        raise ValueError("concurrency must be between 1 and 15")
    maximum = 3 * concurrency
    base = [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 21, 24, 27, 30, 32, 33, 36, 39, 40, 42]
    sizes = sorted({n for n in base if n <= maximum} | {maximum})
    return {"max_cudagraph_capture_size": maximum, "cudagraph_capture_sizes": sizes,
            "custom_ops": ["+rms_norm", "+silu_and_mul"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("concurrency", type=int)
    args = parser.parse_args()
    print(json.dumps(graph_config(args.concurrency), separators=(",", ":")))

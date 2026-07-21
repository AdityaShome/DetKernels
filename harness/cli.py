"""detkernels-audit: quantify and localize nondeterminism in LLM inference.

    detkernels-audit --model Qwen/Qwen3-1.7B --batch-sizes 1,32 --runs 100 --output report.json

Launches a vLLM server for `--model` (unless `--server-url` points at one
already running), sweeps the configured batch sizes, and writes a report of
which batch sizes produced bitwise-identical output across repeats and where
the first divergence occurred.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import List, Optional

from .server import ServerStartupError, VLLMServer
from .sweep import SweepConfig, run_sweep

DEFAULT_PROMPT = (
    "Explain in one paragraph why floating point addition is not associative, "
    "and why that matters for reproducibility."
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detkernels-audit",
        description="Quantify and localize nondeterminism in LLM inference configs.",
    )
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--batch-sizes", default="1,32",
        help="Comma-separated batch sizes to sweep, e.g. '1,8,32'",
    )
    p.add_argument("--runs", type=int, default=100, help="Repeats per batch size")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default="report.json")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--server-url", default=None,
        help="Point at an already-running OpenAI-compatible server instead of "
             "launching a new vLLM subprocess, e.g. http://localhost:8000/v1",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    server = None
    base_url = args.server_url
    if base_url is None:
        server = VLLMServer(
            model=args.model,
            port=args.port,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        try:
            print(f"Starting vLLM server for {args.model}...", file=sys.stderr)
            server.start()
        except ServerStartupError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        base_url = server.base_url

    cfg = SweepConfig(
        model=args.model,
        base_url=base_url,
        prompt=args.prompt,
        batch_sizes=batch_sizes,
        n_repeats=args.runs,
        max_tokens=args.max_tokens,
    )

    try:
        results = asyncio.run(run_sweep(cfg))
    finally:
        if server is not None:
            server.stop()

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "n_repeats": args.runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {
                "batch_size": r.batch_size,
                "all_identical": r.all_identical,
                "n_diverging": r.n_diverging,
                "divergence_indices": r.divergence_indices,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

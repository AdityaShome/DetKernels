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
from .sweep import SweepConfig, SweepResults, run_sweep

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
    p.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Used when --tp-sizes is not given (single-TP run).",
    )
    p.add_argument(
        "--tp-sizes", default=None,
        help="Comma-separated tensor-parallel sizes to sweep, e.g. '1,2,4'. "
             "Restarts the server once per value; requires enough GPUs. "
             "Defaults to just --tensor-parallel-size.",
    )
    p.add_argument(
        "--seq-lengths", default=None,
        help="Comma-separated target prompt lengths (in words) to sweep, "
             "e.g. '50,500,2000'. Prompt is padded with filler text to reach "
             "each length, run at --seqlen-batch-size.",
    )
    p.add_argument(
        "--seqlen-batch-size", type=int, default=32,
        help="Batch size used for the sequence-length sweep (default 32, "
             "matching the batch size where divergence is expected).",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--server-url", default=None,
        help="Point at an already-running OpenAI-compatible server instead of "
             "launching a new vLLM subprocess, e.g. http://localhost:8000/v1. "
             "Incompatible with --tp-sizes (TP requires restarting the server).",
    )
    p.add_argument(
        "--localize", action="store_true",
        help="After the sweep, if any config diverged, load the model via "
             "transformers (frees the vLLM server first) and run the "
             "layer-level activation divergence localizer at the smallest "
             "diverging batch size.",
    )
    return p


def _run_one_tp(args, tp_size: int, batch_sizes: List[int],
                 seq_lengths: List[int]) -> SweepResults:
    server = None
    base_url = args.server_url
    if base_url is None:
        server = VLLMServer(
            model=args.model,
            port=args.port,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
        )
        try:
            print(f"Starting vLLM server for {args.model} (tp={tp_size})...", file=sys.stderr)
            server.start()
        except ServerStartupError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            raise
        base_url = server.base_url

    cfg = SweepConfig(
        model=args.model,
        base_url=base_url,
        prompt=args.prompt,
        batch_sizes=batch_sizes,
        n_repeats=args.runs,
        max_tokens=args.max_tokens,
        sequence_lengths=seq_lengths,
        seqlen_batch_size=args.seqlen_batch_size,
    )

    try:
        return asyncio.run(run_sweep(cfg, tensor_parallel_size=tp_size))
    finally:
        if server is not None:
            server.stop()


def _maybe_localize(args, sweep_results: List[SweepResults]) -> Optional[dict]:
    diverging_bs = None
    for sr in sweep_results:
        for r in sr.batch_size_results:
            if not r.all_identical and r.batch_size > 1:
                if diverging_bs is None or r.batch_size < diverging_bs:
                    diverging_bs = r.batch_size
    if diverging_bs is None:
        print("No divergence found; skipping localization.", file=sys.stderr)
        return None

    from .localizer import localize_divergence

    print(f"Loading {args.model} via transformers for localization "
          f"(batch_size={diverging_bs})...", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
    )
    result = localize_divergence(model, tokenizer, args.prompt, diverging_bs, device="cuda")
    return {
        "batch_size": diverging_bs,
        "first_diverging_layer": result.first_diverging_layer,
        "layer_diffs": [
            {
                "layer_index": d.layer_index,
                "layer_name": d.layer_name,
                "max_abs_diff": d.max_abs_diff,
                "identical": d.identical,
            }
            for d in result.layer_diffs
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    seq_lengths = (
        [int(x) for x in args.seq_lengths.split(",") if x.strip()]
        if args.seq_lengths else []
    )
    tp_sizes = (
        [int(x) for x in args.tp_sizes.split(",") if x.strip()]
        if args.tp_sizes else [args.tensor_parallel_size]
    )
    if len(tp_sizes) > 1 and args.server_url is not None:
        print("ERROR: --tp-sizes requires launching the server ourselves; "
              "cannot combine with --server-url.", file=sys.stderr)
        return 1

    sweep_results: List[SweepResults] = []
    for tp_size in tp_sizes:
        try:
            sweep_results.append(_run_one_tp(args, tp_size, batch_sizes, seq_lengths))
        except ServerStartupError:
            return 1

    localization = _maybe_localize(args, sweep_results) if args.localize else None

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "n_repeats": args.runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensor_parallel_results": [
            {
                "tensor_parallel_size": sr.tensor_parallel_size,
                "batch_size_results": [
                    {
                        "batch_size": r.batch_size,
                        "all_identical": r.all_identical,
                        "n_diverging": r.n_diverging,
                        "divergence_indices": r.divergence_indices,
                    }
                    for r in sr.batch_size_results
                ],
                "sequence_length_results": [
                    {
                        "sequence_length": r.sequence_length,
                        "batch_size": r.batch_size,
                        "all_identical": r.all_identical,
                        "n_diverging": r.n_diverging,
                        "divergence_indices": r.divergence_indices,
                    }
                    for r in sr.sequence_length_results
                ],
            }
            for sr in sweep_results
        ],
        "localization": localization,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

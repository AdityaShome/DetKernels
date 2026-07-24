"""Throughput benchmark comparing the batch-invariant kernels against both
the deliberately batch-variant reference and a plain PyTorch/cuBLAS
baseline, per project.md Phase 2 task 4's requirement to benchmark
performance overhead ("then benchmark performance ... against both (a) the
standard non-deterministic kernel and (b) the reference batch-invariant
implementation") and the Phase 2 gate's "documented, reproducible benchmark
showing the performance cost."

Timed with CUDA events (not wall clock), averaged over `--iters` calls after
a warmup, since GPU kernel launches are asynchronous and wall-clock timing
of a single unsynchronized call would be meaningless.

Usage:
    detkernels-kernel-bench --batch-sizes 1,8,64 --iters 50 --output bench.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone


def _time_cuda(fn, n_iters: int = 50, n_warmup: int = 10) -> float:
    """Returns average milliseconds per call."""
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def bench_rmsnorm(batch_sizes, n_cols: int = 4096, n_iters: int = 50):
    import torch
    from kernels.rmsnorm import rmsnorm_batch_invariant, rmsnorm_batch_variant, rmsnorm_reference

    results = []
    for bs in batch_sizes:
        x = torch.randn(bs, n_cols, device="cuda", dtype=torch.float16)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float16)
        t_ref = _time_cuda(lambda: rmsnorm_reference(x, w), n_iters)
        t_inv = _time_cuda(lambda: rmsnorm_batch_invariant(x, w), n_iters)
        t_var = _time_cuda(lambda: rmsnorm_batch_variant(x, w), n_iters)
        results.append({
            "batch_size": bs, "reference_eager_ms": t_ref,
            "batch_invariant_ms": t_inv, "batch_variant_ms": t_var,
            "invariant_overhead_vs_eager_pct": (t_inv / t_ref - 1) * 100 if t_ref > 0 else None,
        })
    return results


def bench_matmul(batch_sizes, k: int = 4096, n: int = 4096, n_iters: int = 50):
    import torch
    from kernels.matmul import matmul_batch_invariant, matmul_batch_variant

    results = []
    for bs in batch_sizes:
        a = torch.randn(bs, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        t_cublas = _time_cuda(lambda: a @ b, n_iters)  # real cuBLAS baseline via torch
        t_inv = _time_cuda(lambda: matmul_batch_invariant(a, b), n_iters)
        t_var = _time_cuda(lambda: matmul_batch_variant(a, b), n_iters)
        results.append({
            "batch_size": bs, "cublas_ms": t_cublas,
            "batch_invariant_ms": t_inv, "batch_variant_ms": t_var,
            "invariant_overhead_vs_cublas_pct": (t_inv / t_cublas - 1) * 100 if t_cublas > 0 else None,
        })
    return results


def bench_attention(batch_sizes, num_heads: int = 8, head_dim: int = 64,
                     seq_len: int = 2048, n_iters: int = 50):
    import torch
    from kernels.attention import attention_batch_invariant, attention_batch_variant, attention_reference

    results = []
    for bs in batch_sizes:
        q = torch.randn(bs, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(bs, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(bs, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        t_ref = _time_cuda(lambda: attention_reference(q, k, v), n_iters)
        t_inv = _time_cuda(lambda: attention_batch_invariant(q, k, v), n_iters)
        t_var = _time_cuda(lambda: attention_batch_variant(q, k, v), n_iters)
        results.append({
            "batch_size": bs, "reference_eager_ms": t_ref,
            "batch_invariant_ms": t_inv, "batch_variant_ms": t_var,
            "invariant_overhead_vs_eager_pct": (t_inv / t_ref - 1) * 100 if t_ref > 0 else None,
        })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,8,64",
                         help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_iters": args.iters,
        "batch_sizes": batch_sizes,
        "rmsnorm": bench_rmsnorm(batch_sizes, n_iters=args.iters),
        "matmul": bench_matmul(batch_sizes, n_iters=args.iters),
        "attention": bench_attention(batch_sizes, n_iters=args.iters),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

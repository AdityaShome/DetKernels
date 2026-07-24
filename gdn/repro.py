"""Reproduce vLLM issue #42960's core finding at the kernel level: FLA's
`chunk_gated_delta_rule` (the Triton kernel behind Gated-Delta-Net linear
attention, used in Qwen3-Next / Qwen3.6 hybrid models) is not
reduction-order invariant to batch composition -- the same sequence's output
can change depending on what else is batched alongside it, independent of
vLLM or model loading.

This mirrors this project's Phase 0/2 methodology (same-row-different-batch
comparison) applied to a real, currently-unsolved upstream kernel instead of
one we wrote ourselves. See vllm-project/vllm#42960 and the (as of writing,
unmerged, and per reviewer discussion incomplete) fix attempt in PR #45819.

Running this requires the `flash-linear-attention` package and a CUDA GPU --
there is no meaningful CPU path for this kernel, so this module is Colab/GPU
only (see tests/test_gdn_repro.py). Note: FLA's kernels operate in bfloat16
(matching real Qwen3-Next/3.6 usage, and matching the precision the upstream
bug reports are about) -- this may require an sm_80+ GPU (A100/L4), since
Tesla T4 (Turing, sm_75) lacks native bf16 tensor-core support.

Usage:
    detkernels-gdn-repro --batch-sizes 1,64 --repeats 5 --output gdn_repro.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone


def _make_inputs(batch, seq_len, h, hv, k_dim, v_dim, device, dtype, seed):
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq_len, h, k_dim, device=device, dtype=dtype, generator=g)
    k = F.normalize(
        torch.randn(batch, seq_len, h, k_dim, device=device, dtype=dtype, generator=g),
        p=2, dim=-1,
    )
    v = torch.randn(batch, seq_len, hv, v_dim, device=device, dtype=dtype, generator=g)
    beta = torch.rand(batch, seq_len, hv, device=device, dtype=dtype, generator=g).sigmoid()
    gate = torch.nn.functional.logsigmoid(
        torch.rand(batch, seq_len, hv, device=device, dtype=dtype, generator=g)
    )
    return q, k, v, beta, gate


def _run_row(tracked, batch_size, other_rows_seed, seq_len, h, hv, k_dim, v_dim, device, dtype):
    """Run chunk_gated_delta_rule with the tracked sequence at row 0 and
    (batch_size - 1) other random rows appended after it; return row 0's
    output. The tracked row's own (q, k, v, beta, gate) never changes --
    only what else is in the batch does."""
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q0, k0, v0, beta0, g0 = tracked
    if batch_size == 1:
        q, k, v, beta, gate = q0, k0, v0, beta0, g0
    else:
        qr, kr, vr, betar, gr = _make_inputs(
            batch_size - 1, seq_len, h, hv, k_dim, v_dim, device, dtype, other_rows_seed
        )
        q = torch.cat([q0, qr], dim=0)
        k = torch.cat([k0, kr], dim=0)
        v = torch.cat([v0, vr], dim=0)
        beta = torch.cat([beta0, betar], dim=0)
        gate = torch.cat([g0, gr], dim=0)

    o, _ = chunk_gated_delta_rule(q, k, v, gate, beta, output_final_state=False)
    return o[0]  # tracked row's output only: (seq_len, hv, v_dim)


def check_row_invariance(batch_sizes, seq_len: int = 128, h: int = 2, hv: int = 4,
                          k_dim: int = 64, v_dim: int = 64, device: str = "cuda",
                          dtype=None, seed: int = 0):
    """The core check: hold one sequence's (q, k, v, beta, gate) fixed, run
    it through chunk_gated_delta_rule embedded in batches of each size in
    `batch_sizes`, and compare its output across batch sizes. If the kernel
    were batch-invariant, every batch size would produce a bitwise-identical
    result for the tracked row -- vLLM #42960 / PR #45819 report that it
    does not."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    tracked = _make_inputs(1, seq_len, h, hv, k_dim, v_dim, device, dtype, seed)

    outputs = {
        bs: _run_row(tracked, bs, other_rows_seed=1000 + bs, seq_len=seq_len, h=h, hv=hv,
                     k_dim=k_dim, v_dim=v_dim, device=device, dtype=dtype)
        for bs in batch_sizes
    }

    reference_bs = batch_sizes[0]
    reference = outputs[reference_bs]
    comparisons = []
    for bs in batch_sizes[1:]:
        identical = torch.equal(reference, outputs[bs])
        max_abs_diff = (reference.float() - outputs[bs].float()).abs().max().item()
        comparisons.append({
            "batch_size": bs,
            "reference_batch_size": reference_bs,
            "identical": identical,
            "max_abs_diff": max_abs_diff,
        })
    return {"reference_batch_size": reference_bs, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1,
                         help="Repeat the whole check this many times with different "
                              "'other rows' content, to see if divergence is consistent "
                              "or intermittent (cf. Phase 0/1's atomic-ordering findings).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    all_results = []
    for r in range(args.repeats):
        res = check_row_invariance(batch_sizes, seq_len=args.seq_len, seed=r)
        all_results.append(res)
        for c in res["comparisons"]:
            print(f"repeat={r} batch_size={c['batch_size']} vs ref_bs={c['reference_batch_size']}: "
                  f"identical={c['identical']} max_abs_diff={c['max_abs_diff']:.6g}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_sizes": batch_sizes,
        "seq_len": args.seq_len,
        "repeats": args.repeats,
        "results": all_results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

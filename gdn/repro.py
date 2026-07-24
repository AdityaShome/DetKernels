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


def _make_seq(seq_len, h, hv, k_dim, v_dim, device, dtype, seed):
    """One sequence's (q, k, v, beta, gate), each shaped (seq_len, ..., dim)
    -- no batch dim, meant to be concatenated into a packed varlen batch."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(seq_len, h, k_dim, device=device, dtype=dtype, generator=g)
    k = F.normalize(
        torch.randn(seq_len, h, k_dim, device=device, dtype=dtype, generator=g),
        p=2, dim=-1,
    )
    v = torch.randn(seq_len, hv, v_dim, device=device, dtype=dtype, generator=g)
    beta = torch.rand(seq_len, hv, device=device, dtype=dtype, generator=g).sigmoid()
    gate = torch.nn.functional.logsigmoid(
        torch.rand(seq_len, hv, device=device, dtype=dtype, generator=g)
    )
    return q, k, v, beta, gate


def _run_packed(tracked_seq, other_seqs, tracked_position, device):
    """Pack `tracked_seq` among `other_seqs` (list of same-shaped tuples) at
    index `tracked_position` using cu_seqlens (vLLM's actual continuous-
    batching shape -- ragged sequences concatenated along the time axis, NOT
    a padded rectangular batch), run chunk_gated_delta_rule in varlen mode,
    and return the tracked sequence's output slice."""
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    seqs = list(other_seqs)
    seqs.insert(tracked_position, tracked_seq)

    lengths = [s[0].shape[0] for s in seqs]
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), dim=0).tolist()),
        device=device, dtype=torch.long,
    )

    q = torch.cat([s[0] for s in seqs], dim=0).unsqueeze(0)  # (1, T_total, H, K)
    k = torch.cat([s[1] for s in seqs], dim=0).unsqueeze(0)
    v = torch.cat([s[2] for s in seqs], dim=0).unsqueeze(0)
    beta = torch.cat([s[3] for s in seqs], dim=0).unsqueeze(0)
    gate = torch.cat([s[4] for s in seqs], dim=0).unsqueeze(0)

    o, _ = chunk_gated_delta_rule(q, k, v, gate, beta, output_final_state=False,
                                   cu_seqlens=cu_seqlens)

    start = cu_seqlens[tracked_position].item()
    end = cu_seqlens[tracked_position + 1].item()
    return o[0, start:end]  # (tracked_seq_len, hv, v_dim)


def check_packing_position_invariance(tracked_seq_len: int = 48,
                                       other_seq_lens=(37, 53, 29, 61),
                                       h: int = 2, hv: int = 4, k_dim: int = 64,
                                       v_dim: int = 64, device: str = "cuda",
                                       dtype=None, seed: int = 0):
    """The targeted hypothesis after check_row_invariance came back exactly
    identical (max_abs_diff=0.0) on a plain rectangular batch: does the
    tracked sequence's output depend on WHERE it sits within a packed
    (cu_seqlens) batch -- i.e. on the batch's "sequence geometry" -- even
    though its own (q, k, v, beta, gate) content never changes? This
    directly probes PR #45819's claim that FLA's chunked kernel's internal
    chunking depends on sequence geometry, using vLLM's actual ragged-
    packing shape instead of a padded rectangular one."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    tracked = _make_seq(tracked_seq_len, h, hv, k_dim, v_dim, device, dtype, seed)
    others = [
        _make_seq(length, h, hv, k_dim, v_dim, device, dtype, seed=100 + i)
        for i, length in enumerate(other_seq_lens)
    ]

    positions = list(range(len(others) + 1))
    outputs = {pos: _run_packed(tracked, others, pos, device) for pos in positions}

    reference_pos = positions[0]
    reference = outputs[reference_pos]
    comparisons = []
    for pos in positions[1:]:
        identical = torch.equal(reference, outputs[pos])
        max_abs_diff = (reference.float() - outputs[pos].float()).abs().max().item()
        comparisons.append({
            "tracked_position": pos,
            "reference_position": reference_pos,
            "identical": identical,
            "max_abs_diff": max_abs_diff,
        })
    return {"reference_position": reference_pos, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["row", "packing", "both"], default="both",
                         help="'row': rectangular same-length batch (came back "
                              "max_abs_diff=0.0 -- likely not the real bug path). "
                              "'packing': vLLM's actual cu_seqlens ragged-packing shape.")
    parser.add_argument("--batch-sizes", default="1,64", help="For --mode row.")
    parser.add_argument("--seq-len", type=int, default=128, help="For --mode row.")
    parser.add_argument("--tracked-seq-len", type=int, default=48, help="For --mode packing.")
    parser.add_argument("--other-seq-lens", default="37,53,29,61", help="For --mode packing.")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Repeat the whole check this many times with different "
                              "'other sequence' content, to see if divergence is consistent "
                              "or intermittent (cf. Phase 0/1's atomic-ordering findings).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    other_seq_lens = tuple(int(x) for x in args.other_seq_lens.split(","))

    row_results, packing_results = [], []
    for r in range(args.repeats):
        if args.mode in ("row", "both"):
            res = check_row_invariance(batch_sizes, seq_len=args.seq_len, seed=r)
            row_results.append(res)
            for c in res["comparisons"]:
                print(f"[row]     repeat={r} batch_size={c['batch_size']} "
                      f"vs ref_bs={c['reference_batch_size']}: "
                      f"identical={c['identical']} max_abs_diff={c['max_abs_diff']:.6g}")
        if args.mode in ("packing", "both"):
            res = check_packing_position_invariance(
                tracked_seq_len=args.tracked_seq_len, other_seq_lens=other_seq_lens, seed=r,
            )
            packing_results.append(res)
            for c in res["comparisons"]:
                print(f"[packing] repeat={r} tracked_position={c['tracked_position']} "
                      f"vs ref_pos={c['reference_position']}: "
                      f"identical={c['identical']} max_abs_diff={c['max_abs_diff']:.6g}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "batch_sizes": batch_sizes,
        "seq_len": args.seq_len,
        "tracked_seq_len": args.tracked_seq_len,
        "other_seq_lens": list(other_seq_lens),
        "repeats": args.repeats,
        "row_results": row_results,
        "packing_results": packing_results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

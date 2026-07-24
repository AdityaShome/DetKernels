"""Batch-invariant RMSNorm, and a deliberately batch-*variant* reference
kernel to demonstrate the failure mode it fixes.

Per "Defeating Nondeterminism in LLM Inference" (Thinking Machines, Sept
2025), RMSNorm nondeterminism arises when a kernel's reduction strategy for
the sum-of-squares over the hidden dimension depends on how many rows
(batch * seq_len) are being processed concurrently -- e.g. splitting a row's
reduction across multiple thread blocks (combined via atomics) specifically
when there ISN'T "enough" other row-level parallelism to keep the GPU busy
on its own. Atomic-add order across concurrently scheduled thread blocks is
a GPU-scheduler decision, not fixed by the kernel, so it reproduces the
*same-batch-size-sometimes-diverges* pattern actually observed in
docs/PHASE0_RESULTS.md (not just a fixed function of batch size).

Two kernel variants are provided:
- `rmsnorm_batch_invariant`: always one program per row, single-block
  reduction. A program's arithmetic depends only on its own row -- the grid
  size along the batch axis never enters the computation, so this is
  invariant by construction.
- `rmsnorm_batch_variant`: switches to a split-reduction-via-atomics path
  once the batch is too small to saturate the GPU via row-parallelism alone
  (`n_rows < split_threshold`), as a synthetic stand-in for the general class
  of occupancy/batch-size-dependent kernel dispatch described above --
  matching the direction used in matmul.py's split-K and attention.py's
  split-KV. This is NOT a reimplementation of vLLM's actual RMSNorm kernel --
  it's a minimal, honest reproduction of the mechanism (batch-size-dependent
  reduction order -> nondeterministic output), used as the "before" baseline
  to benchmark the fix against.

`torch`/`triton` are imported lazily so this module stays importable without
a GPU present; correctness/invariance is only verifiable on real hardware
(see tests/test_rmsnorm.py and notebooks/phase2_rmsnorm.ipynb).
"""
from __future__ import annotations

from typing import Optional


def rmsnorm_reference(x, weight, eps: float = 1e-6):
    """Plain PyTorch eager RMSNorm, float32 accumulation. Ground-truth math
    for correctness checks -- not a claim about its own determinism."""
    import torch

    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(var + eps)
    return (x_norm * weight.to(torch.float32)).to(orig_dtype)


def _get_triton():
    import triton
    import triton.language as tl

    return triton, tl


def _build_kernels():
    """Defines the Triton JIT kernels lazily (triton.jit decoration requires
    triton to already be importable), and caches them at module level."""
    triton, tl = _get_triton()

    @triton.jit
    def _single_block_kernel(
        x_ptr, weight_ptr, out_ptr,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row. This program's computation is a pure function
        # of its own row's data -- nothing here reads or depends on how many
        # other programs (rows) are in the launch grid, which is exactly what
        # makes this invariant to batch size.
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row_ptr = x_ptr + row * n_cols
        x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = x * rstd

        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_norm * w

        out_row_ptr = out_ptr + row * n_cols
        tl.store(out_row_ptr + col_offsets, y.to(x_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _split_partial_sumsq_kernel(
        x_ptr, partial_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        N_SPLITS: tl.constexpr,
    ):
        # grid = (n_rows, N_SPLITS). Each program sums the squares of its
        # chunk of the row and atomically adds into partial_ptr[row]. Atomic
        # add ordering across concurrently-scheduled programs is decided by
        # the GPU scheduler at runtime, not by this kernel -- the source of
        # genuine run-to-run (not just batch-size-to-batch-size)
        # nondeterminism.
        row = tl.program_id(0)
        split = tl.program_id(1)
        chunk = n_cols // N_SPLITS

        col_offsets = split * chunk + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < (split + 1) * chunk

        row_ptr = x_ptr + row * n_cols
        x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        partial_sum = tl.sum(x * x, axis=0)
        tl.atomic_add(partial_ptr + row, partial_sum)

    @triton.jit
    def _split_finalize_kernel(
        x_ptr, weight_ptr, partial_ptr, out_ptr,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        var = tl.load(partial_ptr + row) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        row_ptr = x_ptr + row * n_cols
        x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = x * rstd

        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_norm * w

        out_row_ptr = out_ptr + row * n_cols
        tl.store(out_row_ptr + col_offsets, y.to(x_ptr.dtype.element_ty), mask=mask)

    return _single_block_kernel, _split_partial_sumsq_kernel, _split_finalize_kernel


_KERNEL_CACHE = {}


def _kernels():
    if not _KERNEL_CACHE:
        single, partial, finalize = _build_kernels()
        _KERNEL_CACHE["single"] = single
        _KERNEL_CACHE["partial"] = partial
        _KERNEL_CACHE["finalize"] = finalize
    return _KERNEL_CACHE["single"], _KERNEL_CACHE["partial"], _KERNEL_CACHE["finalize"]


def rmsnorm_batch_invariant(x, weight, eps: float = 1e-6):
    """Batch-invariant RMSNorm: always one program per row, single-block
    reduction, fixed num_warps -- nothing about a row's computation depends
    on how many other rows are in the same launch."""
    import torch

    triton, tl = _get_triton()
    single_block_kernel, _, _ = _kernels()

    assert x.is_cuda, "rmsnorm_batch_invariant requires a CUDA tensor"
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x2d.shape
    out = torch.empty_like(x2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    NUM_WARPS = 8  # fixed regardless of n_rows -- part of the invariance guarantee

    single_block_kernel[(n_rows,)](
        x2d, weight, out, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    return out.reshape(orig_shape)


def rmsnorm_batch_variant(x, weight, eps: float = 1e-6, split_threshold: int = 16,
                           n_splits: int = 4):
    """Deliberately batch-*variant* RMSNorm: switches to a split-reduction-
    via-atomics path once `n_rows < split_threshold` (i.e. when the batch is
    too small to otherwise saturate the GPU via row-parallelism alone), as a
    synthetic reproduction of occupancy-driven, batch-size-dependent kernel
    dispatch -- matching the direction used in matmul.py's split-K and
    attention.py's split-KV. This is the "before" baseline -- see module
    docstring."""
    import torch

    triton, tl = _get_triton()
    single_block_kernel, partial_kernel, finalize_kernel = _kernels()

    assert x.is_cuda, "rmsnorm_batch_variant requires a CUDA tensor"
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x2d.shape
    out = torch.empty_like(x2d)

    if n_rows >= split_threshold:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        single_block_kernel[(n_rows,)](
            x2d, weight, out, n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        )
    else:
        chunk = -(-n_cols // n_splits)  # ceil div
        BLOCK_SIZE = triton.next_power_of_2(chunk)
        partial = torch.zeros(n_rows, device=x.device, dtype=torch.float32)
        partial_kernel[(n_rows, n_splits)](
            x2d, partial, n_cols, BLOCK_SIZE=BLOCK_SIZE, N_SPLITS=n_splits,
        )
        finalize_BLOCK_SIZE = triton.next_power_of_2(n_cols)
        finalize_kernel[(n_rows,)](
            x2d, weight, partial, out, n_cols, eps,
            BLOCK_SIZE=finalize_BLOCK_SIZE,
        )
    return out.reshape(orig_shape)

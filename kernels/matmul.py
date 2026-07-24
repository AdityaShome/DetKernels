"""Batch-invariant matmul, and a deliberately batch-*variant* reference
kernel to demonstrate the failure mode it fixes.

Per "Defeating Nondeterminism in LLM Inference" (Thinking Machines, Sept
2025) and the LLM-42 paper's follow-up analysis, matmul nondeterminism arises
because libraries like cuBLAS pick a reduction strategy for the K dimension
based on the overall problem shape -- in particular, when M (batch * seq_len)
is small relative to the GPU's thread-block count, there aren't enough
independent (m_tile, n_tile) output tiles to keep the GPU busy, so the
library splits K across additional thread blocks ("split-K") and combines
their partial sums via atomics to recover parallelism. When M is large
enough that M/N tiling alone already saturates the GPU, no split is needed
and each tile is reduced sequentially in one thread block. Which path runs
-- and therefore the floating-point summation order over K -- depends on M,
i.e. on batch size.

Two kernel variants are provided:
- `matmul_batch_invariant`: always the direct (non-split-K) kernel --  one
  program per (m_tile, n_tile) output tile, sequential reduction over K in a
  local accumulator. A tile's arithmetic depends only on its own tile, never
  on the total grid size, so this is invariant to M by construction,
  regardless of how small or large the batch is.
- `matmul_batch_variant`: switches to a split-K kernel (atomics across
  SPLIT_K partial sums) once M drops below `split_m_threshold`, as a
  synthetic stand-in for the occupancy-driven split-K dispatch described
  above. This is NOT a reimplementation of cuBLAS's actual heuristics -- it's
  a minimal, honest reproduction of the mechanism (batch-size-dependent
  reduction order -> nondeterministic output), used as the "before" baseline
  to benchmark the fix against. Split-K decomposition itself is also the
  specific technique flagged in project.md as missing from the naive
  batch-invariant reference (the 194 vs 527 TFLOPS gap) -- here it's
  deliberately misused (dispatched conditionally on M) rather than absent,
  since the goal is a determinism bug to fix, not peak throughput.

`torch`/`triton` are imported lazily so this module stays importable without
a GPU present; correctness/invariance is only verifiable on real hardware
(see tests/test_matmul.py).
"""
from __future__ import annotations


def matmul_reference(a, b):
    """Plain PyTorch eager matmul, float32 accumulation. Ground-truth math
    for correctness checks -- not a claim about its own determinism."""
    import torch

    orig_dtype = a.dtype
    out = a.to(torch.float32) @ b.to(torch.float32)
    return out.to(orig_dtype)


def _get_triton():
    import triton
    import triton.language as tl

    return triton, tl


def _build_kernels():
    """Defines the Triton JIT kernels lazily (triton.jit decoration requires
    triton to already be importable), and caches them at module level."""
    triton, tl = _get_triton()

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # One program per (m_tile, n_tile) output tile. K is reduced
        # sequentially, in the same order, regardless of how many other
        # tiles/programs are in the launch grid -- nothing here depends on
        # M beyond this program's own tile bounds, which is exactly what
        # makes this invariant to batch size.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
            b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

    @triton.jit
    def _matmul_splitk_partial_kernel(
        a_ptr, b_ptr, partial_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_pm, stride_pn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        # grid = (m_tiles, n_tiles, SPLIT_K). Each program reduces its own
        # chunk of K for a given output tile and atomically adds its partial
        # sum into partial_ptr. Atomic-add ordering across concurrently
        # scheduled programs is decided by the GPU scheduler at runtime, not
        # by this kernel -- the source of genuine run-to-run nondeterminism.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        k_chunk = tl.cdiv(K, SPLIT_K)
        k_start = pid_k * k_chunk
        k_end = tl.minimum(k_start + k_chunk, K)

        a_ptrs = a_ptr + rm[:, None] * stride_am + (k_start + rk[None, :]) * stride_ak
        b_ptrs = b_ptr + (k_start + rk[:, None]) * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, k_chunk, BLOCK_K):
            k_offset = k_start + k
            a_mask = (rm[:, None] < M) & (k_offset + rk[None, :] < k_end)
            b_mask = (k_offset + rk[:, None] < k_end) & (rn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        p_ptrs = partial_ptr + rm[:, None] * stride_pm + rn[None, :] * stride_pn
        p_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.atomic_add(p_ptrs, acc, mask=p_mask)

    return _matmul_kernel, _matmul_splitk_partial_kernel


_KERNEL_CACHE = {}


def _kernels():
    if not _KERNEL_CACHE:
        direct, splitk = _build_kernels()
        _KERNEL_CACHE["direct"] = direct
        _KERNEL_CACHE["splitk"] = splitk
    return _KERNEL_CACHE["direct"], _KERNEL_CACHE["splitk"]


def matmul_batch_invariant(a, b, block_m: int = 64, block_n: int = 64, block_k: int = 32):
    """Batch-invariant matmul: always the direct (non-split-K) kernel, one
    program per output tile, fixed sequential reduction over K -- nothing
    about a tile's computation depends on how many other tiles/rows (M) are
    in the same launch."""
    import torch

    triton, tl = _get_triton()
    direct_kernel, _ = _kernels()

    assert a.is_cuda and b.is_cuda, "matmul_batch_invariant requires CUDA tensors"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match, got {K} and {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    direct_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c


def matmul_batch_variant(a, b, split_m_threshold: int = 64, split_k: int = 4,
                          block_m: int = 64, block_n: int = 64, block_k: int = 32):
    """Deliberately batch-*variant* matmul: switches to a split-K-via-atomics
    path once `M < split_m_threshold`, as a synthetic reproduction of
    occupancy-driven, batch-size-dependent kernel dispatch. This is the
    "before" baseline -- see module docstring."""
    import torch

    triton, tl = _get_triton()
    direct_kernel, splitk_kernel = _kernels()

    assert a.is_cuda and b.is_cuda, "matmul_batch_variant requires CUDA tensors"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match, got {K} and {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if M >= split_m_threshold:
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        direct_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
    else:
        partial = torch.zeros((M, N), device=a.device, dtype=torch.float32)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), split_k)
        splitk_kernel[grid](
            a, b, partial,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            partial.stride(0), partial.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        )
        c.copy_(partial.to(a.dtype))
    return c

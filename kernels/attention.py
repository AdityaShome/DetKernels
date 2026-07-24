"""Batch-invariant single-query (decode-step) attention, and a deliberately
batch-*variant* reference kernel to demonstrate the failure mode it fixes.

This models the decode-phase attention shape used in autoregressive LLM
inference: one query vector per (batch item, head) attending over an
already-materialized KV cache of length `seq_len`. Per "Defeating
Nondeterminism in LLM Inference" (Thinking Machines, Sept 2025) and the
open vLLM/SGLang issues cited in project.md (vLLM #25404, SGLang #10278 /
#11513), attention nondeterminism at decode time arises from "split-KV" /
FlashDecoding-style dispatch: when the batch (and therefore total
batch*heads work) is too small to saturate the GPU on its own, the KV
sequence is additionally split across extra thread blocks to recover
parallelism, and those partial (running-max, running-sum, weighted-V)
statistics are combined via atomics whose order is decided by the GPU
scheduler at runtime -- not fixed by the kernel. When the batch is large
enough, no KV split is needed and each (batch, head) is reduced sequentially
by a single program. Which path runs (and therefore the float summation
order over the KV sequence) depends on batch size.

Two kernel variants are provided:
- `attention_batch_invariant`: always the direct kernel -- one program per
  (batch, head), sequential online-softmax reduction over the whole KV
  sequence in a fixed block order. A program's arithmetic depends only on
  its own (batch, head) row, never on how many other programs are in the
  launch grid, so this is invariant to batch size by construction.
- `attention_batch_variant`: switches to a two-pass split-KV kernel (atomics
  across SPLIT_S partial max/sum/weighted-V combinations) once
  `batch * num_heads < split_bh_threshold`, as a synthetic stand-in for
  occupancy-driven split-KV dispatch (e.g. FlashDecoding). This is NOT a
  reimplementation of vLLM's actual PagedAttention kernel -- it's a minimal,
  honest reproduction of the mechanism (batch-size-dependent reduction order
  -> nondeterministic output), used as the "before" baseline to benchmark
  the fix against.

Simplifications vs. a real production kernel (documented for honesty, same
spirit as kernels/rmsnorm.py and kernels/matmul.py): no causal masking (the
KV cache is already the full context at decode time, so every key is
attended to), no paging (KV is a plain contiguous tensor, not vLLM's
block-table-indexed PagedAttention), single query per row (no prefill/
multi-token chunking), fp32 accumulation throughout.

`torch`/`triton` are imported lazily so this module stays importable without
a GPU present; correctness/invariance is only verifiable on real hardware
(see tests/test_attention.py).
"""
from __future__ import annotations


def attention_reference(q, k, v, scale: float | None = None):
    """Plain PyTorch eager decode-step attention, float32 accumulation.
    Ground-truth math for correctness checks -- not a claim about its own
    determinism.

    q: (B, H, D) -- one query vector per (batch, head).
    k, v: (B, H, S, D) -- KV cache of length S per (batch, head).
    """
    import torch

    orig_dtype = q.dtype
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d ** 0.5)

    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)

    scores = torch.einsum("bhd,bhsd->bhs", q32, k32) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", probs, v32)
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
    def _attn_direct_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_od,
        seq_len, head_dim, scale,
        BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # One program per (batch, head). The whole KV sequence is reduced
        # sequentially, in fixed block order, via online (flash-attention
        # style) softmax -- nothing here depends on how many other programs
        # (i.e. how large the batch is) are in the launch grid.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim

        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_offsets * stride_qd
        q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for s_start in range(0, seq_len, BLOCK_S):
            s_offsets = s_start + tl.arange(0, BLOCK_S)
            s_mask = s_offsets < seq_len

            k_ptrs = (k_ptr + pid_b * stride_kb + pid_h * stride_kh
                      + s_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd)
            k_mask = s_mask[:, None] & d_mask[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(s_mask, scores, float("-inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            l_i = alpha * l_i + tl.sum(p, axis=0)

            v_ptrs = (v_ptr + pid_b * stride_vb + pid_h * stride_vh
                      + s_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd)
            v_mask = s_mask[:, None] & d_mask[None, :]
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + d_offsets * stride_od
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=d_mask)

    @triton.jit
    def _attn_split_max_kernel(
        q_ptr, k_ptr, m_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_mb, stride_mh,
        seq_len, head_dim, scale,
        BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, SPLIT_S: tl.constexpr,
    ):
        # Pass 1/2 of the split-KV path. grid = (B, H, SPLIT_S). Each program
        # computes the max score over its own chunk of the KV sequence and
        # combines it into a global per-(batch,head) max via atomic_max.
        # Max is associative/commutative bit-for-bit for IEEE floats, so this
        # particular reduction is not itself a source of nondeterminism --
        # the nondeterminism comes from the sum/weighted-V combination in
        # the second pass below.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_offsets * stride_qd
        q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        chunk = tl.cdiv(seq_len, SPLIT_S)
        s_start = pid_s * chunk
        s_end = tl.minimum(s_start + chunk, seq_len)

        local_max = float("-inf")
        for s0 in range(0, chunk, BLOCK_S):
            s_offsets = s_start + s0 + tl.arange(0, BLOCK_S)
            s_mask = s_offsets < s_end

            k_ptrs = (k_ptr + pid_b * stride_kb + pid_h * stride_kh
                      + s_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd)
            k_mask = s_mask[:, None] & d_mask[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(s_mask, scores, float("-inf"))
            local_max = tl.maximum(local_max, tl.max(scores, axis=0))

        tl.atomic_max(m_ptr + pid_b * stride_mb + pid_h * stride_mh, local_max)

    @triton.jit
    def _attn_split_sumacc_kernel(
        q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, acc_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_mb, stride_mh,
        stride_lb, stride_lh,
        stride_accb, stride_acch, stride_accd,
        seq_len, head_dim, scale,
        BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, SPLIT_S: tl.constexpr,
    ):
        # Pass 2/2 of the split-KV path. Given the finalized global max from
        # _attn_split_max_kernel, each program computes exp(scores - m) and
        # the weighted-V sum over its own KV chunk, then atomically adds its
        # partial (sum-of-exp, weighted-V) into shared accumulators. The
        # order in which concurrently scheduled programs perform these
        # atomic adds is a GPU-scheduler decision -- the source of genuine
        # run-to-run (not just batch-size-to-batch-size) nondeterminism.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_offsets * stride_qd
        q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        m = tl.load(m_ptr + pid_b * stride_mb + pid_h * stride_mh)

        chunk = tl.cdiv(seq_len, SPLIT_S)
        s_start = pid_s * chunk
        s_end = tl.minimum(s_start + chunk, seq_len)

        local_l = 0.0
        local_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for s0 in range(0, chunk, BLOCK_S):
            s_offsets = s_start + s0 + tl.arange(0, BLOCK_S)
            s_mask = s_offsets < s_end

            k_ptrs = (k_ptr + pid_b * stride_kb + pid_h * stride_kh
                      + s_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd)
            k_mask = s_mask[:, None] & d_mask[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(s_mask, scores, float("-inf"))
            p = tl.exp(scores - m)
            p = tl.where(s_mask, p, 0.0)
            local_l += tl.sum(p, axis=0)

            v_ptrs = (v_ptr + pid_b * stride_vb + pid_h * stride_vh
                      + s_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd)
            v_mask = s_mask[:, None] & d_mask[None, :]
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
            local_acc += tl.sum(p[:, None] * v, axis=0)

        tl.atomic_add(l_ptr + pid_b * stride_lb + pid_h * stride_lh, local_l)
        acc_ptrs = acc_ptr + pid_b * stride_accb + pid_h * stride_acch + d_offsets * stride_accd
        tl.atomic_add(acc_ptrs, local_acc, mask=d_mask)

    return _attn_direct_kernel, _attn_split_max_kernel, _attn_split_sumacc_kernel


_KERNEL_CACHE = {}


def _kernels():
    if not _KERNEL_CACHE:
        direct, split_max, split_sumacc = _build_kernels()
        _KERNEL_CACHE["direct"] = direct
        _KERNEL_CACHE["split_max"] = split_max
        _KERNEL_CACHE["split_sumacc"] = split_sumacc
    return _KERNEL_CACHE["direct"], _KERNEL_CACHE["split_max"], _KERNEL_CACHE["split_sumacc"]


def _run_direct(q, k, v, scale, block_s):
    import torch

    triton, tl = _get_triton()
    direct_kernel, _, _ = _kernels()

    B, H, D = q.shape
    _, _, S, _ = k.shape
    out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D)

    direct_kernel[(B, H)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        S, D, scale,
        BLOCK_S=block_s, BLOCK_D=BLOCK_D,
    )
    return out


def attention_batch_invariant(q, k, v, scale: float | None = None, block_s: int = 64):
    """Batch-invariant decode-step attention: always the direct kernel, one
    program per (batch, head), fixed sequential online-softmax reduction
    over the KV sequence -- nothing about a row's computation depends on how
    many other rows (batch size) are in the same launch."""
    assert q.is_cuda and k.is_cuda and v.is_cuda, "attention_batch_invariant requires CUDA tensors"
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    return _run_direct(q, k, v, scale, block_s)


def attention_batch_variant(q, k, v, scale: float | None = None,
                             split_bh_threshold: int = 32, split_s: int = 4,
                             block_s: int = 64):
    """Deliberately batch-*variant* decode-step attention: switches to a
    two-pass split-KV kernel (atomics across SPLIT_S chunks) once
    `batch * num_heads < split_bh_threshold`, as a synthetic reproduction of
    occupancy-driven, batch-size-dependent split-KV dispatch (e.g.
    FlashDecoding). This is the "before" baseline -- see module docstring."""
    import torch

    triton, tl = _get_triton()
    _, split_max_kernel, split_sumacc_kernel = _kernels()

    assert q.is_cuda and k.is_cuda and v.is_cuda, "attention_batch_variant requires CUDA tensors"
    B, H, D = q.shape
    _, _, S, _ = k.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if B * H >= split_bh_threshold:
        return _run_direct(q, k, v, scale, block_s)

    BLOCK_D = triton.next_power_of_2(D)
    m = torch.full((B, H), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((B, H), device=q.device, dtype=torch.float32)
    acc = torch.zeros((B, H, D), device=q.device, dtype=torch.float32)

    grid = (B, H, split_s)
    split_max_kernel[grid](
        q, k, m,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        m.stride(0), m.stride(1),
        S, D, scale,
        BLOCK_S=block_s, BLOCK_D=BLOCK_D, SPLIT_S=split_s,
    )
    split_sumacc_kernel[grid](
        q, k, v, m, l, acc,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        m.stride(0), m.stride(1),
        l.stride(0), l.stride(1),
        acc.stride(0), acc.stride(1), acc.stride(2),
        S, D, scale,
        BLOCK_S=block_s, BLOCK_D=BLOCK_D, SPLIT_S=split_s,
    )
    return (acc / l.unsqueeze(-1)).to(q.dtype)

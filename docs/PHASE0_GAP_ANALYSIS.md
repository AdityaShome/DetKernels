# Phase 0 — Prior Art Search & Target Gap Decision

Satisfies Phase 0 gate requirement (c): "a documented decision on which specific
gap you're attacking and why, based on the search in step 4."

## Landscape as of 2026-07-20 (10 months after the original Sept 2025 posts)

The brief was written against the Sept 2025 state of the world. Re-checked against
current GitHub/arXiv state before committing further:

### vLLM
Batch invariance is no longer a proof-of-concept — it shipped as an official,
documented feature (`VLLM_BATCH_INVARIANT=1`, see
[docs.vllm.ai/en/stable/features/batch_invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)),
tracked to near-completion in
[vllm-project/vllm#27433](https://github.com/vllm-project/vllm/issues/27433):

- Done: FlashInfer backend, DeepSeek-V3, DeepGEMM on Blackwell, R1 @ TP8 on
  Blackwell, torch.compile/CUDA graph support, TRITON_MLA, and **performance
  optimizations already landed** (fused BMM, fused RMSNorm, Cutlass fp8).
- Still open: FLASHINFER_MLA support (blocked on external FlashInfer team
  involvement — not independently attackable), general "more perf work welcome."

Implication: the brief's core performance-gap target (194 -> 527 TFLOPS via
split-K/TMA/warp specialization on a from-scratch Triton kernel) is **partially
already closed** by vLLM's own merged optimizations. Building our own kernels
(Phase 2) is still worth doing for the harness integration and for having our own
measured numbers, but claiming to close this gap "first" would not be honest —
the writeup needs to say plainly that vLLM ships this natively now.

### SGLang
Both issues named explicitly in the brief are **already closed**:

- [#11513](https://github.com/sgl-project/sglang/issues/11513) (deterministic
  inference broken on Blackwell TP4) — closed. Was reproducible on B200/GB200 at
  TP4 for Qwen3-8B and Qwen3-30B-A3B, opened Oct 2025, since fixed.
- [#10785](https://github.com/sgl-project/sglang/issues/10785) (MoE nondeterministic
  at large TP, Qwen3-30B-A3B TP>=4 producing 5-77 unique outputs instead of 1) —
  closed, fixed via PR #10930.

Implication: two of the brief's three named fallback angles no longer exist as
open problems. Re-verify before relying on either.

### The gap that is still open and fresh
[vllm-project/vllm#42960](https://github.com/vllm-project/vllm/issues/42960)
(filed May 2026, **open**): batch-invariant mode hard-fails at startup for
Qwen3-Next / Qwen3.6 hybrid Mamba + Gated-Delta-Net (GDN) linear-attention models —
`"VLLM batch_invariant mode is not supported for GDN_ATTN"`, no fallback path. The
PR that added SM80 batch-invariant support for standard attention (#42456)
explicitly did not cover the linear-attention path. Root cause lives in
`vllm/v1/attention/selector.py:154`.

This matches the brief's own suggested fallback ("extend determinism to a
currently-broken case") more precisely than the original MoE-at-large-TP framing,
since that specific MoE case is now fixed. GDN/linear-attention is a different
reduction pattern than the standard softmax attention Thinking Machines/SGLang
already solved, so it is a genuinely distinct technical problem, not a rehash.

## Decision

**Target gap for Phase 3: extend batch-invariant determinism to GDN_ATTN
(hybrid Mamba + Gated-Delta-Net linear attention, as used in Qwen3-Next/3.6).**

Phase 1-2 proceed as planned (harness + from-scratch RMSNorm/matmul/attention),
but the Phase 2 writeup must explicitly acknowledge vLLM's native
`VLLM_BATCH_INVARIANT=1` as prior art rather than presenting the from-scratch
kernels as closing a gap nobody else has closed. The actual novel contribution is
(a) the reusable measurement harness, and (b) closing the GDN_ATTN gap in Phase 3.

**Why:** it's the most concrete, current, independently-attackable open gap found
in this search — no external team dependency (unlike FLASHINFER_MLA), not already
fixed (unlike both named SGLang issues), and technically distinct from the
already-solved softmax-attention case.

**How to apply:** Phase 3 scope = get vLLM's `VLLM_BATCH_INVARIANT` mode running,
reproduce the GDN_ATTN startup failure locally, then design and implement a
batch-invariant reduction strategy for the GDN linear-attention recurrence
(different from standard attention's softmax reduction). Track progress against
issue #42960 upstream — check periodically whether someone else lands a fix first,
and re-run this search before finalizing Phase 3 in case the landscape moves again.

See [DECISIONS.md](./DECISIONS.md) for the compute-path decision alongside this.

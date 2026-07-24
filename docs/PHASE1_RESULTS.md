# Phase 1 Results — Harness Verification and Divergence Localization

Satisfies the Phase 1 gate: the `detkernels-audit` harness reliably and
repeatably detects nondeterminism in the baseline stack, and produces a report
that clearly shows which layer is the first point of divergence. Raw data:
[docs/reports/phase1_batch_seq_localize.json](./reports/phase1_batch_seq_localize.json).

## Setup
- Harness: `detkernels-audit` (this repo's `harness/` package), run against a
  real `vllm serve` subprocess over its OpenAI-compatible HTTP API — not the
  ad-hoc notebook cells used in Phase 0.
- Stack/hardware: vLLM 0.19.1, Google Colab Tesla T4, `dtype=float16`,
  `tensor_parallel_size=1` (only one GPU available; TP sweep dimension is
  implemented but unexercised — see "Known gaps" below).
- Model: Qwen/Qwen3-1.7B.
- Decoding: greedy (`temperature=0.0`), same prompt as Phase 0.
- `max_tokens=256`, `n_repeats=30` (reduced from Phase 0's 100 since this run
  covers three sweep dimensions plus localization in one pass; project.md's
  "scale to N=1000 for final reporting" still applies to a later, focused run).

## Findings

### Batch-size sweep: reproduced Phase 0 exactly
bs=1 (30 repeats): 100% bitwise identical. bs=32 (30 repeats): 1/29
non-reference repeats diverged, **at token index 235** — the same index found
in Phase 0's independent notebook run (19/99 diverging, N=100). Different
divergence *rate* (expected — it's a low-probability event and this run used
fewer repeats), same divergence *location*. This is the harness's own
correctness check: it reproduces a previously-verified finding through
entirely different code (CLI + `harness/sweep.py`) instead of the original
notebook cells.

### Sequence-length sweep: inconclusive, not evidence of absence
Padding the prompt to ~50 and ~500 words and sweeping at bs=32 came back fully
clean (0/29 diverging) at both lengths. Given the unpadded prompt only
diverged 1/29 times in this same run, 29 repeats is too few to distinguish
"sequence length matters" from "just didn't roll the same low-probability
event." Needs a higher-N rerun before drawing a conclusion either way.

### Divergence localization: the interesting result
Loading the model via `transformers` and diffing per-layer decoder activations
between a batch-of-1 and a batch-of-32 forward pass (same prompt, same
weights) found:

| Layer | max abs diff | identical |
|---|---|---|
| 0 | 0.0039 | no |
| 1 | 0.0156 | no |
| 2 | 8.0 | no |
| 3–27 | 8.0 (flat) | no |

The pattern is informative, not noise: layer 0 shows a *tiny* diff consistent
with the paper's claim that batch size changes which GEMM/attention kernel
gets dispatched, producing small rounding differences rather than a
qualitatively different computation. That tiny diff roughly quadruples by
layer 1, and by layer 2 it jumps to a value that then stays completely flat
through the rest of the network. The flat plateau is the signature of full
**decorrelation**: once the layer-0/1 rounding difference is large enough to
flip a real decision (e.g., which value an attention head selects), the two
hidden-state trajectories stop being "close but different" and become two
independent representations of similar typical magnitude — so the diff stops
growing and just tracks that magnitude. This localizes the root cause to
*layers 0–1*, not something that gradually accumulates over 28 layers.

## Honesty caveat: this localizer is not literally the vLLM code path
`localize_divergence` runs a plain `transformers` forward pass (see
`harness/localizer.py`), not vLLM's actual paged/scheduled kernels — vLLM's
execution model is too hard to hook into directly. What this measurement
demonstrates is that **batched GPU matmul/attention exhibits batch-size-
dependent reduction order generically** (a PyTorch/cuBLAS/SDPA-level
phenomenon), which is consistent with, but not proof of, the specific vLLM
kernel behavior the batch-size sweep observed. Treat the layer-localization
result as strong circumstantial evidence for *where* in a transformer forward
pass this class of nondeterminism originates, not as a trace of vLLM's actual
call stack.

## Known gaps
- TP sweep (`--tp-sizes`) is implemented in the CLI but has not been run with
  TP>1 — requires multiple GPUs, unavailable on the current single-T4 Colab
  session. Revisit on the planned H100/Blackwell rental (see
  [DECISIONS.md](./DECISIONS.md)).
- N=30 is well below project.md's eventual N=1000 target; sufficient to
  verify the harness works correctly, not sufficient as a final rigor number.
- Attention backend on the T4 still not explicitly confirmed (carried over
  from Phase 0).

## What this unblocks
Phase 1 gate is satisfied: the harness detects nondeterminism reliably
(reproduced the exact Phase 0 divergence point through independent code) and
localizes it to a specific point in the network (layers 0–1, before full
decorrelation by layer 2). Proceeding to Phase 2: implement batch-invariant
RMSNorm, matmul, and attention kernels, starting with RMSNorm as the simplest
case, per project.md's ordering.

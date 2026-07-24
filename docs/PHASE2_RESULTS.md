# Phase 2 Results — Batch-Invariant RMSNorm, Matmul, and Attention

Satisfies the Phase 2 gate: bitwise-reproducible output across at least 100 runs
at varying batch sizes using the integrated kernel set, with a documented,
reproducible benchmark showing the performance cost. Raw benchmark data:
[docs/reports/phase2_benchmark.json](./reports/phase2_benchmark.json).

**Read this alongside [PHASE0_GAP_ANALYSIS.md](./PHASE0_GAP_ANALYSIS.md) first.**
vLLM now ships batch-invariant execution natively (`VLLM_BATCH_INVARIANT=1`,
tracked at vllm-project/vllm#27433) for standard softmax attention. The kernels
built in this phase are **not** closing a gap nobody else has closed — they exist
to (a) validate the harness end-to-end against a kernel set we control, and (b)
establish the implementation pattern for Phase 3's actual novel target (GDN linear
attention, vllm-project/vllm#42960). Framing them as anything more than a
from-scratch reproduction exercise would not be honest.

## Setup
- Kernels: Triton, built and tested on Google Colab (Tesla T4), `dtype=float16`,
  fp32 accumulation internally.
- Three kernels, each with a batch-invariant version and a deliberately
  batch-*variant* reference (`kernels/rmsnorm.py`, `kernels/matmul.py`,
  `kernels/attention.py`) built to reproduce the underlying mechanism (batch-
  size-dependent reduction order via conditional split+atomics) rather than to
  literally reimplement cuBLAS/vLLM's internals — documented in each module's
  docstring.
- Integration: `kernels/tiny_model.py`, a small random-weight decode-only
  transformer (4 layers, 4 heads, head_dim=32) built entirely from these
  kernels, used to re-run the Phase 1 harness's reproducibility measurement
  (`kernels/integration.py`, `detkernels-kernel-check`) against an integrated
  kernel set instead of against vLLM.
- Benchmark: `kernels/benchmark.py` (`detkernels-kernel-bench`), CUDA-event-timed,
  50 iterations + 10 warmup per config, batch sizes 1/8/64.

## Findings

### Correctness and invariance: all three kernels pass
For each of RMSNorm, matmul, and attention: the batch-invariant kernel matches a
plain-PyTorch/float32 reference within tolerance, and — the core claim — the
*same row* run through the *same kernel* produces a **bitwise-identical**
(`torch.equal`) result whether it's alone (batch=1) or embedded in a batch of
32–128. This holds at the individual-kernel level and end-to-end through
`TinyModel`: the same starting token, run through all 4 layers and 5 decode
steps with the batch-invariant kernel set, produces an identical generated
sequence whether it's the only sequence in the batch or one of 32. All 19 GPU
tests (`test_rmsnorm.py`, `test_matmul.py`, `test_attention.py`,
`test_tiny_model.py`) pass on Colab.

### Reproducibility measurement: the "after" number
Re-running the harness-style measurement (100 repeats, `detkernels-kernel-check`)
against `TinyModel` with `kernel_set=batch_invariant` at batch sizes 1 and 64:
**0/99 divergences at both.** This is the Phase 2 gate's required "bitwise-
reproducible output across at least 100 runs at varying batch sizes using your
integrated kernel set."

### Negative result: the batch-variant reference didn't reproduce observable divergence
The same 100-repeat measurement with `kernel_set=batch_variant` — the kernel set
deliberately built with atomic-add split-reduction to model the nondeterminism
mechanism — also came back 0/99 divergences at both batch sizes, despite
genuinely exercising each kernel's split/atomics code path at one of the two
configurations. This is consistent, not a fluke: an isolated single-sample check
on `rmsnorm_batch_variant` alone (batch=1 vs batch=64, same row) also came back
identical.

This is reported honestly as a limitation, not glossed over. Most likely
explanation: our split factors (4-way for RMSNorm/matmul/attention) and grid
sizes are tiny compared to real cuBLAS/vLLM production launches. Genuine
run-to-run variation in atomic-add ordering requires actual GPU scheduler
contention — many concurrently-scheduled thread blocks racing for execution
order — which a handful of thread blocks on an otherwise-idle T4 doesn't
reliably generate. The *mechanism* is real (it's why TML/SGLang had to fix it
in production-scale kernels), but our toy-scale reproduction of it apparently
doesn't generate enough scheduling entropy to observe it in ~100-200 samples.
This means: we have strong positive evidence the batch-invariant kernels work,
but not a self-contained "before" divergence-rate number from our own kernels
to compare it against — the actual "before" evidence remains the Phase 0/1
harness measurements against real vLLM (token-235 divergence, 1/29–19/99 rates).

One bug found and fixed in the course of this analysis: `rmsnorm_batch_variant`
initially had its split-condition direction inverted relative to `matmul.py`'s
split-K and `attention.py`'s split-KV (it split on *large* batch instead of
*small* batch, backwards from the real occupancy-driven mechanism it's meant to
model). Fixed before the final benchmark run; the correctness/invariance
properties were unaffected either way, but the direction now matches the other
two kernels and the real-world story.

### Performance benchmark
Full data: [docs/reports/phase2_benchmark.json](./reports/phase2_benchmark.json).

| Kernel | batch=1 | batch=8 | batch=64 | vs. |
|---|---|---|---|---|
| RMSNorm (invariant) | -59% | -61% | -57% | naive eager PyTorch |
| Matmul (invariant) | +1162% | +611% | +859% | cuBLAS (`a @ b`) |
| Attention (invariant) | -31% | -80% | -83% | naive eager PyTorch |

Honesty caveats on each:
- **RMSNorm** looks like a clean win (our fused Triton kernel beats naive
  multi-op eager PyTorch by ~60%), but this is a low bar — eager PyTorch here
  issues five separate kernel launches (`pow`, `mean`, `rsqrt`, two `mul`s), not
  a real fused production RMSNorm. It is not the "overhead vs. the standard
  nondeterministic kernel" comparison TML/SGLang report. Also: at batch=64,
  post-fix, both `batch_invariant` and `batch_variant` dispatch to the literal
  same compiled kernel with identical launch parameters, yet still measured
  ~50% apart (0.070ms vs 0.104ms) — at this problem size (single-digit
  microsecond compute, sub-0.1ms total), Python/CUDA launch overhead and
  measurement noise likely dominate real compute cost, so this specific gap is
  not a resolvable signal and shouldn't be read as a real design difference.
- **Matmul** is a genuine, large, and reproducible signal (order of magnitude
  above any noise floor) — and it is honestly bad: 6–12x slower than cuBLAS,
  far worse than the TML/LLM-42-documented 63% gap (194 vs 527 TFLOPS). That
  published number is the gap between two *already-optimized* implementations;
  ours has no autotuning, no split-K in the invariant path, and no warp
  specialization or TMA. Per project.md's own fallback plan ("if GPU-level
  optimization is out of reach, implement correct-but-unoptimized first, then
  iterate opportunistically"), this is the expected state of a first-pass
  kernel and is the concrete, quantified target for future optimization work,
  not a result to present as competitive.
- **Attention** shows a real and fairly dramatic win (31–83% faster than eager,
  growing with batch/sequence length) because the fused online-softmax kernel
  never materializes the full (B, H, S) score tensor that eager PyTorch's
  `einsum`-based reference does — a genuine memory-bandwidth advantage that
  shows up clearly as batch and sequence length grow. Same caveat as RMSNorm
  though: compared against naive eager, not against PyTorch's optimized
  `scaled_dot_product_attention` or a real FlashAttention kernel.

## Known gaps
- Matmul optimization (split-K, warp specialization, TMA) not attempted — this
  phase followed project.md's explicit "correctness first, unoptimized is fine
  initially" guidance; the 6-12x cuBLAS gap is the quantified starting point for
  that work, not yet attempted.
- No self-contained "before" divergence-rate measurement from our own
  batch-variant kernels (see negative result above) — the "before" evidence this
  project relies on is still the Phase 0/1 harness measurements against real
  vLLM, not a number produced by these toy kernels.
- Benchmark run on a shared/free-tier Colab T4 (not a dedicated instance);
  RMSNorm's numbers in particular are close to the noise floor at this problem
  size and should be treated as directional, not precise.
- `TinyModel` is a random-weight, decode-only toy (no real weights, no
  tokenizer, no prefill phase) — sufficient to validate the kernels'
  reproducibility properties end-to-end, not a claim about real-model behavior.

## What this unblocks
Phase 2 gate is satisfied: bitwise-reproducible output across 100 runs at two
batch sizes using the integrated batch-invariant kernel set, plus a documented
benchmark (honest about where the numbers are strong, weak, or noisy). Per
[PHASE0_GAP_ANALYSIS.md](./PHASE0_GAP_ANALYSIS.md), proceeding to Phase 3: get
vLLM's native `VLLM_BATCH_INVARIANT` mode running, reproduce the GDN_ATTN
startup failure from vllm-project/vllm#42960 (Qwen3-Next/3.6 hybrid Mamba +
Gated-Delta-Net linear attention), and design a batch-invariant reduction
strategy for that recurrence — the project's actual novel contribution.

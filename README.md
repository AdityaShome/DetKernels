# DetKernels

Making LLM inference **bitwise-reproducible** across batch sizes, batch
composition, and tensor-parallel configurations by measuring exactly where
nondeterminism enters a real inference stack, then closing the gap with
batch-invariant kernels.

Nondeterministic inference (the same prompt + greedy decoding producing
different tokens run-to-run) comes from batch-size-dependent reduction order
in GPU kernels (RMSNorm, matmul, attention), not floating-point
non-associativity per se see Thinking Machines Lab's ["Defeating
Nondeterminism in LLM Inference"](https://thinkingmachines.ai) and SGLang's
follow-up. This repo is a from-scratch reproduction and extension of that
work: **vLLM already ships a native fix** (`VLLM_BATCH_INVARIANT=1`,
[vllm-project/vllm#27433](https://github.com/vllm-project/vllm/issues/27433))
for standard attention this project doesn't claim to have discovered that.
What it does:

1. **A reproducibility measurement harness** (`harness/`) quantifies and
   localizes nondeterminism in a real vLLM server across batch size,
   tensor-parallel size, and sequence length.
2. **Batch-invariant kernels** (`kernels/`) from-scratch Triton
   implementations of RMSNorm, matmul, and attention, each paired with a
   deliberately batch-*variant* twin, integrated into a standalone toy
   transformer to prove bitwise reproducibility end-to-end.
3. **An extension to a currently-unsolved case** (`gdn/`) vLLM's native
   batch-invariant mode hard-fails on Gated-Delta-Net linear attention
   (Qwen3-Next / Qwen3.6 hybrid Mamba+MoE models,
   [vllm-project/vllm#42960](https://github.com/vllm-project/vllm/issues/42960)).
   This is the project's actual novel target in progress.

See [`project.md`](./project.md) for the full phased build brief and
[`docs/`](./docs) for phase-by-phase results, including honest negative
results and known gaps.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Feasibility check, reproduce baseline nondeterminism | Done [docs/PHASE0_RESULTS.md](./docs/PHASE0_RESULTS.md), [docs/PHASE0_GAP_ANALYSIS.md](./docs/PHASE0_GAP_ANALYSIS.md) |
| 1 | Reproducibility measurement harness | Done [docs/PHASE1_RESULTS.md](./docs/PHASE1_RESULTS.md) |
| 2 | Batch-invariant RMSNorm / matmul / attention kernels | Done [docs/PHASE2_RESULTS.md](./docs/PHASE2_RESULTS.md) |
| 3 | Extend determinism to GDN linear attention (vLLM #42960) | In progress |
| 4 | Polish, publish, writeup | Not started |

## Requirements

- Python >= 3.10
- A CUDA GPU for anything in `kernels/` or `gdn/` (Triton has no meaningful
  CPU path here); `harness/` additionally needs a GPU capable of running
  vLLM. This project has been developed against a free-tier Google Colab
  Tesla T4 see [docs/DECISIONS.md](./docs/DECISIONS.md) for the compute
  strategy.
- On CPU-only / no-GPU machines, all GPU-only tests skip cleanly (they don't
  fail) via layered `importorskip`/`cuda.is_available()` guards.

## Install

```bash
pip install -e ".[dev]"          # harness core + test runner, CPU-only dev
pip install -e ".[localizer]"    # + transformers/accelerate, for divergence localization
pip install -e ".[kernels]"      # + torch/triton, for kernels/
pip install -e ".[gdn]"          # + flash-linear-attention, for gdn/
```

## Quickstart

### 1. Measure nondeterminism in a real vLLM server

```bash
detkernels-audit --model Qwen/Qwen3-1.7B --batch-sizes 1,32 --runs 100 \
    --output report.json --localize
```

Launches (or attaches to, via `--server-url`) a vLLM server, runs the same
prompt `--runs` times at each batch size, and reports whether outputs were
bitwise-identical across repeats and, if not, the first diverging token
index. `--localize` additionally loads the model via `transformers` and diffs
per-layer activations between the smallest diverging batch size and batch=1,
to find which layer nondeterminism first appears at. Also supports
`--tp-sizes` (tensor-parallel sweep) and `--seq-lengths` (padding sweep).

### 2. Validate the batch-invariant kernels

```bash
detkernels-kernel-check --kernel-set batch_invariant --batch-sizes 1,64 --runs 100
detkernels-kernel-check --kernel-set batch_variant   --batch-sizes 1,64 --runs 100
```

Runs a small random-weight decode-only transformer
(`kernels/tiny_model.py`, built entirely from this repo's own RMSNorm/matmul/
attention kernels) `--runs` times per batch size and checks whether one
tracked sequence's generated tokens stay bitwise-identical regardless of
what else is in the batch. `batch_invariant` should show 0 divergences;
`batch_variant` is the deliberately-nondeterministic counterpart kept for
comparison.

### 3. Benchmark kernel performance overhead

```bash
detkernels-kernel-bench --batch-sizes 1,8,64 --iters 50 --output bench.json
```

CUDA-event-timed comparison of each batch-invariant kernel against its
batch-variant twin and a plain PyTorch/cuBLAS baseline. See
[docs/PHASE2_RESULTS.md](./docs/PHASE2_RESULTS.md) for results and honesty
caveats (e.g. the matmul kernel is unoptimized and 6–12x slower than cuBLAS
— a known gap, not a hidden one).

### 4. Reproduce the GDN linear-attention nondeterminism (Phase 3, in progress)

```bash
detkernels-gdn-repro --mode decode --n-steps 300 --n-other-seqs 0,63 --repeats 2 \
    --output gdn_repro.json
```

Probes `flash-linear-attention`'s `chunk_gated_delta_rule` kernel (the
Triton kernel behind Qwen3-Next/3.6's Gated-Delta-Net linear attention) for
reduction-order invariance, independent of vLLM or model loading. `--mode`
selects which hypothesis to test: `row` (rectangular batch composition),
`packing` (position within a `cu_seqlens` ragged-packed batch), or `decode`
(bf16 recurrent-state drift across many simulated incremental-decode steps
— see `gdn/repro.py`'s docstring for why this is the current leading
hypothesis).

## Tests

```bash
python -m pytest tests/ -v
```

GPU-only test files skip (not fail) when torch/triton/fla/CUDA aren't
available locally, so the full suite runs on a CPU-only dev machine with the
GPU-dependent tests reported as skipped.

## Repository layout

```
harness/    Phase 1: nondeterminism measurement + divergence localization against a real vLLM server
kernels/    Phase 2: batch-invariant RMSNorm/matmul/attention (+ batch-variant twins), TinyModel, integration + benchmark
gdn/        Phase 3: reproduction harness for the Gated-Delta-Net linear-attention nondeterminism gap (vLLM #42960)
docs/       Phase-by-phase results, gap analysis, decision log
tests/      pytest suite (CPU-safe; GPU-only tests skip cleanly without a GPU)
project.md  The full phased build brief this project follows
```

## Design notes

- Every kernel module ships a batch-invariant version (fixed reduction
  strategy regardless of batch size) alongside a deliberately batch-*variant*
  twin (switches reduction strategy direct vs. split-via-atomics based on
  a size threshold) to demonstrate the actual mechanism, not just assert
  correctness of one implementation.
- Correctness is checked via **row invariance**: hold one sequence's content
  fixed, embed it in batches of different sizes/composition, and assert
  `torch.equal` (bitwise identity) on that row's output alone. This same
  technique is applied consistently from single kernels up through the
  integrated `TinyModel` and the `gdn/` reproduction experiments.
- Negative results are documented, not hidden e.g. the batch-variant
  kernels didn't reliably reproduce observable divergence at toy scale (see
  [docs/PHASE2_RESULTS.md](./docs/PHASE2_RESULTS.md)), and two of three GDN
  reproduction hypotheses came back clean before a third identified likely
  cause. See [docs/DECISIONS.md](./docs/DECISIONS.md) for the full running
  log of what was tried and why.

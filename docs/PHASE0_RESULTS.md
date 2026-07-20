# Phase 0 Results — Nondeterminism Reproduction

Satisfies Phase 0 gate items (a) and (b): a working reference stack, and a
reproduced, precisely-characterized instance of batch-size-dependent
nondeterminism. Raw data: [docs/reports/phase0_baseline.json](./reports/phase0_baseline.json).

## Setup
- Stack: vLLM 0.19.1, served via `vllm serve` (real subprocess, not embedded),
  OpenAI-compatible HTTP API.
- Hardware: Google Colab, Tesla T4 (compute capability 7.5), `dtype=float16`.
- Model: Qwen/Qwen3-1.7B.
- Decoding: greedy (`temperature=0.0, top_p=1.0`) — isolates kernel-level
  reduction-order nondeterminism from sampling randomness.
- Prompt: "Explain in one paragraph why floating point addition is not
  associative, and why that matters for reproducibility."
- `max_tokens=256`, `n_repeats=100`.

## Findings

**Baseline (batch size 1, 100 repeats): 100% bitwise identical.** As expected —
a single, unbatched request has no batch-size-dependent reduction order to vary.

**Batch size 32, 100 repeats: NOT identical.** 19 of the 99 non-reference repeats
diverged from run 0, and **every one of them diverged at exactly token index
235** — the token `"."` immediately following `"...through calculations"` in the
generated text. This is not diffuse noise scattered across many token positions;
it's one specific, repeatable decision point where accumulated floating-point
differences (driven by batch-size-dependent reduction order) flip which token
wins the greedy argmax.

The direct bs=1-run-0 vs bs=32-run-0 comparison happened to match in this
particular pair, but the within-batch-32 repeat comparisons make the underlying
claim from "Defeating Nondeterminism in LLM Inference" unambiguous: the same
prompt, same weights, same decoding config, produces different outputs purely
because of how many concurrent requests it was batched with.

## Interim result at N=20/max_tokens=64 (superseded)
An earlier pass at lower scale (`N_REPEATS=20`, `MAX_TOKENS=64`) came back
completely clean (no divergence at all) — worth noting in the writeup as a
lesson: greedy-decoding divergence often needs enough repeats and generation
length before a genuinely close logit call gets hit. A clean result at low scale
is not evidence of absence.

## Attention backend note
Not yet confirmed which attention backend vLLM selected on this T4 (compute
capability 7.5 lacks support for some newer kernels available on Ampere+). Worth
checking `vllm_server.log` in a future run before assuming this generalizes to
the A100/H100 hardware the original papers tested on — flag as a known gap in
this baseline, to revisit if results look different on the eventual H100 rental
(see [DECISIONS.md](./DECISIONS.md)).

## What this unblocks
Phase 0 gate is satisfied. Proceeding to Phase 1: build the reusable
`detkernels-audit` harness that generalizes this exact sweep (batch sizes,
TP configs, sequence lengths) at proper scale (N=100 -> 1000) with automated
divergence localization down to the layer/kernel level.

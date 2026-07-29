# Phase 3 Results — GDN Linear-Attention Reproduction Attempt

Targets [PHASE0_GAP_ANALYSIS.md](./PHASE0_GAP_ANALYSIS.md)'s decision: extend
batch-invariant determinism to GDN_ATTN (Gated-Delta-Net linear attention,
Qwen3-Next/3.6 hybrid Mamba+MoE models), tracked upstream at
[vllm-project/vllm#42960](https://github.com/vllm-project/vllm/issues/42960).
Raw script: [`gdn/repro.py`](../gdn/repro.py); CLI:
`detkernels-gdn-repro` / `python -m gdn.repro`.

**Result: a rigorous negative result.** Four independent, well-grounded
hypotheses about how `flash-linear-attention`'s `chunk_gated_delta_rule`
Triton kernel (the kernel behind GDN_ATTN) could fail to be batch-invariant
were each tested directly against the library and each came back bit-exact
identical (`max_abs_diff=0.0`). This phase did not close the gap. It's
reported honestly as a limitation, with the reasoning for why, rather than
manufactured into a false positive.

## Reframing #42960 itself

Before analyzing the negative results, a correction to how this phase
initially understood the target: issue #42960 (filed 2026-05-18, still open)
is **vLLM's `VLLM_BATCH_INVARIANT=1` hard-refusing to start** when the
GDN_ATTN backend is selected (`RuntimeError` from
`vllm/v1/attention/selector.py:154`, "not supported"). That is a
precautionary coverage gap — nobody has implemented or verified a
batch-invariant code path for GDN_ATTN — not a report of a confirmed,
reproduced numeric divergence. The unmerged fix attempt,
[PR #45819](https://github.com/vllm-project/vllm/pull/45819), and its
reviewer's ("Birol") three flagged drift sources are what a correct fix
would need to address, identified by code inspection during review — useful,
concrete engineering targets, but not each independently confirmed via a
live repro. One of the three (variable-KV-length attention reduction) turned
out to be about the hybrid model's *regular* full-attention layers, not
`chunk_gated_delta_rule` at all, and so was out of scope for a GDN-kernel-only
test from the start.

## Setup
- `flash-linear-attention` (PyPI `flash-linear-attention`, latest available
  at test time — no version is pinned by either #42960 or PR #45819, so
  there's no way to match a "known-broken" version even if one exists).
- Google Colab, CUDA GPU (A100/L4-class, sm_80+, for native bf16 tensor-core
  support — required since FLA's kernels operate in bfloat16, matching real
  Qwen3-Next/3.6 usage).
- All checks use this project's established row-invariance methodology: hold
  one sequence's content fixed, vary something about what it's batched
  alongside, and assert `torch.equal` (bitwise identity) plus max-abs-diff on
  its output alone.

## The four hypotheses tested

### 1. Rectangular batch composition (`--mode row`)
Same tracked sequence (seq_len=128), embedded in a padded rectangular batch
of size 1 vs. 64, one-shot `chunk_gated_delta_rule` call.
**Result: identical across 5 repeats, `max_abs_diff=0.0`.**

### 2. Position within a fixed-total-size packed batch (`--mode packing`)
vLLM's actual continuous-batching shape — ragged sequences concatenated via
`cu_seqlens`, not padded — with the tracked sequence moved to each of 5
positions within a batch of otherwise-fixed total size and composition.
**Result: identical across 5 repeats × 4 positions, `max_abs_diff=0.0`.**

### 3. bf16 recurrent-state drift over incremental decode (`--mode decode`)
The reviewer-identified mechanism, simulated directly: 300 sequential
single-token decode steps, `initial_state`/`final_state` explicitly cast to
bf16 between calls (mimicking a persistent KV/state cache's storage
precision), comparing the tracked sequence's trajectory alone (`n_other=0`)
vs. concurrently decoding alongside 63 other sequences (`n_other=63`).
**Result: identical across 2 repeats × 300 steps, `max_abs_diff=0.0`,
`first_diverging_step=None`.**

### 4. Total packed-call size / chunk-boundary geometry (`--mode geometry`)
Closes a gap in check #2: that test held the packed batch's *total* token
count constant (only reordering within it), so it couldn't have exercised a
total-size-dependent grid/tile selection. This holds the tracked sequence
fixed at both content and position (always index 0) and varies only how much
*other* content is packed alongside it in the same `cu_seqlens` call — 0,
500, and 2000 extra tokens — with both the tracked sequence (50 tokens) and
filler sequences (47 tokens) deliberately not divisible by the kernel's
fixed `chunk_size=64`, to land on ragged chunk boundaries rather than a
clean 2-chunk case.
**Result: identical across 3 repeats × 2 totals, `max_abs_diff=0.0`.**

## What this means
Four different, real numeric mechanisms that plausibly could have caused
batch-composition-dependent output — and were specifically called out by a
maintainer's own code review as areas of concern — were each tested and
found to produce bit-exact identical output on the currently-available
`flash-linear-attention` library. This is meaningfully different from "we
didn't find it" after one attempt; it's a structured elimination across the
hypothesis space a domain expert (the PR reviewer) themselves identified.

The most likely explanation, in order of plausibility:
1. **The mechanism only manifests inside vLLM's own integration layer**
   (kernel dispatch/config selection tied to the *scheduler's* view of total
   batch load across paged KV-cache blocks), not in FLA's public
   `chunk_gated_delta_rule` function called directly and identically
   regardless of context — i.e. we may have been testing the right kernel
   but the wrong call site.
2. The currently pip-installed FLA version happens to not exhibit whatever
   numeric issue motivated the vLLM-side precautionary block (unverifiable —
   no version is pinned anywhere in the upstream discussion to compare
   against).
3. The precautionary block in #42960 was conservative from the start, and
   `chunk_gated_delta_rule` may already be closer to batch-invariant than
   assumed — plausible given points 1–3 of the reviewer's analysis are
   concerns raised during code review, not each independently confirmed via
   a working repro on their end either, as far as this project's research
   into the PR thread found.

## Known gaps
- **No real vLLM + GDN model run attempted.** This would be the most
  faithful reproduction (exercising the actual dispatch/scheduler code path
  hypothesis above), but Qwen3-Next/3.6 has no small variant — the flagship
  is Qwen3-Next-80B-A3B (MoE), ~160GB in bf16, beyond a single Colab GPU or
  the compute budget logged in [DECISIONS.md](./DECISIONS.md). PR #45819 is
  also unmerged and flagged "needs-rebase," so reproducing the exact
  upstream state would require manually patching it onto a matching vLLM
  commit.
- **No FLA version pinning.** Neither #42960 nor PR #45819 references a
  specific `flash-linear-attention` commit or version, so there's no way to
  confirm whether this project tested the same code the bug reports are
  about.
- **JSON report artifacts from these four runs were not preserved as
  tracked files** (Colab-session-local only) — the per-run summary numbers
  above are transcribed from console output captured during the runs, not
  linked as raw data the way Phase 1/2's reports are. If revisited, rerun
  with `--output` and commit the JSON alongside this doc.
- Only 2–5 repeats per hypothesis (compute/time-bounded), not the 100+ used
  in Phase 1/2's own measurements — a very-low-probability intermittent
  effect (cf. Phase 0/1's own 1/29–19/99 divergence rates on real vLLM)
  can't be fully ruled out at this repeat count.

## What this unblocks
Phase 3's specific target (reproduce, then fix, GDN_ATTN nondeterminism at
the kernel level) is not closed. Per project.md's standing instruction ("if
a phase's gate condition cannot be met after reasonable effort, stop and
flag it explicitly... report back with options"), this was raised
explicitly rather than pushed forward on a guess. Legitimate paths forward,
not yet decided:
- Attempt the real vLLM + GDN model run despite its cost, to test the
  "wrong call site" hypothesis directly.
- Pivot to one of project.md's other fallback angles (e.g. MoE routing
  determinism, or a non-CUDA/portable backend) as Phase 3's target instead.
- Treat the harness + Phase 2 kernels + this structured negative result as
  the project's deliverable, and move toward Phase 4 (polish/writeup) with
  an honest account of an unclosed gap — itself consistent with project.md's
  explicit instruction to report negative results credibly rather than
  inflate the success narrative.

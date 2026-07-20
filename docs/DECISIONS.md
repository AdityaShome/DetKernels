# Decision Log

Running log of choices made during the project, why, and what was tried. This is raw material for the Phase 4 writeup — keep entries honest, including dead ends.

---

## 2026-07-20 — Compute path

**Decision:** Start on Google Colab (free/pro tier GPU, typically T4 16GB, sometimes A100) for Phase 0-2 work: reference stack setup, nondeterminism reproduction, harness build, and kernel correctness + split-K + warp-specialization work. Later, rent a Hopper/Blackwell GPU (H100 or similar, ~$2-3/hr on RunPod/Lambda/vast.ai) specifically to benchmark the Tensor Memory Accelerator (TMA) path and produce a direct comparison.

**Why:**
- Local hardware is an RTX 3050 laptop GPU with only 4GB VRAM — enough for batch=1 sanity checks but not for the batch-size sweeps (1 vs 32) that are the core of the reproducibility harness, and nowhere near enough for vLLM's KV-cache-heavy memory model.
- TMA is a hardware feature that only exists on Hopper (sm_90) and Blackwell (sm_100/sm_120) GPUs — it cannot be exercised on Ampere/Ada cards regardless of VRAM. Colab's typical T4/A100 allocations don't have it either, so the TMA-specific portion of the matmul optimization (the piece the LLM-42 paper attributes for closing 194->527 TFLOPS) requires separate access.
- Splitting compute into two stages (cheap/free Colab first, paid rental later) keeps early iteration cost at zero while deferring the expensive GPU-hour spend until the harness and kernels are already correctness-validated — avoids burning rental budget on debugging.

**How to apply:** Phase 0-2 deliverables (harness, RMSNorm, matmul w/ split-K + warp specialization, attention) should all be developed and correctness-tested on Colab first. Only the TMA benchmark comparison (and any final "after" numbers meant to be compared against the 527 TFLOPS ceiling) require the rented GPU. Document both sets of numbers separately in the final report — don't conflate a Colab T4/A100 result with an H100 TMA result.

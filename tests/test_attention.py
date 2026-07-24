"""GPU-only tests for kernels/attention.py. Skip cleanly (not error) when
torch is missing/non-functional, when CUDA isn't available, or when triton
isn't installed -- this repo's local dev environment has none of the three;
these tests are meant to run for real on Colab/rented GPU hardware.
"""
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
if not hasattr(torch, "zeros"):
    pytest.skip("local torch install is a non-functional stub", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)
pytest.importorskip("triton", reason="requires triton")

from kernels.attention import (  # noqa: E402
    attention_batch_invariant,
    attention_batch_variant,
    attention_reference,
)


def _random_qkv(b, h, s, d, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, h, d, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(b, h, s, d, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(b, h, s, d, device="cuda", dtype=dtype, generator=g)
    return q, k, v


def test_batch_invariant_matches_reference():
    q, k, v = _random_qkv(8, 4, 128, 64)
    got = attention_batch_invariant(q, k, v)
    want = attention_reference(q, k, v)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_variant_matches_reference():
    # Correctness (not invariance) should hold regardless of which reduction
    # path (direct vs split-KV) is taken.
    q_small, k_small, v_small = _random_qkv(1, 1, 128, 64, seed=1)
    q_large, k_large, v_large = _random_qkv(8, 4, 128, 64, seed=1)
    for q, k, v in ((q_small, k_small, v_small), (q_large, k_large, v_large)):
        got = attention_batch_variant(q, k, v)
        want = attention_reference(q, k, v)
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_invariant_row_is_identical_regardless_of_batch_size():
    """The core claim: the SAME (query, KV-cache) pair, run through the SAME
    kernel, produces a bitwise-identical result whether it's alone (batch=1)
    or embedded in a much larger batch (batch=64)."""
    s, d = 128, 64
    torch.manual_seed(0)
    q_row = torch.randn(1, 1, d, device="cuda", dtype=torch.float16)
    k_row = torch.randn(1, 1, s, d, device="cuda", dtype=torch.float16)
    v_row = torch.randn(1, 1, s, d, device="cuda", dtype=torch.float16)

    solo_out = attention_batch_invariant(q_row, k_row, v_row)

    q_batch = torch.randn(64, 1, d, device="cuda", dtype=torch.float16)
    k_batch = torch.randn(64, 1, s, d, device="cuda", dtype=torch.float16)
    v_batch = torch.randn(64, 1, s, d, device="cuda", dtype=torch.float16)
    q_batch[0] = q_row[0]
    k_batch[0] = k_row[0]
    v_batch[0] = v_row[0]
    batch_out = attention_batch_invariant(q_batch, k_batch, v_batch)

    assert torch.equal(solo_out[0], batch_out[0])


def test_batch_variant_row_can_diverge_across_batch_sizes():
    """Documents (rather than strictly asserts) the failure mode: the SAME
    (query, KV-cache) pair through the batch-variant kernel is allowed to
    differ between batch=1 (split-KV path, below split_bh_threshold=32) and
    batch=64 (direct path). We only assert both runs are internally
    well-formed (no NaNs); whether they actually differ is empirical and
    reported in docs/PHASE2_RESULTS.md rather than hard-asserted here, since
    atomic-add ordering isn't guaranteed to disagree on every hardware/seed
    combination.
    """
    s, d = 128, 64
    torch.manual_seed(0)
    q_row = torch.randn(1, 1, d, device="cuda", dtype=torch.float16)
    k_row = torch.randn(1, 1, s, d, device="cuda", dtype=torch.float16)
    v_row = torch.randn(1, 1, s, d, device="cuda", dtype=torch.float16)

    solo_out = attention_batch_variant(q_row, k_row, v_row)

    q_batch = torch.randn(64, 1, d, device="cuda", dtype=torch.float16)
    k_batch = torch.randn(64, 1, s, d, device="cuda", dtype=torch.float16)
    v_batch = torch.randn(64, 1, s, d, device="cuda", dtype=torch.float16)
    q_batch[0] = q_row[0]
    k_batch[0] = k_row[0]
    v_batch[0] = v_row[0]
    batch_out = attention_batch_variant(q_batch, k_batch, v_batch)

    assert not torch.isnan(solo_out).any()
    assert not torch.isnan(batch_out).any()

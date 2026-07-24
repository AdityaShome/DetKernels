"""GPU-only tests for kernels/matmul.py. Skip cleanly (not error) when torch
is missing/non-functional, when CUDA isn't available, or when triton isn't
installed -- this repo's local dev environment has none of the three; these
tests are meant to run for real on Colab/rented GPU hardware.
"""
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
if not hasattr(torch, "zeros"):
    pytest.skip("local torch install is a non-functional stub", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)
pytest.importorskip("triton", reason="requires triton")

from kernels.matmul import (  # noqa: E402
    matmul_batch_invariant,
    matmul_batch_variant,
    matmul_reference,
)


def _random_input(m, k, n, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=dtype, generator=g)
    b = torch.randn(k, n, device="cuda", dtype=dtype, generator=g)
    return a, b


def test_batch_invariant_matches_reference():
    a, b = _random_input(128, 256, 128)
    got = matmul_batch_invariant(a, b)
    want = matmul_reference(a, b)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_variant_matches_reference():
    # Correctness (not invariance) should hold regardless of which reduction
    # path (direct vs split-K) is taken.
    a_small, b = _random_input(8, 256, 128, seed=1)
    a_large, _ = _random_input(128, 256, 128, seed=1)
    for a in (a_small, a_large):
        got = matmul_batch_variant(a, b)
        want = matmul_reference(a, b)
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_invariant_row_is_identical_regardless_of_batch_size():
    """The core claim: the SAME row, run through the SAME kernel, produces a
    bitwise-identical result whether it's alone (M=1) or embedded in a much
    larger batch (M=128)."""
    k, n = 256, 128
    torch.manual_seed(0)
    row = torch.randn(1, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(k, n, device="cuda", dtype=torch.float16)

    solo_out = matmul_batch_invariant(row, weight)

    batch = torch.randn(128, k, device="cuda", dtype=torch.float16)
    batch[0] = row[0]
    batch_out = matmul_batch_invariant(batch, weight)

    assert torch.equal(solo_out[0], batch_out[0])


def test_batch_variant_row_can_diverge_across_batch_sizes():
    """Documents (rather than strictly asserts) the failure mode: the SAME
    row through the batch-variant kernel is allowed to differ between M=1
    (split-K path, below split_m_threshold=64) and M=128 (direct path). We
    only assert both runs are internally well-formed (no NaNs); whether they
    actually differ is empirical and reported in docs/PHASE2_RESULTS.md
    rather than hard-asserted here, since atomic-add ordering isn't
    guaranteed to disagree on every hardware/seed combination.
    """
    k, n = 256, 128
    torch.manual_seed(0)
    row = torch.randn(1, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(k, n, device="cuda", dtype=torch.float16)

    solo_out = matmul_batch_variant(row, weight)

    batch = torch.randn(128, k, device="cuda", dtype=torch.float16)
    batch[0] = row[0]
    batch_out = matmul_batch_variant(batch, weight)

    assert not torch.isnan(solo_out).any()
    assert not torch.isnan(batch_out).any()

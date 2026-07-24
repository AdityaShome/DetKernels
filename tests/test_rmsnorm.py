"""GPU-only tests for kernels/rmsnorm.py. Skip cleanly (not error) when torch
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

from kernels.rmsnorm import (  # noqa: E402
    rmsnorm_batch_invariant,
    rmsnorm_batch_variant,
    rmsnorm_reference,
)


def _random_input(n_rows, n_cols, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype, generator=g)
    weight = torch.randn(n_cols, device="cuda", dtype=dtype, generator=g)
    return x, weight


def test_batch_invariant_matches_reference():
    x, weight = _random_input(8, 4096)
    got = rmsnorm_batch_invariant(x, weight)
    want = rmsnorm_reference(x, weight)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_variant_matches_reference():
    # Correctness (not invariance) should hold regardless of which reduction
    # path is taken.
    x_small, weight = _random_input(4, 4096, seed=1)
    x_large, _ = _random_input(32, 4096, seed=1)
    for x in (x_small, x_large):
        got = rmsnorm_batch_variant(x, weight)
        want = rmsnorm_reference(x, weight)
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_batch_invariant_row_is_identical_regardless_of_batch_size():
    """The core claim: the SAME row, run through the SAME kernel, produces a
    bitwise-identical result whether it's alone (batch=1) or embedded in a
    much larger batch (batch=64)."""
    n_cols = 4096
    torch.manual_seed(0)
    row = torch.randn(1, n_cols, device="cuda", dtype=torch.float16)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float16)

    solo_out = rmsnorm_batch_invariant(row, weight)

    batch = torch.randn(64, n_cols, device="cuda", dtype=torch.float16)
    batch[0] = row[0]
    batch_out = rmsnorm_batch_invariant(batch, weight)

    assert torch.equal(solo_out[0], batch_out[0])


def test_batch_variant_row_can_diverge_across_batch_sizes():
    """Documents (rather than strictly asserts) the failure mode: the SAME
    row through the batch-variant kernel is allowed to differ between
    batch=1 (single-block path) and batch=64 (split-atomics path, above
    split_threshold=16). We only assert that both runs are internally
    well-formed (no NaNs); whether they actually differ is empirical and
    reported in docs/PHASE2_RESULTS.md rather than hard-asserted here, since
    atomic-add ordering isn't guaranteed to disagree on every hardware/seed
    combination.
    """
    n_cols = 4096
    torch.manual_seed(0)
    row = torch.randn(1, n_cols, device="cuda", dtype=torch.float16)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float16)

    solo_out = rmsnorm_batch_variant(row, weight)

    batch = torch.randn(64, n_cols, device="cuda", dtype=torch.float16)
    batch[0] = row[0]
    batch_out = rmsnorm_batch_variant(batch, weight)

    assert not torch.isnan(solo_out).any()
    assert not torch.isnan(batch_out).any()

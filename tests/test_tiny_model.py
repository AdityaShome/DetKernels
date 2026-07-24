"""GPU-only tests for kernels/tiny_model.py -- the end-to-end integration
of RMSNorm + matmul + attention into a runnable decode loop (project.md
Phase 2 task 5). Skip cleanly (not error) when torch is missing/non-
functional, when CUDA isn't available, or when triton isn't installed.
"""
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
if not hasattr(torch, "zeros"):
    pytest.skip("local torch install is a non-functional stub", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)
pytest.importorskip("triton", reason="requires triton")

from kernels.tiny_model import TinyModel, TinyModelConfig  # noqa: E402


def _small_model():
    cfg = TinyModelConfig(vocab_size=64, num_layers=2, num_heads=2, head_dim=32,
                           mlp_hidden=128, seed=0)
    return TinyModel(cfg, device="cuda")


def test_generate_produces_well_formed_output_both_kernel_sets():
    model = _small_model()
    for kernel_set in ("batch_invariant", "batch_variant"):
        tokens = torch.randint(0, model.config.vocab_size, (8,), device="cuda")
        out = model.generate(tokens, n_steps=4, kernel_set=kernel_set)
        assert out.shape == (8, 4)
        assert (out >= 0).all() and (out < model.config.vocab_size).all()


def test_batch_invariant_generation_is_identical_regardless_of_batch_size():
    """The end-to-end claim: the SAME starting token, run through the SAME
    model with the batch-invariant kernel set, produces a bitwise-identical
    generated sequence whether it's the only sequence in the batch or
    embedded in a much larger one -- across multiple layers and multiple
    autoregressive decode steps, not just a single kernel call."""
    model = _small_model()
    torch.manual_seed(0)
    start_token = torch.tensor([5], device="cuda")

    solo_out = model.generate(start_token, n_steps=5, kernel_set="batch_invariant")

    batch_tokens = torch.randint(0, model.config.vocab_size, (32,), device="cuda")
    batch_tokens[0] = start_token[0]
    batch_out = model.generate(batch_tokens, n_steps=5, kernel_set="batch_invariant")

    assert torch.equal(solo_out[0], batch_out[0])


def test_batch_variant_generation_can_diverge_across_batch_sizes():
    """Documents (rather than strictly asserts) the failure mode at the
    integrated-model level: the SAME starting token through the batch-
    variant kernel set is allowed to produce a different generated sequence
    between batch=1 and batch=32. We only assert well-formedness; whether it
    actually diverges is empirical and reported in docs/PHASE2_RESULTS.md.
    """
    model = _small_model()
    torch.manual_seed(0)
    start_token = torch.tensor([5], device="cuda")

    solo_out = model.generate(start_token, n_steps=5, kernel_set="batch_variant")

    batch_tokens = torch.randint(0, model.config.vocab_size, (32,), device="cuda")
    batch_tokens[0] = start_token[0]
    batch_out = model.generate(batch_tokens, n_steps=5, kernel_set="batch_variant")

    assert (solo_out[0] >= 0).all() and (solo_out[0] < model.config.vocab_size).all()
    assert (batch_out[0] >= 0).all() and (batch_out[0] < model.config.vocab_size).all()

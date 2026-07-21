import pytest

torch = pytest.importorskip("torch", reason="localizer tensor diffing requires torch")
if not hasattr(torch, "zeros"):
    pytest.skip("local torch install is a non-functional stub", allow_module_level=True)

from harness.localizer import diff_activations  # noqa: E402


def test_identical_activations_yield_no_divergence():
    acts1 = {"layer_0": torch.zeros(1, 4, 8), "layer_1": torch.ones(1, 4, 8)}
    actsN = {"layer_0": torch.zeros(2, 4, 8), "layer_1": torch.ones(2, 4, 8)}

    result = diff_activations(acts1, actsN, first_seq_index=0, atol=0.0)

    assert result.first_diverging_layer is None
    assert all(d.identical for d in result.layer_diffs)


def test_finds_first_diverging_layer():
    acts1 = {
        "layer_0": torch.zeros(1, 4, 8),
        "layer_1": torch.zeros(1, 4, 8),
        "layer_2": torch.zeros(1, 4, 8),
    }
    actsN = {
        "layer_0": torch.zeros(2, 4, 8),
        "layer_1": torch.full((2, 4, 8), 1e-3),  # diverges here first
        "layer_2": torch.full((2, 4, 8), 1.0),   # also diverges, but not first
    }

    result = diff_activations(acts1, actsN, first_seq_index=0, atol=0.0)

    assert result.first_diverging_layer == 1
    assert result.layer_diffs[0].identical is True
    assert result.layer_diffs[1].identical is False
    assert result.layer_diffs[2].identical is False


def test_atol_tolerance_absorbs_small_diffs():
    acts1 = {"layer_0": torch.zeros(1, 4, 8)}
    actsN = {"layer_0": torch.full((2, 4, 8), 1e-8)}

    result = diff_activations(acts1, actsN, first_seq_index=0, atol=1e-6)

    assert result.first_diverging_layer is None


def test_uses_the_requested_sequence_index_from_the_batch():
    acts1 = {"layer_0": torch.full((1, 4, 8), 5.0)}
    actsN = {
        "layer_0": torch.stack(
            [torch.full((4, 8), 1.0), torch.full((4, 8), 5.0), torch.full((4, 8), 9.0)]
        )
    }

    result = diff_activations(acts1, actsN, first_seq_index=1, atol=0.0)

    assert result.first_diverging_layer is None

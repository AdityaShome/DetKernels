"""GPU-only smoke test for gdn/repro.py -- confirms the reproduction script
runs end-to-end and produces well-formed output against the real FLA
library. Does NOT assert a specific divergence outcome: whether
chunk_gated_delta_rule actually diverges across batch sizes is the open
empirical question this script exists to answer (see vLLM #42960 / PR
#45819), not something to hard-code into a test. Skips cleanly (not error)
when torch/triton/fla are missing/non-functional or CUDA isn't available.
"""
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
if not hasattr(torch, "zeros"):
    pytest.skip("local torch install is a non-functional stub", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)
pytest.importorskip("triton", reason="requires triton")
pytest.importorskip("fla", reason="requires flash-linear-attention")

from gdn.repro import check_packing_position_invariance, check_row_invariance  # noqa: E402


def test_check_row_invariance_produces_well_formed_comparisons():
    result = check_row_invariance(batch_sizes=[1, 8], seq_len=32, h=2, hv=4,
                                   k_dim=32, v_dim=32, seed=0)
    assert result["reference_batch_size"] == 1
    assert len(result["comparisons"]) == 1
    c = result["comparisons"][0]
    assert c["batch_size"] == 8
    assert isinstance(c["identical"], bool)
    assert c["max_abs_diff"] >= 0.0
    import math
    assert not math.isnan(c["max_abs_diff"])


def test_check_packing_position_invariance_produces_well_formed_comparisons():
    result = check_packing_position_invariance(
        tracked_seq_len=16, other_seq_lens=(12, 20), h=2, hv=4, k_dim=32, v_dim=32, seed=0,
    )
    assert result["reference_position"] == 0
    assert len(result["comparisons"]) == 2
    import math
    for c in result["comparisons"]:
        assert isinstance(c["identical"], bool)
        assert c["max_abs_diff"] >= 0.0
        assert not math.isnan(c["max_abs_diff"])

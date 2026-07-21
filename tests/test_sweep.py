from harness.sweep import pad_prompt_to_length, summarize_batch_results


def test_all_identical_runs_flagged_as_deterministic():
    runs = [
        ("a", "b", "c"),
        ("a", "b", "c"),
        ("a", "b", "c"),
    ]
    result = summarize_batch_results(batch_size=1, runs=runs)
    assert result.batch_size == 1
    assert result.all_identical is True
    assert result.n_diverging == 0
    assert result.divergence_indices == []
    assert result.reference_tokens == ("a", "b", "c")


def test_diverging_runs_flagged_as_nondeterministic():
    runs = [
        ("a", "b", "c", "d"),  # reference (run 0)
        ("a", "b", "c", "d"),  # identical to reference
        ("a", "b", "X", "d"),  # diverges at index 2
        ("a", "Y", "c", "d"),  # diverges at index 1
    ]
    result = summarize_batch_results(batch_size=32, runs=runs)
    assert result.all_identical is False
    assert result.n_diverging == 2
    assert sorted(result.divergence_indices) == [1, 2]


def test_single_run_is_trivially_identical():
    result = summarize_batch_results(batch_size=1, runs=[("only", "one", "run")])
    assert result.all_identical is True
    assert result.n_diverging == 0


def test_empty_runs_raises():
    import pytest

    with pytest.raises(ValueError):
        summarize_batch_results(batch_size=1, runs=[])


def test_matches_phase0_observed_pattern():
    # Regression guard modeled on the actual Phase 0 finding
    # (docs/PHASE0_RESULTS.md): 81 identical repeats, 19 diverging, all at the
    # same token index.
    reference = tuple(str(i) for i in range(300))
    diverging = reference[:235] + ("DIFFERENT",) + reference[236:]

    runs = [reference] * 81 + [diverging] * 19
    result = summarize_batch_results(batch_size=32, runs=runs)

    assert result.all_identical is False
    assert result.n_diverging == 19
    assert set(result.divergence_indices) == {235}


def test_pad_prompt_to_length_reaches_target_word_count():
    padded = pad_prompt_to_length("short prompt", target_words=20)
    assert len(padded.split()) >= 20
    assert padded.startswith("short prompt")


def test_pad_prompt_to_length_is_noop_if_already_long_enough():
    prompt = " ".join(["word"] * 30)
    assert pad_prompt_to_length(prompt, target_words=10) == prompt

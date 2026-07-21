from harness.diff import first_divergence


def test_identical_sequences_return_none():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_identical_empty_sequences_return_none():
    assert first_divergence([], []) is None


def test_divergence_at_first_element():
    assert first_divergence([1, 2, 3], [9, 2, 3]) == 0


def test_divergence_in_the_middle():
    assert first_divergence(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 2


def test_divergence_at_last_element():
    assert first_divergence([1, 2, 3], [1, 2, 9]) == 2


def test_shorter_sequence_diverges_at_its_own_length():
    assert first_divergence([1, 2, 3, 4], [1, 2, 3]) == 3
    assert first_divergence([1, 2, 3], [1, 2, 3, 4]) == 3


def test_works_with_token_strings_not_just_ints():
    assert first_divergence((" the", " cat", " sat"), (" the", " cat", " sat")) is None
    assert first_divergence((" the", " cat", " sat"), (" the", " dog", " sat")) == 1

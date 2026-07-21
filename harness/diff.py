"""Pure sequence-diffing utilities, deliberately dependency-free so they can be
unit tested without a model, a GPU, or a network connection.
"""
from __future__ import annotations

from typing import Optional, Sequence


def first_divergence(seq_a: Sequence, seq_b: Sequence) -> Optional[int]:
    """Return the index of the first element where `seq_a` and `seq_b` differ,
    or `None` if they're identical. If one is a prefix of the other, the
    divergence index is the length of the shorter sequence."""
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return i
    if len(seq_a) != len(seq_b):
        return min(len(seq_a), len(seq_b))
    return None

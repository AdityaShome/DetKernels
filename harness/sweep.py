"""Batch-size (and, later, TP / sequence-length) sweep logic: fire concurrent
requests at a running OpenAI-compatible server, check for bitwise-identical
outputs across repeats, and localize the first token index of divergence.

The pure aggregation logic (`summarize_batch_results`) has no network or model
dependency and is unit tested directly in tests/test_sweep.py. The async
fetch/orchestration functions require the `openai` package and a live server,
and are exercised end-to-end via the CLI against a real (e.g. Colab) GPU.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .diff import first_divergence


@dataclass
class SweepConfig:
    model: str
    base_url: str
    prompt: str
    batch_sizes: List[int]
    n_repeats: int = 100
    max_tokens: int = 256
    temperature: float = 0.0
    # Sequence-length sweep: each value is padded/repeated onto `prompt` to
    # reach roughly that many words, then swept at `seqlen_batch_size` (the
    # dimension most likely to actually surface padding-related divergence).
    sequence_lengths: List[int] = field(default_factory=list)
    seqlen_batch_size: int = 32


@dataclass
class BatchSizeResult:
    batch_size: int
    all_identical: bool
    n_diverging: int
    divergence_indices: List[int]
    reference_tokens: Tuple


@dataclass
class SequenceLengthResult:
    sequence_length: int
    batch_size: int
    all_identical: bool
    n_diverging: int
    divergence_indices: List[int]


@dataclass
class SweepResults:
    tensor_parallel_size: int
    batch_size_results: List[BatchSizeResult] = field(default_factory=list)
    sequence_length_results: List[SequenceLengthResult] = field(default_factory=list)


def pad_prompt_to_length(prompt: str, target_words: int) -> str:
    """Pad `prompt` with filler text until it reaches roughly `target_words`
    words, to probe sequence-length/padding-dependent divergence. No-op if
    `prompt` is already at or past the target length."""
    filler = "The quick brown fox jumps over the lazy dog. "
    padded = prompt
    while len(padded.split()) < target_words:
        padded += " " + filler
    return padded


def summarize_batch_results(batch_size: int, runs: Sequence[Sequence]) -> BatchSizeResult:
    """Given `runs` (one token sequence per repeat, all for the same
    `batch_size`), compare every repeat against the first and report whether
    they're all identical plus where each divergence first occurs.

    Pure function — no I/O, no model. This is the piece the harness's own
    correctness depends on, so it's the primary target of the self-tests.
    """
    if not runs:
        raise ValueError("summarize_batch_results requires at least one run")

    reference = runs[0]
    divergence_indices = []
    for run in runs[1:]:
        d = first_divergence(reference, run)
        if d is not None:
            divergence_indices.append(d)

    return BatchSizeResult(
        batch_size=batch_size,
        all_identical=(len(divergence_indices) == 0),
        n_diverging=len(divergence_indices),
        divergence_indices=divergence_indices,
        reference_tokens=tuple(reference),
    )


async def _single_request(client, cfg: SweepConfig, prompt: str) -> Tuple:
    resp = await client.completions.create(
        model=cfg.model,
        prompt=prompt,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        logprobs=0,  # ask the server to return per-token strings we can diff
    )
    choice = resp.choices[0]
    return tuple(choice.logprobs.tokens) if choice.logprobs else (choice.text,)


async def _run_batch(client, cfg: SweepConfig, batch_size: int) -> Tuple:
    """Fire `batch_size` requests concurrently so the server's scheduler
    batches them together; return the token sequence of the FIRST completed
    request. Concurrency (not a request parameter) is what changes the
    server's internal reduction order."""
    tasks = [_single_request(client, cfg, cfg.prompt) for _ in range(batch_size)]
    results = await asyncio.gather(*tasks)
    return results[0]


async def sweep_batch_size(client, cfg: SweepConfig, batch_size: int) -> BatchSizeResult:
    # Repeats run sequentially (not gathered) so each repeat is genuinely
    # served at the configured batch size, with nothing else in flight that
    # could change the effective batch composition.
    runs = []
    for _ in range(cfg.n_repeats):
        runs.append(await _run_batch(client, cfg, batch_size))
    return summarize_batch_results(batch_size, runs)


async def sweep_sequence_length(client, cfg: SweepConfig, seq_len: int) -> SequenceLengthResult:
    padded_prompt = pad_prompt_to_length(cfg.prompt, seq_len)
    runs = []
    for _ in range(cfg.n_repeats):
        tasks = [
            _single_request(client, cfg, padded_prompt)
            for _ in range(cfg.seqlen_batch_size)
        ]
        results = await asyncio.gather(*tasks)
        runs.append(results[0])
    summary = summarize_batch_results(cfg.seqlen_batch_size, runs)
    return SequenceLengthResult(
        sequence_length=seq_len,
        batch_size=cfg.seqlen_batch_size,
        all_identical=summary.all_identical,
        n_diverging=summary.n_diverging,
        divergence_indices=summary.divergence_indices,
    )


async def run_sweep(cfg: SweepConfig, tensor_parallel_size: int = 1) -> SweepResults:
    from openai import AsyncOpenAI  # lazy import: keep this module importable

    client = AsyncOpenAI(base_url=cfg.base_url, api_key="EMPTY")
    results = SweepResults(tensor_parallel_size=tensor_parallel_size)
    for bs in cfg.batch_sizes:
        results.batch_size_results.append(await sweep_batch_size(client, cfg, bs))
    for seq_len in cfg.sequence_lengths:
        results.sequence_length_results.append(
            await sweep_sequence_length(client, cfg, seq_len)
        )
    return results

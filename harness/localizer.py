"""Divergence localizer: find which decoder layer's intermediate activations
first diverge between a batch-size-1 forward pass and a batch-size-N forward
pass of the same prompt, run through the same model weights.

This intentionally bypasses vLLM (whose paged/scheduled execution isn't easy to
hook into) and loads the model directly via `transformers`, registering a
forward hook on every decoder layer. Batch-size-dependent reduction-order
nondeterminism is a GPU/CUDA-kernel phenomenon (see "Defeating Nondeterminism
in LLM Inference" and docs/PHASE0_RESULTS.md) — running this on CPU will not
reproduce real divergence; `diff_activations` exists to validate the harness's
own comparison logic without a GPU, and `localize_divergence` becomes a real
signal only when run on an actual GPU (Colab / rented hardware).

`torch`/`transformers` are imported lazily inside functions so this module
stays importable (and its pure logic testable) in environments without those
packages installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerDiff:
    layer_index: int
    layer_name: str
    max_abs_diff: float
    identical: bool


@dataclass
class LocalizationResult:
    first_diverging_layer: Optional[int]
    layer_diffs: List[LayerDiff] = field(default_factory=list)


def diff_activations(batch1_acts: Dict, batchN_acts: Dict, first_seq_index: int = 0,
                      atol: float = 0.0) -> LocalizationResult:
    """Compare per-layer activations captured for a lone batch-of-1 request
    against the FIRST sequence's activations from a batch-of-N request.
    Returns the first layer (in insertion order of `batch1_acts`) whose max
    absolute difference exceeds `atol` (0.0 = exact/bitwise comparison).

    `batch1_acts` / `batchN_acts` map layer_name -> tensor, and must use
    consistent layer names across the two dicts.
    """
    diffs: List[LayerDiff] = []
    first_diverging = None

    for idx, (name, t1) in enumerate(batch1_acts.items()):
        tN = batchN_acts[name][first_seq_index : first_seq_index + 1]
        max_abs_diff = (t1 - tN).abs().max().item()
        identical = max_abs_diff <= atol
        diffs.append(LayerDiff(idx, name, max_abs_diff, identical))
        if not identical and first_diverging is None:
            first_diverging = idx

    return LocalizationResult(first_diverging_layer=first_diverging, layer_diffs=diffs)


class ActivationCapture:
    """Registers forward hooks on every decoder layer of a HF causal LM and
    records each layer's output hidden state, keyed by layer name."""

    def __init__(self, model):
        self.model = model
        self.activations: Dict[str, "torch.Tensor"] = {}
        self._handles = []

    def _hook(self, name):
        def fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[name] = hidden.detach()
        return fn

    def __enter__(self) -> "ActivationCapture":
        for i, layer in enumerate(self._find_decoder_layers()):
            name = f"layer_{i}"
            self._handles.append(layer.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc_info) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _find_decoder_layers(self):
        # Works for the common `model.model.layers` layout used by the
        # Llama/Qwen/Mistral-family architectures in transformers.
        base = getattr(self.model, "model", self.model)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise AttributeError(
                "Could not find decoder layers at model.model.layers; "
                "this model architecture isn't supported yet."
            )
        return layers


def localize_divergence(model, tokenizer, prompt: str, batch_size: int,
                         device: str = "cuda", atol: float = 0.0) -> LocalizationResult:
    """Run one batch-of-1 forward pass and one batch-of-`batch_size` forward
    pass (same prompt repeated) and diff every decoder layer's output."""
    import torch

    model.eval()
    inputs1 = tokenizer([prompt], return_tensors="pt").to(device)
    inputsN = tokenizer([prompt] * batch_size, return_tensors="pt").to(device)

    with torch.no_grad():
        with ActivationCapture(model) as cap1:
            model(**inputs1)
            acts1 = dict(cap1.activations)

        with ActivationCapture(model) as capN:
            model(**inputsN)
            actsN = dict(capN.activations)

    return diff_activations(acts1, actsN, first_seq_index=0, atol=atol)

"""A minimal, from-scratch decode-loop transformer used to exercise the
RMSNorm / matmul / attention kernels in kernels/ as an actual end-to-end
inference path, per project.md Phase 2 task 5: "Integrate the three kernels
into a runnable end-to-end inference path ... as part of your own minimal
inference loop if [vLLM/SGLang] integration proves too complex."

This is deliberately NOT a real model: weights are randomly initialized
(fixed seed, so a given TinyModel instance's weights are stable across
calls), there's no tokenizer, and it only implements the decode phase (one
query token per step attending over a KV cache built up so far) -- there is
no separate prefill path. That's an intentional, honest scope match to
kernels/attention.py, which only implements single-query decode-step
attention. What this exercises for real: RMSNorm -> QKV projection -> multi-
layer attention-with-growing-KV-cache -> MLP -> LM head, wired together
exactly the way a real decoder does it, using our own kernels for every
RMSNorm/matmul/attention call -- not just each kernel in isolation.

`kernel_set="batch_invariant"` selects the batch-invariant kernel from each
of kernels/{rmsnorm,matmul,attention}.py for every call in the model.
`kernel_set="batch_variant"` selects the deliberately batch-variant one.
Everything else (weights, elementwise ops like SiLU, shapes) is identical
between the two, so any output difference between kernel sets at the same
batch size is attributable to the kernels, not to model differences.

`torch` is imported lazily; this module requires a CUDA GPU + triton to run
for real (see tests/test_tiny_model.py and kernels/integration.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from kernels.rmsnorm import rmsnorm_batch_invariant, rmsnorm_batch_variant
from kernels.matmul import matmul_batch_invariant, matmul_batch_variant
from kernels.attention import attention_batch_invariant, attention_batch_variant

_KERNEL_SETS = {
    "batch_invariant": {
        "rmsnorm": rmsnorm_batch_invariant,
        "matmul": matmul_batch_invariant,
        "attention": attention_batch_invariant,
    },
    "batch_variant": {
        "rmsnorm": rmsnorm_batch_variant,
        "matmul": matmul_batch_variant,
        "attention": attention_batch_variant,
    },
}


@dataclass
class TinyModelConfig:
    vocab_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 32
    mlp_hidden: int = 256
    eps: float = 1e-6
    seed: int = 0

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim


class TinyModel:
    """Random-weight decode-only transformer. Not a real trained model --
    only useful for testing reproducibility properties of the kernels it's
    built from."""

    def __init__(self, config: TinyModelConfig, device: str = "cuda"):
        import torch

        self.config = config
        self.device = device
        self.dtype = torch.float16

        h = config.hidden_size
        g = torch.Generator(device=device).manual_seed(config.seed)

        def randn(*shape):
            return 0.02 * torch.randn(*shape, device=device, dtype=self.dtype, generator=g)

        self.embed = randn(config.vocab_size, h)
        self.layers = []
        for _ in range(config.num_layers):
            self.layers.append({
                "norm1_w": randn(h) + 1.0,
                "w_q": randn(h, h),
                "w_k": randn(h, h),
                "w_v": randn(h, h),
                "w_o": randn(h, h),
                "norm2_w": randn(h) + 1.0,
                "w_mlp1": randn(h, config.mlp_hidden),
                "w_mlp2": randn(config.mlp_hidden, h),
            })
        self.norm_f_w = randn(h) + 1.0
        self.w_lm_head = randn(h, config.vocab_size)

    def forward_step(self, token_ids, kv_cache, kernel_set: str):
        """One decode step: token_ids (B,) int64 -> next-token logits (B,
        vocab_size), and the updated per-layer KV cache."""
        import torch

        fns = _KERNEL_SETS[kernel_set]
        rmsnorm, matmul, attention = fns["rmsnorm"], fns["matmul"], fns["attention"]
        cfg = self.config
        B = token_ids.shape[0]
        H, D = cfg.num_heads, cfg.head_dim

        x = self.embed[token_ids]  # (B, hidden)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            residual = x
            xn = rmsnorm(x, layer["norm1_w"], cfg.eps)
            q = matmul(xn, layer["w_q"]).view(B, H, D)
            k_new = matmul(xn, layer["w_k"]).view(B, H, 1, D)
            v_new = matmul(xn, layer["w_v"]).view(B, H, 1, D)

            prev = kv_cache[i]
            if prev is None:
                k_cache, v_cache = k_new, v_new
            else:
                k_cache = torch.cat([prev["k"], k_new], dim=2)
                v_cache = torch.cat([prev["v"], v_new], dim=2)
            new_kv_cache.append({"k": k_cache, "v": v_cache})

            attn_out = attention(q, k_cache, v_cache).reshape(B, H * D)
            attn_out = matmul(attn_out, layer["w_o"])
            x = residual + attn_out

            residual2 = x
            xn2 = rmsnorm(x, layer["norm2_w"], cfg.eps)
            hmid = matmul(xn2, layer["w_mlp1"])
            hmid = torch.nn.functional.silu(hmid)
            hmid = matmul(hmid, layer["w_mlp2"])
            x = residual2 + hmid

        xn_f = rmsnorm(x, self.norm_f_w, cfg.eps)
        logits = matmul(xn_f, self.w_lm_head)
        return logits, new_kv_cache

    def generate(self, prompt_token_ids, n_steps: int, kernel_set: str):
        """prompt_token_ids: (B,) int64, a single starting token per
        sequence. Returns (B, n_steps) generated token ids (greedy/argmax --
        deterministic decoding, matching the harness's temperature=0.0
        convention, so any run-to-run difference is attributable to the
        kernels, not sampling)."""
        import torch

        kv_cache = [None] * self.config.num_layers
        current = prompt_token_ids
        generated = []
        for _ in range(n_steps):
            logits, kv_cache = self.forward_step(current, kv_cache, kernel_set)
            current = logits.argmax(dim=-1)
            generated.append(current)
        return torch.stack(generated, dim=1)

"""Re-run the Phase 1 harness's reproducibility measurement against the
Phase 2 integrated kernel set (kernels/tiny_model.py), instead of against
vLLM. This is project.md's Phase 2 task 5 "after" measurement: N repeats per
batch size, checked for bitwise output identity, using the exact same
aggregation logic (harness.sweep.summarize_batch_results) as the Phase 0/1
"before" measurement against real vLLM -- so the two are methodologically
comparable, even though the model under test differs (real Qwen3-1.7B via
vLLM vs. our tiny random-weight model via our own kernels; see
kernels/tiny_model.py's docstring for that scope caveat).

For each repeat, one tracked sequence (row 0, fixed starting token) is
compared across repeats; every other row in the batch gets a fresh random
starting token each repeat, so the batch *composition* varies run to run
while the thing being measured stays fixed -- mirroring how the real harness
tracks one request's output while concurrent load around it varies.

Usage:
    detkernels-kernel-check --kernel-set batch_invariant \
        --batch-sizes 1,64 --runs 100 --steps 8 --output report.json
    detkernels-kernel-check --kernel-set batch_variant \
        --batch-sizes 1,64 --runs 100 --steps 8 --output report_variant.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from harness.sweep import summarize_batch_results
from kernels.tiny_model import TinyModel, TinyModelConfig


def run_repeats(model, kernel_set: str, batch_size: int, n_steps: int, n_repeats: int,
                 tracked_token: int = 0, seed_base: int = 1000):
    import torch

    runs = []
    for i in range(n_repeats):
        g = torch.Generator(device=model.device).manual_seed(seed_base + i)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size,),
                                device=model.device, generator=g)
        tokens[0] = tracked_token
        out = model.generate(tokens, n_steps=n_steps, kernel_set=kernel_set)
        runs.append(tuple(out[0].tolist()))
    return summarize_batch_results(batch_size, runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-set", choices=["batch_invariant", "batch_variant"],
                         default="batch_invariant")
    parser.add_argument("--batch-sizes", default="1,64",
                         help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="Model weight init seed.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    cfg = TinyModelConfig(
        vocab_size=args.vocab_size, num_layers=args.num_layers, num_heads=args.num_heads,
        head_dim=args.head_dim, mlp_hidden=args.mlp_hidden, seed=args.seed,
    )
    model = TinyModel(cfg, device="cuda")

    batch_size_results = []
    for bs in batch_sizes:
        r = run_repeats(model, args.kernel_set, bs, args.steps, args.runs)
        print(f"batch_size={bs}: all_identical={r.all_identical} "
              f"n_diverging={r.n_diverging}/{args.runs - 1} "
              f"divergence_indices={r.divergence_indices}")
        batch_size_results.append(r)

    report = {
        "kernel_set": args.kernel_set,
        "model_config": {
            "vocab_size": cfg.vocab_size, "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads, "head_dim": cfg.head_dim,
            "mlp_hidden": cfg.mlp_hidden, "seed": cfg.seed,
        },
        "n_repeats": args.runs,
        "n_steps": args.steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_size_results": [
            {
                "batch_size": r.batch_size,
                "all_identical": r.all_identical,
                "n_diverging": r.n_diverging,
                "divergence_indices": r.divergence_indices,
            }
            for r in batch_size_results
        ],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

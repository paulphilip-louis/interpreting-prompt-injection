"""
Compute and save the shared injection-following direction for use as a runtime defense.

Extracts the shared steering direction across all injected tasks at a given layer,
using the diff-of-means (combine − naive) decomposition from experiment 6.
The saved vector can then be applied at inference time with a negative coefficient
to suppress injection-following without recomputing residuals.

Saved file contains:
  shared_vec  — unit direction (mean of normalised per-task vectors)
  full_vec    — mean of raw per-task diff-of-means (shared_vec * ||full||)
  layer       — layer the vector was extracted from
  metadata    — task, injections, n_train, seed, model

Usage
-----
    python experiments/compute_defense_vector.py \\
        --task sentiment --layer 21 --n-train 50 --seed 0 \\
        --out results/defense_vector.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer_lens import HookedTransformer

import src.data.opi as opi
from src.utils.steering import cache_resid, decompose_steering_vecs
from src.utils.variables import DEVICE, MODEL_NAME
from experiments.exp6_decomposition import make_split, build_task_residuals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default=MODEL_NAME)
    p.add_argument("--task",       default="sentiment", choices=list(opi.FORMAT.keys()))
    p.add_argument("--injections", nargs="+", default=opi.INJECTIONS)
    p.add_argument("--layer",      type=int, default=21)
    p.add_argument("--n-train",    type=int, default=100)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out",        default="results/defense_vector.pt")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    # ── Data ───────────────────────────────────────────────────────────────
    print(f"Loading OPI (task={args.task}) …")
    prompts_all = opi.load_opi_per_task(model, args.task)

    n_available = min(
        min(len(prompts_all[inj]["prompts"]["combine"]) for inj in args.injections),
        min(len(prompts_all[inj]["prompts"]["naive"])   for inj in args.injections),
    )
    train_idx, _ = make_split(n_available, args.n_train,
                               n_test=0, seed=args.seed)
    print(f"Train split: {len(train_idx)} / {n_available} examples (seed={args.seed})")

    # ── Cache residuals ────────────────────────────────────────────────────
    task_residuals = build_task_residuals(
        model, prompts_all, args.injections,
        layers=[args.layer], batch=4, train_idx=train_idx,
    )

    # ── Decompose ──────────────────────────────────────────────────────────
    decomp = decompose_steering_vecs(task_residuals, args.injections, args.layer)

    full_vec = torch.stack(
        [decomp["task_vecs"][t] for t in args.injections]
    ).mean(0)

    print(f"\nDecomposition at L={args.layer}:")
    print(f"  ||full_vec||  = {full_vec.norm().item():.3f}")
    print(f"  ||shared||    = {decomp['shared'].norm().item():.3f}  (unit vector)")
    for inj in args.injections:
        proj  = decomp["projections"][inj]
        ts_n  = decomp["task_specific"][inj].norm().item()
        frac  = abs(proj) / decomp["task_vecs"][inj].norm().item()
        print(f"  {inj:>10}: proj={proj:+.3f}  ||task_specific||={ts_n:.3f}  shared_frac={frac:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    payload = {
        "shared_vec": decomp["shared"].cpu(),   # unit vector
        "full_vec":   full_vec.cpu(),            # scaled: shared * ||full||
        "layer":      args.layer,
        "model":      args.model,
        "task":       args.task,
        "injections": args.injections,
        "n_train":    args.n_train,
        "seed":       args.seed,
    }
    torch.save(payload, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

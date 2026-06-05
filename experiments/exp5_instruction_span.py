"""
Experiment 5 — Instruction-span steering.

Tests whether the last-token diff-of-means vector, applied at instruction-token
positions with a *negative* coefficient, reduces ASR on hard-trigger (combine) prompts.

Background
----------
Instruction tokens precede the user turn and therefore cannot attend to injected content
(causal mask), so their residual streams are identical across injection conditions and
no diff-of-means can be extracted there.  Instead we take the combine−naive steering
vector computed at the last token and apply it at instruction positions, hypothesising
that altering the key/value representations written by those tokens changes how later
(injection-aware) positions process the injected content.

Conditions compared
-------------------
  instruction-span  — hook applied to instruction token positions [tok_start, tok_end)
  last-token        — standard hook applied to the final position (reference)

Both are swept over the same coefficient range on the same "combine" target prompts.
Negative coefficients push combine prompts toward the naive (no-trigger) direction.

The steering vector is computed on a held-out train split; evaluation uses a
non-overlapping test split to avoid overfitting the direction to the eval examples.

Usage
-----
    python experiments/exp5_instruction_span.py \\
        --task sentiment --injection spam \\
        --layer 21 \\
        --coefs -4 -3 -2 -1 0 1 2 3 4 \\
        --n-train 50 --n-test 50 --seed 0

    # Auto-pick layer by largest ||v||, run all injected tasks:
    python experiments/exp5_instruction_span.py --task sentiment --all-injections
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

import src.data.opi as opi
from src.utils.steering import (
    cache_resid,
    compute_metrics,
    diff_of_means,
    get_instruction_span,
    make_steering_hook,
    save_results,
    steer_instruction_span,
)
from src.utils.utils import to_first_token_ids
from src.utils.variables import DEVICE, MODEL_NAME


def get_instruction_text(ds, task_type: str) -> str:
    """Return the instruction string exactly as it appears in formatted prompts."""
    row = next(r for r in ds if r["task_type"] == task_type)
    old, new = opi.FORMAT[task_type]
    return row["instruction"].replace(old, new)


def make_split(n_available: int, n_train: int, n_test: int, seed: int):
    """
    Return (train_idx, test_idx) — non-overlapping random index lists.
    Raises ValueError if the pool is too small.
    """
    if n_train + n_test > n_available:
        raise ValueError(
            f"n_train ({n_train}) + n_test ({n_test}) = {n_train + n_test} "
            f"exceeds available examples ({n_available}). "
            f"Reduce --n-train or --n-test."
        )
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_available, generator=generator).tolist()
    return perm[:n_train], perm[n_train:n_train + n_test]


def run_one(model, prompts, inj_ids, cor_ids, instruction_text, layers, coefs, args):
    """
    Core experiment for one (task, injection) pair.
    Steering vector is computed on train_idx; evaluation runs on test_idx.
    Returns (span_results, ref_results, best_layer, baselines).
    """
    n_available = min(len(prompts["combine"]), len(prompts["naive"]))
    train_idx, test_idx = make_split(n_available, args.n_train, args.n_test, args.seed)

    train_combine = [prompts["combine"][i] for i in train_idx]
    train_naive   = [prompts["naive"][i]   for i in train_idx]
    test_combine  = [prompts["combine"][i] for i in test_idx]

    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test "
          f"(seed={args.seed}, n_available={n_available})")

    # ── Baselines (test set only, so they're comparable to steered results) ──
    print("  Baselines (test set):")
    baselines = {}
    for cond in ("safe", "naive", "combine"):
        plist = [prompts[cond][i] for i in test_idx]
        logits, _ = cache_resid(model, plist, batch_size=args.batch)
        baselines[cond] = compute_metrics(logits, cor_ids, inj_ids)
        m = baselines[cond]
        print(f"    [{cond:7}] ASR={m['asr']:.3f}  LD={m['mean_logit_diff']:+.3f}")

    # ── Cache residuals (train set only) ──────────────────────────────────
    print(f"  Caching residuals at {len(layers)} layer(s) on {len(train_idx)} train prompts …")
    _, res_combine = cache_resid(model, train_combine,
                                  batch_size=args.batch, cache_layers=layers)
    _, res_naive   = cache_resid(model, train_naive,
                                  batch_size=args.batch, cache_layers=layers)
    steering_vecs = diff_of_means(res_combine, res_naive, DEVICE)

    # ── Pick best layer ────────────────────────────────────────────────────
    if args.layer is not None:
        best_layer = args.layer
    else:
        best_layer = max(layers, key=lambda l: steering_vecs[l].norm().item())
        print(f"  Best layer by ||v||: L={best_layer} "
              f"(norm={steering_vecs[best_layer].norm().item():.2f})")

    vec    = steering_vecs[best_layer]
    target = test_combine  # held-out test set

    # Print instruction span for transparency
    tok_start, tok_end = get_instruction_span(model, target[0], instruction_text)
    n_toks = tok_end - tok_start
    print(f"  Instruction span: tokens [{tok_start}, {tok_end}) — {n_toks} tokens")

    # ── Instruction-span sweep ────────────────────────────────────────────
    print(f"  [Exp 5] Instruction-span steering at L={best_layer} …")
    span_results = steer_instruction_span(
        model, target, vec, best_layer, coefs,
        cor_ids, inj_ids, instruction_text,
        batch_size=args.batch,
    )

    # ── Last-token sweep (reference) ───────────────────────────────────────
    print(f"  [Ref]  Last-token steering at L={best_layer} …")
    ref_results = {"asr": np.zeros(len(coefs)), "ld": np.zeros(len(coefs))}
    for j, c in enumerate(tqdm(coefs, desc="  last-token")):
        hook = make_steering_hook(vec, c)
        logits, _ = cache_resid(
            model, target, batch_size=args.batch,
            fwd_hooks=[(f"blocks.{best_layer}.hook_resid_post", hook)],
        )
        m = compute_metrics(logits, cor_ids, inj_ids)
        ref_results["asr"][j] = m["asr"]
        ref_results["ld"][j]  = m["mean_logit_diff"]

    return span_results, ref_results, best_layer, baselines


def print_table(coefs, span, ref):
    print(f"\n  {'coef':>6}  {'span_asr':>9}  {'last_asr':>9}  {'span_ld':>9}  {'last_ld':>9}")
    for j, c in enumerate(coefs):
        print(f"  {c:>6.1f}  {span['asr'][j]:>9.3f}  {ref['asr'][j]:>9.3f}"
              f"  {span['ld'][j]:>9.3f}  {ref['ld'][j]:>9.3f}")


def plot_results(coefs, span, ref, baselines, task, injection, best_layer, model_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, ylabel in zip(axes, ("asr", "ld"), ("ASR", "Logit diff")):
        ax.plot(coefs, span[key], marker="o", label="instruction span")
        ax.plot(coefs, ref[key],  marker="s", ls="--", label="last token")
        ax.axvline(0, color="gray", lw=0.8, ls=":", label="coef=0")
        ax.axhline(baselines["combine"][{"asr": "asr", "ld": "mean_logit_diff"}[key]],
                   color="red", lw=0.8, ls=":", label="combine baseline")
        ax.axhline(baselines["naive"][{"asr": "asr", "ld": "mean_logit_diff"}[key]],
                   color="green", lw=0.8, ls=":", label="naive baseline")
        ax.set(xlabel="coefficient", ylabel=ylabel,
               title=f"{ylabel} — {task}/{injection} L={best_layer}")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Exp 5: instruction-span vs last-token steering (combine prompts)\n"
        f"model={model_name.split('/')[-1]}  task={task}  injection={injection}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"  Plot saved → {path}")
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--task", default="sentiment",
                   choices=list(opi.FORMAT.keys()),
                   help="Primary task the model is asked to perform.")
    p.add_argument("--injection", default="spam",
                   choices=opi.INJECTIONS,
                   help="Injected task (ignored when --all-injections is set).")
    p.add_argument("--all-injections", action="store_true",
                   help="Run all injected tasks sequentially.")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to steer at. Omit to auto-pick by largest ||v||.")
    p.add_argument("--coefs", type=float, nargs="+",
                   default=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-train", type=int, default=50,
                   help="Prompts used to compute the steering vector.")
    p.add_argument("--n-test", type=int, default=50,
                   help="Prompts used for evaluation (non-overlapping with train).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for the train/test split.")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    injections = opi.INJECTIONS if args.all_injections else [args.injection]

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    n_layers = model.cfg.n_layers
    layers = list(range(n_layers)) if args.layer is None else [args.layer]

    # ── Dataset (load once) ────────────────────────────────────────────────
    print("Loading OPI dataset …")
    opi_ds = opi.load_opi_dataset()
    instruction_text = get_instruction_text(opi_ds, args.task)
    print(f"Instruction: {instruction_text!r}\n")

    # ── Run per injection ──────────────────────────────────────────────────
    for injection in injections:
        print(f"{'='*60}")
        print(f"Task={args.task}  Injection={injection}")
        print(f"{'='*60}")

        prompts  = opi.data_all_attack_types(
            opi_ds, model,
            task_type=args.task, injected_task=injection,
            include_clean=True,
        )
        inj_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[injection])
        cor_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[args.task])

        span, ref, best_layer, baselines = run_one(
            model, prompts, inj_ids, cor_ids,
            instruction_text, layers, args.coefs, args,
        )

        print_table(args.coefs, span, ref)

        tag = f"{args.task}_{injection}_L{best_layer}"
        save_results(
            {**{f"span_{k}": v for k, v in span.items()},
             **{f"ref_{k}": v for k, v in ref.items()},
             **{f"baseline_{c}_{k}": v
                for c, m in baselines.items()
                for k, v in m.items() if k != "logit_diff_per_ex"}},
            os.path.join(args.output_dir, f"exp5_{tag}.json"),
            task=args.task, injection=injection, layer=best_layer,
            coefs=args.coefs, model=args.model,
            n_train=args.n_train, n_test=args.n_test, seed=args.seed,
        )

        plot_results(
            args.coefs, span, ref, baselines,
            args.task, injection, best_layer, args.model,
            os.path.join(args.output_dir, f"exp5_{tag}.png"),
        )
        print()


if __name__ == "__main__":
    main()

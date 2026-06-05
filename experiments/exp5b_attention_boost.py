"""
Experiment 5b — Attention re-focus intervention.

Instead of injecting a residual vector, directly modify the last-token attention
weights so the model pays more attention to the instruction span and less to the
injected content.

The hook targets `blocks.L.attn.hook_pattern` (post-softmax attention weights).
For each selected head h and query position -1 (last token):
    pattern[:, h, -1, inst_start:inst_end] += alpha
    → clamp to 0, renormalise over key positions

This re-focuses the model on the instruction without requiring a condition-specific
direction at instruction positions (which cannot exist due to the causal mask).

Two sweeps
----------
  alpha sweep  — fix hook_layers, vary alpha on combine prompts
  layer sweep  — fix alpha, apply one layer at a time to find responsive layers

Usage
-----
    python experiments/exp5b_attention_boost.py \\
        --task sentiment --all-injections \\
        --hook-layers 15 16 17 18 19 20 21 22 23 \\
        --alphas 0 0.05 0.1 0.2 0.5 1.0 2.0 5.0 \\
        --n-test 50

    # Per-layer sweep to identify which layers respond
    python experiments/exp5b_attention_boost.py \\
        --task sentiment --injection spam --layer-sweep --alpha 1.0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

import src.data.opi as opi
from src.utils.attention_intervention import (
    attention_boost_layer_sweep,
    attention_boost_sweep,
)
from src.utils.steering import cache_resid, compute_metrics
from src.utils.utils import to_first_token_ids
from src.utils.variables import DEVICE, MODEL_NAME


def get_instruction_text(ds, task_type):
    row = next(r for r in ds if r["task_type"] == task_type)
    old, new = opi.FORMAT[task_type]
    return row["instruction"].replace(old, new)


def print_table(alphas, results, label="combine"):
    print(f"\n  [{label}]  {'alpha':>7}  {'asr':>8}  {'ld':>10}")
    for i, a in enumerate(alphas):
        print(f"           {a:>7.3f}  {results['asr'][i]:>8.3f}  {results['ld'][i]:>10.3f}")


def plot_alpha_sweep(alphas, results_combine, results_naive, baseline,
                     task, injection, hook_layers, model_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    keys = {"asr": "ASR", "ld": "Logit diff"}

    for ax, (key, ylabel) in zip(axes, keys.items()):
        ax.plot(alphas, results_combine[key], marker="o", label="combine + boost")
        ax.plot(alphas, results_naive[key],   marker="s", ls="--", label="naive + boost")
        ax.axhline(baseline["combine"][{"asr": "asr", "ld": "mean_logit_diff"}[key]],
                   color="red",   lw=0.8, ls=":", label="combine baseline")
        ax.axhline(baseline["naive"][{"asr": "asr", "ld": "mean_logit_diff"}[key]],
                   color="green", lw=0.8, ls=":", label="naive baseline")
        ax.set(xlabel="alpha", ylabel=ylabel,
               title=f"{ylabel} — {task}/{injection}  layers={sorted(hook_layers)}")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Exp 5b: attention re-focus (instruction boost)\n"
        f"model={model_name.split('/')[-1]}  task={task}  injection={injection}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_layer_sweep(layer_results, baseline_asr, alpha,
                     task, injection, model_name, path):
    layers = sorted(layer_results)
    asrs = [layer_results[l]["asr"] for l in layers]
    lds  = [layer_results[l]["ld"]  for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(layers, asrs, label=f"alpha={alpha}")
    axes[0].axhline(baseline_asr, color="red", lw=0.8, ls="--", label="no hook")
    axes[0].set(xlabel="layer", ylabel="ASR",
                title=f"Per-layer attention boost — {task}/{injection}")
    axes[0].legend(fontsize=8)

    axes[1].plot(layers, lds, marker="o")
    axes[1].set(xlabel="layer", ylabel="logit diff",
                title="Per-layer logit diff")

    plt.suptitle(
        f"Exp 5b layer sweep (alpha={alpha}, all heads)\n"
        f"model={model_name.split('/')[-1]}  task={task}  injection={injection}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def run_one(model, prompts, inj_ids, cor_ids, instruction_text,
            hook_layers_dict, alphas, args):

    test_combine = prompts["combine"][:args.n_test]
    test_naive   = prompts["naive"][:args.n_test]

    # ── Baselines ──────────────────────────────────────────────────────────
    print("  Baselines:")
    baseline = {}
    for cond in ("safe", "naive", "combine"):
        logits, _ = cache_resid(model, prompts[cond][:args.n_test], batch_size=args.batch)
        baseline[cond] = compute_metrics(logits, cor_ids, inj_ids)
        m = baseline[cond]
        print(f"    [{cond:7}] ASR={m['asr']:.3f}  LD={m['mean_logit_diff']:+.3f}")

    if args.layer_sweep:
        # ── Per-layer sweep (one layer at a time) ──────────────────────────
        print(f"\n  [Layer sweep] alpha={args.alpha} on combine prompts …")
        all_layers = list(range(model.cfg.n_layers))
        head_indices = list(range(model.cfg.n_heads))
        layer_results = attention_boost_layer_sweep(
            model, test_combine, cor_ids, inj_ids,
            instruction_text, all_layers, args.alpha,
            model.cfg.n_heads, batch_size=args.batch,
        )
        return None, None, baseline, layer_results

    # ── Alpha sweep ────────────────────────────────────────────────────────
    print(f"\n  [Alpha sweep] hook layers={sorted(hook_layers_dict)} on combine …")
    res_combine = attention_boost_sweep(
        model, test_combine, cor_ids, inj_ids,
        instruction_text, hook_layers_dict, alphas, batch_size=args.batch,
    )
    print(f"  [Alpha sweep] on naive …")
    res_naive = attention_boost_sweep(
        model, test_naive, cor_ids, inj_ids,
        instruction_text, hook_layers_dict, alphas, batch_size=args.batch,
    )
    return res_combine, res_naive, baseline, None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--task", default="sentiment", choices=list(opi.FORMAT.keys()))
    p.add_argument("--injection", default="spam", choices=opi.INJECTIONS)
    p.add_argument("--all-injections", action="store_true")
    p.add_argument("--hook-layers", type=int, nargs="+",
                   default=list(range(15, 24)),
                   help="Layers to apply the attention boost to simultaneously.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--layer-sweep", action="store_true",
                   help="Instead of an alpha sweep, apply alpha to one layer at a time.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Fixed alpha used during --layer-sweep.")
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
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
    n_heads = model.cfg.n_heads

    # All heads in each hook layer
    hook_layers_dict = {l: list(range(n_heads)) for l in args.hook_layers}

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading OPI dataset …")
    opi_ds = opi.load_opi_dataset()
    instruction_text = get_instruction_text(opi_ds, args.task)
    print(f"Instruction: {instruction_text!r}\n")

    # ── Run ────────────────────────────────────────────────────────────────
    all_results = {}
    for injection in injections:
        print(f"{'='*60}")
        print(f"Task={args.task}  Injection={injection}")
        print(f"{'='*60}")

        prompts = opi.data_all_attack_types(
            opi_ds, model,
            task_type=args.task, injected_task=injection,
            include_clean=True,
        )
        inj_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[injection])
        cor_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[args.task])

        res_combine, res_naive, baseline, layer_results = run_one(
            model, prompts, inj_ids, cor_ids, instruction_text,
            hook_layers_dict, args.alphas, args,
        )

        tag = f"{args.task}_{injection}"

        if args.layer_sweep:
            print("\n  Per-layer ASR (combine, alpha={:.2f}):".format(args.alpha))
            print(f"  {'layer':>6}  {'asr':>8}  {'ld':>10}")
            for l in sorted(layer_results):
                r = layer_results[l]
                print(f"  {l:>6}  {r['asr']:>8.3f}  {r['ld']:>10.3f}")

            plot_layer_sweep(
                layer_results, baseline["combine"]["asr"], args.alpha,
                args.task, injection, args.model,
                path=os.path.join(args.output_dir, f"exp5b_{tag}_layersweep.png"),
            )
            all_results[injection] = {
                "baseline": {c: {k: v for k, v in m.items() if k != "logit_diff_per_ex"}
                             for c, m in baseline.items()},
                "layer_sweep": {str(l): r for l, r in layer_results.items()},
                "alpha": args.alpha,
            }
        else:
            print_table(args.alphas, res_combine, label="combine")
            print_table(args.alphas, res_naive,   label="naive")

            plot_alpha_sweep(
                args.alphas, res_combine, res_naive, baseline,
                args.task, injection, args.hook_layers, args.model,
                path=os.path.join(args.output_dir, f"exp5b_{tag}_alphasweep.png"),
            )
            all_results[injection] = {
                "baseline": {c: {k: v for k, v in m.items() if k != "logit_diff_per_ex"}
                             for c, m in baseline.items()},
                "combine": res_combine,
                "naive":   res_naive,
                "alphas":  args.alphas,
                "hook_layers": args.hook_layers,
            }
        print()

    # ── Save ───────────────────────────────────────────────────────────────
    suffix = "layersweep" if args.layer_sweep else "alphasweep"
    out_path = os.path.join(args.output_dir, f"exp5b_{args.task}_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {"task": args.task, "model": args.model,
                     "n_test": args.n_test, "hook_layers": args.hook_layers},
            "results": all_results,
        }, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()

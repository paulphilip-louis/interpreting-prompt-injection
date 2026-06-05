"""
Experiment 6 — Steering vector decomposition: shared vs task-specific components.

Cross-task transfer is only partial: at high steering coefficients, the model
starts following the *wrong* injected task.  This suggests the steering vector
contains two components:

    v_task = alpha * v_shared + v_task_specific

  v_shared        — direction common to injection success across tasks
  v_task_specific — orthogonal residual encoding which task to follow

Question: can v_shared alone steer injection without cross-task contamination?
Does v_task_specific by itself carry meaningful signal?

Method
------
1. Cache residuals (combine − naive) for all injected tasks at layer L.
2. Decompose: shared = renormalised mean of unit vectors; task_specific = v − proj.
3. Compare steering on naive prompts: full / shared-only / task-specific-only.
4. (optional) LOO: train shared direction on all-but-one, evaluate on held-out.

Conditions compared
-------------------
  full           — mean of all per-task diff-of-means (current baseline)
  shared         — unit shared direction scaled to ||full|| (comparable coefs)
  task_specific  — orthogonal residual for the test task itself (sanity check)

Steering vectors are computed on a train split; evaluation uses a non-overlapping
test split so the decomposition cannot overfit to the evaluation examples.

Usage
-----
    python experiments/exp6_decomposition.py --task sentiment --layer 21

    # Sweep coefs, all injections, with LOO
    python experiments/exp6_decomposition.py \\
        --task sentiment --layer 21 \\
        --coefs 0 1 2 3 4 5 6 --n-train 50 --n-test 50 --loo
"""

import argparse
import json
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
    decompose_steering_vecs,
    make_steering_hook,
    steer_decomposed_coef,
)
from src.utils.variables import DEVICE, MODEL_NAME


# ---------------------------------------------------------------------------
# Data / split
# ---------------------------------------------------------------------------

def make_split(n_available: int, n_train: int, n_test: int, seed: int):
    """
    Return (train_idx, test_idx) — non-overlapping random index lists.
    A single split is computed and applied to all injections so comparisons
    across tasks are over the same underlying examples.
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


def build_task_residuals(model, prompts_all, injections, layers, batch, train_idx):
    """
    Cache residuals for each injection × {naive, combine} at the given layers,
    restricted to train_idx examples only.
    Returns {injection: {"naive": {layer: [N_train, d_model]}, "combine": {...}}}
    """
    task_residuals = {}
    for inj in tqdm(injections, desc="caching train residuals"):
        task_residuals[inj] = {}
        for cond in ("naive", "combine"):
            plist = [prompts_all[inj]["prompts"][cond][i] for i in train_idx]
            _, resids = cache_resid(model, plist, batch_size=batch, cache_layers=layers)
            task_residuals[inj][cond] = resids
    return task_residuals


def build_test_prompts(prompts_all, injections, test_idx):
    """
    Return a prompts dict restricted to test_idx, with the same structure as
    prompts_all, so it can be passed directly to steer_decomposed_coef / run_loo.
    """
    test_prompts = {}
    for inj in injections:
        test_prompts[inj] = {
            "prompts": {
                cond: [prompts_all[inj]["prompts"][cond][i] for i in test_idx]
                for cond in prompts_all[inj]["prompts"]
            },
            "cor_ids": prompts_all[inj]["cor_ids"],
            "inj_ids": prompts_all[inj]["inj_ids"],
        }
    return test_prompts


def pick_best_layer(task_residuals, injections, layers):
    """Layer with the largest mean ||combine − naive|| across injections."""
    mean_norms = {}
    for l in layers:
        norms = [
            (task_residuals[inj]["combine"][l].mean(0)
             - task_residuals[inj]["naive"][l].mean(0)).norm().item()
            for inj in injections
        ]
        mean_norms[l] = float(np.mean(norms))
    best = max(mean_norms, key=mean_norms.__getitem__)
    return best, mean_norms


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def print_decomposition_table(decomp, injections):
    header = f"  {'injection':>12}  {'||v||':>8}  {'proj_onto_shared':>17}  {'||task_specific||':>18}  {'shared_frac':>12}"
    print(header)
    for inj in injections:
        v      = decomp["task_vecs"][inj]
        proj   = decomp["projections"][inj]
        ts_norm = decomp["task_specific"][inj].norm().item()
        v_norm  = v.norm().item()
        frac    = abs(proj) / v_norm if v_norm > 0 else 0.0
        print(f"  {inj:>12}  {v_norm:>8.3f}  {proj:>17.3f}  {ts_norm:>18.3f}  {frac:>12.3f}")


def print_cosine_table(decomp, injections):
    print("  " + "".join(f"{inj:>10}" for inj in [""] + injections))
    for a in injections:
        row = f"  {a:>10}"
        for b in injections:
            va, vb = decomp["task_vecs"][a], decomp["task_vecs"][b]
            cos = torch.nn.functional.cosine_similarity(
                va.unsqueeze(0), vb.unsqueeze(0)
            ).item()
            row += f"{cos:>10.3f}"
        print(row)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cosine_heatmap(vecs, injections, title, path):
    n = len(injections)
    mat = np.zeros((n, n))
    for i, a in enumerate(injections):
        for j, b in enumerate(injections):
            mat[i, j] = torch.nn.functional.cosine_similarity(
                vecs[a].unsqueeze(0), vecs[b].unsqueeze(0)
            ).item()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(injections, rotation=45, ha="right")
    ax.set_yticklabels(injections)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_decomp_bars(decomp, injections, path):
    """Shared projection magnitude vs task-specific norm per injection."""
    proj_norms = [abs(decomp["projections"][inj]) for inj in injections]
    ts_norms   = [decomp["task_specific"][inj].norm().item() for inj in injections]

    x, w = np.arange(len(injections)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, proj_norms, w, label="|proj onto shared|")
    ax.bar(x + w/2, ts_norms,   w, label="||task_specific||")
    ax.set_xticks(x); ax.set_xticklabels(injections)
    ax.set_ylabel("norm (same units as ||v||)")
    ax.set_title("Shared vs task-specific component magnitude per injection")
    ax.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_steering_comparison(results, coefs, injections, best_layer, task, model_name, path):
    n = len(injections)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    styles = {"full": "-", "shared": "--", "task_specific": ":"}
    colors = {"full": "tab:blue", "shared": "tab:orange", "task_specific": "tab:green"}

    for row, inj in enumerate(injections):
        for cond in ("full", "shared", "task_specific"):
            axes[row, 0].plot(coefs, results[inj][cond]["asr"],
                              label=cond, ls=styles[cond], color=colors[cond], marker="o", ms=4)
            axes[row, 1].plot(coefs, results[inj][cond]["ld"],
                              label=cond, ls=styles[cond], color=colors[cond], marker="o", ms=4)
        for ax in axes[row]:
            ax.axvline(0, color="gray", lw=0.5, ls=":")
        axes[row, 0].set(ylabel="ASR", title=f"injection={inj} — ASR")
        axes[row, 1].set(ylabel="logit diff", title=f"injection={inj} — LD")
        axes[row, 0].legend(fontsize=8); axes[row, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("coefficient")

    plt.suptitle(
        f"Exp 6: full / shared-only / task-specific steering on naive prompts\n"
        f"model={model_name.split('/')[-1]}  task={task}  L={best_layer}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_loo(loo_results, coefs, injections, best_layer, task, model_name, path):
    n = len(injections)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    styles = {"full": "-", "shared": "--"}
    colors = {"full": "tab:blue", "shared": "tab:orange"}

    for row, held_out in enumerate(injections):
        for cond in ("full", "shared"):
            axes[row, 0].plot(coefs, loo_results[held_out][cond]["asr"],
                              label=cond, ls=styles[cond], color=colors[cond], marker="o", ms=4)
            axes[row, 1].plot(coefs, loo_results[held_out][cond]["ld"],
                              label=cond, ls=styles[cond], color=colors[cond], marker="o", ms=4)
        for ax in axes[row]:
            ax.axvline(0, color="gray", lw=0.5, ls=":")
        axes[row, 0].set(ylabel="ASR", title=f"held-out={held_out} — ASR")
        axes[row, 1].set(ylabel="logit diff", title=f"held-out={held_out} — LD")
        axes[row, 0].legend(fontsize=8); axes[row, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("coefficient")

    plt.suptitle(
        f"Exp 6 LOO: shared direction trained on other tasks → held-out task\n"
        f"model={model_name.split('/')[-1]}  task={task}  L={best_layer}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# LOO sweep
# ---------------------------------------------------------------------------

def run_loo(model, test_prompts, task_residuals, injections, best_layer, coefs, batch):
    """
    For each injection as held-out:
      - decompose using the remaining injections' *train* residuals only
      - scale shared vector to match the train-tasks full vector norm
      - evaluate full and shared-only on the held-out task's *test* prompts
    """
    print("\n  [LOO] Leave-one-out evaluation …")
    loo_results = {}

    for held_out in injections:
        train_tasks = [t for t in injections if t != held_out]
        decomp_loo  = decompose_steering_vecs(task_residuals, train_tasks, best_layer)

        full_vec   = torch.stack(
            [decomp_loo["task_vecs"][t] for t in train_tasks]
        ).mean(0).to(DEVICE)
        shared = decomp_loo["shared"].to(DEVICE)
        shared_vec = (shared * full_vec.norm())

        target  = test_prompts[held_out]["prompts"]["naive"]
        cor_ids = test_prompts[held_out]["cor_ids"]
        inj_ids = test_prompts[held_out]["inj_ids"]

        loo_results[held_out] = {
            "full":   {"asr": [], "ld": []},
            "shared": {"asr": [], "ld": []},
        }
        for coef in tqdm(coefs, desc=f"    LOO held-out={held_out}"):
            for label, vec in [("full", full_vec), ("shared", shared_vec)]:
                hook = make_steering_hook(vec, coef)
                logits, _ = cache_resid(
                    model, target, batch_size=batch,
                    fwd_hooks=[(f"blocks.{best_layer}.hook_resid_post", hook)],
                )
                m = compute_metrics(logits, cor_ids, inj_ids)
                loo_results[held_out][label]["asr"].append(m["asr"])
                loo_results[held_out][label]["ld"].append(m["mean_logit_diff"])

    return loo_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--task", default="sentiment",
                   choices=list(opi.FORMAT.keys()),
                   help="Primary task the model is asked to perform.")
    p.add_argument("--injections", nargs="+", default=opi.INJECTIONS,
                   help="Which injected tasks to include.")
    p.add_argument("--layer", type=int, default=None,
                   help="Layer for decomposition. Omit to auto-pick by mean ||v||.")
    p.add_argument("--coefs", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-train", type=int, default=50,
                   help="Prompts used to compute the steering vectors.")
    p.add_argument("--n-test", type=int, default=50,
                   help="Prompts used for evaluation (non-overlapping with train).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for the train/test split.")
    p.add_argument("--loo", action="store_true",
                   help="Run leave-one-out cross-task evaluation.")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    injections = args.injections

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    n_layers = model.cfg.n_layers
    layers   = list(range(n_layers)) if args.layer is None else [args.layer]

    # ── Data ───────────────────────────────────────────────────────────────
    print(f"Loading OPI (task={args.task}, injections={injections}) …")
    prompts_all = opi.load_opi_per_task(model, args.task)

    # ── Train / test split ─────────────────────────────────────────────────
    n_available = min(
        min(len(prompts_all[inj]["prompts"]["combine"]) for inj in injections),
        min(len(prompts_all[inj]["prompts"]["naive"])   for inj in injections),
    )
    train_idx, test_idx = make_split(n_available, args.n_train, args.n_test, args.seed)
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test "
          f"(seed={args.seed}, n_available={n_available})")

    test_prompts = build_test_prompts(prompts_all, injections, test_idx)

    # ── Cache residuals (train only) ───────────────────────────────────────
    print(f"\nCaching residuals ({len(injections)} injections × 2 conditions × {len(layers)} layer(s)) …")
    task_residuals = build_task_residuals(
        model, prompts_all, injections, layers, args.batch, train_idx
    )

    # ── Layer selection ────────────────────────────────────────────────────
    if args.layer is None:
        best_layer, mean_norms = pick_best_layer(task_residuals, injections, layers)
        print(f"Best layer by mean ||v||: L={best_layer} (norm={mean_norms[best_layer]:.2f})")
    else:
        best_layer = args.layer
    tag = f"{args.task}_L{best_layer}"

    # ── Decomposition ──────────────────────────────────────────────────────
    print(f"\nDecomposing steering vectors at L={best_layer} …")
    decomp = decompose_steering_vecs(task_residuals, injections, best_layer)

    print("\n  Component norms:")
    print_decomposition_table(decomp, injections)

    print("\n  Full-vector cosine similarity table:")
    print_cosine_table(decomp, injections)

    plot_cosine_heatmap(
        decomp["task_vecs"], injections,
        title=f"Cosine similarity — full vectors  L={best_layer}",
        path=os.path.join(args.output_dir, f"exp6_{tag}_cosine_full.png"),
    )
    plot_cosine_heatmap(
        decomp["task_specific"], injections,
        title=f"Cosine similarity — task-specific only  L={best_layer}",
        path=os.path.join(args.output_dir, f"exp6_{tag}_cosine_taskspec.png"),
    )
    plot_decomp_bars(
        decomp, injections,
        path=os.path.join(args.output_dir, f"exp6_{tag}_decomp_bars.png"),
    )

    # ── Steering comparison ────────────────────────────────────────────────
    print(f"\nSteering comparison (full / shared / task_specific) at L={best_layer} …")
    steer_results = steer_decomposed_coef(
        model, test_prompts, decomp,
        test_tasks=injections,
        layer=best_layer,
        coefs=args.coefs,
        batch=args.batch,
        n_max=len(test_idx),
        plotting=False,
    )

    print("\n  ASR summary (full / shared / task_specific) at each coef:")
    print(f"  {'coef':>6}" + "".join(
        f"  {inj}_full  {inj}_shared  {inj}_ts" for inj in injections
    ))
    for j, c in enumerate(args.coefs):
        row = f"  {c:>6.1f}"
        for inj in injections:
            r = steer_results[inj]
            row += (f"  {r['full']['asr'][j]:>9.3f}"
                    f"  {r['shared']['asr'][j]:>10.3f}"
                    f"  {r['task_specific']['asr'][j]:>6.3f}")
        print(row)

    plot_steering_comparison(
        steer_results, args.coefs, injections,
        best_layer, args.task, args.model,
        path=os.path.join(args.output_dir, f"exp6_{tag}_steering.png"),
    )

    # ── LOO ────────────────────────────────────────────────────────────────
    loo_results = None
    if args.loo and len(injections) > 1:
        loo_results = run_loo(
            model, test_prompts, task_residuals, injections,
            best_layer, args.coefs, args.batch,
        )
        print("\n  LOO ASR (full / shared) at max coef:")
        max_idx = len(args.coefs) - 1
        for held_out in injections:
            r = loo_results[held_out]
            print(f"  held-out={held_out:>10}: "
                  f"full={r['full']['asr'][max_idx]:.3f}  "
                  f"shared={r['shared']['asr'][max_idx]:.3f}")

        plot_loo(
            loo_results, args.coefs, injections,
            best_layer, args.task, args.model,
            path=os.path.join(args.output_dir, f"exp6_{tag}_loo.png"),
        )

    # ── Save ───────────────────────────────────────────────────────────────
    def to_json(obj):
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, dict):         return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):         return [to_json(v) for v in obj]
        return obj

    payload = {
        "meta": {
            "task": args.task, "injections": injections,
            "layer": best_layer, "coefs": args.coefs, "model": args.model,
            "n_train": args.n_train, "n_test": args.n_test, "seed": args.seed,
        },
        "decomp": {
            "projections":        decomp["projections"],
            "task_vec_norms":     {inj: decomp["task_vecs"][inj].norm().item()     for inj in injections},
            "task_specific_norms":{inj: decomp["task_specific"][inj].norm().item() for inj in injections},
            "shared_frac":        {
                inj: abs(decomp["projections"][inj]) / decomp["task_vecs"][inj].norm().item()
                for inj in injections
            },
            "shared_vec": decomp["shared"].tolist(),
        },
        "steering": to_json(steer_results),
    }
    if loo_results:
        payload["loo"] = to_json(loo_results)

    out_path = os.path.join(args.output_dir, f"exp6_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

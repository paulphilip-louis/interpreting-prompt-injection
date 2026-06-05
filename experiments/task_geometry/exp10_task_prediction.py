"""
Experiment 10 — Predicting injection directions from task representations.

Can we predict the injection-following direction for a new primary task T
using only that task's representation (computable from a single clean forward
pass), without running any adversarial examples?

If yes: given an unseen instruction, run one clean forward pass → task_vec
→ apply W → predicted injection_vec → steer with negative coefficient
→ zero-shot defense against prompt injection.

Method
------
We treat this as a leave-one-out (LOO) linear regression problem:

    injection_vec[T] ≈ W @ task_vec[T]

With n=5 tasks (or 4 after excluding the task==injection case), we have
5 data points in d_model-dimensional space.  A full d_model×d_model W
is unidentifiable — there are infinitely many W fitting 4 points in
3000-dimensional space.  We instead project task_vecs to their top-k PCA
components (k < n_train, so the system is overdetermined in a meaningful way)
and fit ridge regression there.

For each held-out task T:
  1. Fit PCA on the remaining task_vecs (k=min(n_tasks-2, 3)).
  2. Fit ridge regression W_low: inj_vec[T'] ≈ W_low @ pca_coords[T'] for T'≠T.
  3. Predict: inj_vec_pred[T] = W_low @ pca_coords[T] (projected back to d_model).
  4. Measure:
       (a) cos(inj_vec_pred[T], inj_vec_empirical[T])
       (b) ASR with predicted vector vs empirical vector vs random baseline
           on held-out test prompts for task T.

Visualisations
--------------
  (a) Bar chart: LOO cosine similarity per task
  (b) ASR comparison: predicted / empirical / random at fixed coefficient
  (c) Singular values of the regression map (rank of the task→injection relationship)
  (d) Scatter: predicted vs actual injection vec coordinates in PCA space

Usage
-----
    python experiments/task_geometry/exp10_task_prediction.py \\
        --injection spam --layer 21 \\
        --task-vectors results/task_geometry/task_vectors.pt \\
        --inj-vectors  results/task_geometry/injection_vectors_spam_L21.pt

    # Recompute everything from scratch:
    python experiments/task_geometry/exp10_task_prediction.py \\
        --injection spam --layer 21 --n-train 100 --n-test 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

import src.data.opi as opi
from src.utils.steering import cache_resid, compute_metrics, make_steering_hook
from src.utils.utils import to_first_token_ids
from src.utils.variables import DEVICE, MODEL_NAME
from experiments.task_geometry.exp8_task_geometry import (
    TASKS, TASK_COLORS, pca_fit, pca_project, load_safe_prompts,
)
from experiments.task_geometry.exp9_injection_geometry import (
    compute_injection_vecs, load_task_vecs,
)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def ridge_regression(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-2):
    """
    Fit W such that Y ≈ W @ X.T (each column of X/Y is one data point).
    X: [d, n_train], Y: [d, n_train]
    Returns W: [d, d] (effectively low-rank because n_train << d)
    """
    # W = Y @ X.T @ (X @ X.T + λI)^{-1}
    XtX = X @ X.T                                   # [n, n]
    W = Y @ X.T @ torch.linalg.inv(XtX + lam * torch.eye(XtX.shape[0]))
    return W   # [d, n] @ [n, n] → [d, n] then [d, d] — actually [d, d]


def low_rank_regression(task_vecs: list, inj_vecs: list, k: int, lam: float = 1e-4):
    """
    Fit regression in the k-dim PCA subspace of task_vecs.
    task_vecs: list of [d_model] tensors (training tasks)
    inj_vecs:  list of [d_model] tensors (corresponding injection vecs)
    Returns (predict_fn, comps [k,d], pca_mean [d])
    """
    X = torch.stack(task_vecs).float()   # [n, d]
    Y = torch.stack(inj_vecs).float()    # [n, d]

    comps, pca_mean = pca_fit(X, k=k)                   # [k, d], [d]
    X_proj = torch.tensor(pca_project(X, comps, pca_mean))   # [n, k]
    Y_proj = torch.tensor(pca_project(Y, comps, pca_mean))   # [n, k] (project inj into same space)

    # Fit k→k regression: Y_proj ≈ W_kk @ X_proj.T
    XtX = X_proj.T @ X_proj + lam * torch.eye(k)         # [k, k]
    W_kk = Y_proj.T @ X_proj @ torch.linalg.inv(XtX)     # [k, k]

    def predict(task_vec: torch.Tensor) -> torch.Tensor:
        """Given task_vec [d], return predicted injection_vec [d]."""
        coords = comps @ (task_vec.float() - pca_mean)    # [k]
        pred_coords = W_kk @ coords                        # [k]
        # Reconstruct in d_model space: back-project via PCA components
        return (comps.T @ pred_coords) + pca_mean          # [d]

    return predict, W_kk, comps, pca_mean


def random_unit_vec(d: int, seed: int) -> torch.Tensor:
    """Random unit vector of dimension d."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------
# ASR evaluation
# ---------------------------------------------------------------------------

def eval_asr(model, prompts, cor_ids, inj_ids, vec, coef, layer, batch):
    """Apply steering hook and measure ASR."""
    hook = make_steering_hook(vec.to(DEVICE), coef)
    logits, _ = cache_resid(
        model, prompts, batch_size=batch,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)],
    )
    return compute_metrics(logits, cor_ids, inj_ids)["asr"]


def load_test_prompts(opi_ds, model, task, injection, n_test, seed):
    """Load n_test held-out attack prompts for (task, injection)."""
    prompts = opi.data_all_attack_types(
        opi_ds, model, task_type=task, injected_task=injection, include_clean=False,
    )
    n = min(len(prompts["naive"]), n_test)
    gen = torch.Generator().manual_seed(seed + 999)   # different seed from training
    idx = torch.randperm(n, generator=gen)[:n].tolist()
    return {k: [v[i] for i in idx] for k, v in prompts.items()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loo_cosines(tasks, cosines, injection, layer, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [TASK_COLORS[t] for t in tasks]
    bars = ax.bar(tasks, cosines, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.4)
    for bar, cos in zip(bars, cosines):
        ax.text(bar.get_x() + bar.get_width() / 2, cos + 0.02,
                f"{cos:.2f}", ha="center", fontsize=9)
    ax.set(xlabel="held-out task", ylabel="cos(predicted, empirical)",
           ylim=(-0.2, 1.1),
           title=f"LOO: predicted vs empirical injection direction\n"
                 f"injection={injection}  L={layer}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp10_loo_cosines_{injection}_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_asr_comparison(tasks, asr_pred, asr_emp, asr_rand, coef, injection, layer, out_dir):
    x = np.arange(len(tasks))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, asr_pred, w, label="predicted vec",  color="steelblue",  edgecolor="black", lw=0.5)
    ax.bar(x,     asr_emp,  w, label="empirical vec",  color="tomato",     edgecolor="black", lw=0.5)
    ax.bar(x + w, asr_rand, w, label="random vec",     color="lightgray",  edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set(ylabel="ASR", ylim=(0, 1.05),
           title=f"Defense: ASR with predicted / empirical / random vector\n"
                 f"coef={coef}  injection={injection}  L={layer}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp10_asr_comparison_{injection}_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_singular_values(W_kk_matrices, tasks, injection, layer, out_dir):
    """
    For each LOO fold, show singular values of W_kk (the low-dim regression map).
    Reveals the effective rank of the task → injection relationship.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (t, W) in enumerate(zip(tasks, W_kk_matrices)):
        sv = torch.linalg.svdvals(W).numpy()
        ax.plot(range(1, len(sv) + 1), sv, marker="o", ms=6,
                color=TASK_COLORS[t], label=f"held-out={t}")
    ax.set(xlabel="singular value index", ylabel="magnitude",
           title=f"Singular values of low-dim regression map W\n"
                 f"injection={injection}  L={layer}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp10_singular_values_{injection}_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_predicted_vs_actual_pca(pred_vecs, emp_vecs, tasks, injection, layer, out_dir):
    """
    Scatter of predicted and empirical injection vecs in PCA space.
    Lines connect each predicted point to its empirical counterpart.
    """
    all_vecs = [emp_vecs[t].float() for t in tasks] + \
               [pred_vecs[t].float() for t in tasks if t in pred_vecs]
    X = torch.stack(all_vecs)
    comps, mean = pca_fit(X, k=2)

    emp_coords  = pca_project(torch.stack([emp_vecs[t].float()  for t in tasks]), comps, mean)
    pred_coords = pca_project(
        torch.stack([pred_vecs[t].float() for t in tasks if t in pred_vecs]),
        comps, mean
    )
    pred_tasks = [t for t in tasks if t in pred_vecs]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, t in enumerate(tasks):
        c = TASK_COLORS[t]
        ax.scatter(emp_coords[i, 0], emp_coords[i, 1],
                   color=c, s=180, marker="o", zorder=5, label=f"{t} empirical",
                   edgecolors="black", lw=0.5)
    for i, t in enumerate(pred_tasks):
        c = TASK_COLORS[t]
        ax.scatter(pred_coords[i, 0], pred_coords[i, 1],
                   color=c, s=180, marker="X", zorder=5, alpha=0.7,
                   edgecolors="black", lw=0.5)
        # connect predicted to empirical
        ei = tasks.index(t)
        ax.plot([emp_coords[ei, 0], pred_coords[i, 0]],
                [emp_coords[ei, 1], pred_coords[i, 1]],
                color=c, lw=1.0, ls="--", alpha=0.6)
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"Predicted (×) vs empirical (●) injection vecs\n"
                 f"injection={injection}  L={layer}")
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0], [0], marker="o", ls="", color=TASK_COLORS[t], ms=9) for t in tasks]
    legend_el += [
        Line2D([0], [0], marker="o", ls="", color="gray", ms=9, label="● empirical"),
        Line2D([0], [0], marker="X", ls="", color="gray", ms=9, label="× predicted"),
    ]
    ax.legend(handles=legend_el, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp10_predicted_vs_actual_L{layer}_{injection}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model",        default=MODEL_NAME)
    p.add_argument("--injection",    default="spam", choices=opi.INJECTIONS)
    p.add_argument("--layer",        type=int, default=21)
    p.add_argument("--n-train",      type=int, default=100)
    p.add_argument("--n-test",       type=int, default=50)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--batch",        type=int, default=4)
    p.add_argument("--coef",         type=float, default=-3.0,
                   help="Coefficient for steering at test time (negative = defense).")
    p.add_argument("--ridge-lam",    type=float, default=1e-4,
                   help="Ridge regularisation lambda.")
    p.add_argument("--pca-k",        type=int, default=None,
                   help="PCA components for regression. Default: n_tasks - 2.")
    p.add_argument("--task-vectors", default="results/task_geometry/task_vectors.pt")
    p.add_argument("--inj-vectors",  default=None,
                   help="Path to injection_vectors_*.pt from exp9. Computed if absent.")
    p.add_argument("--output-dir",   default="results/task_geometry")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    # ── Load / compute task vectors ────────────────────────────────────────
    if os.path.exists(args.task_vectors):
        print(f"Loading task vectors from {args.task_vectors} …")
        task_vecs = load_task_vecs(args.task_vectors, args.layer)
    else:
        print("Computing task vectors on the fly …")
        opi_ds = opi.load_opi_dataset()
        task_vecs = {}
        for task in tqdm(TASKS, desc="task vecs"):
            prompts = load_safe_prompts(opi_ds, model, task, args.n_train, args.seed)
            _, resids = cache_resid(model, prompts, batch_size=args.batch,
                                    cache_layers=[args.layer])
            task_vecs[task] = resids[args.layer].float().mean(0).cpu()

    # ── Load / compute injection vectors ───────────────────────────────────
    inj_path = args.inj_vectors or os.path.join(
        args.output_dir, f"injection_vectors_{args.injection}_L{args.layer}.pt"
    )
    if os.path.exists(inj_path):
        print(f"Loading injection vectors from {inj_path} …")
        data = torch.load(inj_path, map_location="cpu")
        inj_vecs_raw = data["inj_vecs"]
        inj_vecs = {t: inj_vecs_raw[t][args.layer] for t in data["tasks"]}
    else:
        print("Computing injection vectors on the fly …")
        opi_ds = opi.load_opi_dataset()
        raw = compute_injection_vecs(
            model, opi_ds, TASKS, args.injection,
            layers=[args.layer], n_train=args.n_train,
            seed=args.seed, batch=args.batch,
        )
        inj_vecs = {t: raw[t][args.layer] for t in raw}

    valid_tasks = [t for t in TASKS if t in inj_vecs]
    k = args.pca_k if args.pca_k is not None else max(1, len(valid_tasks) - 2)
    print(f"\nTasks: {valid_tasks}  |  PCA k={k}  |  λ={args.ridge_lam}")

    # ── LOO regression ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Leave-one-out regression  (task_vec → injection_vec)")
    print(f"{'='*60}")

    loo_cosines     = []
    asr_predicted   = []
    asr_empirical   = []
    asr_random      = []
    W_kk_matrices   = []
    pred_vecs_dict  = {}

    opi_ds = opi.load_opi_dataset()

    for held_out in valid_tasks:
        train_tasks = [t for t in valid_tasks if t != held_out]
        train_tvecs = [task_vecs[t].float() for t in train_tasks]
        train_ivecs = [inj_vecs[t].float()  for t in train_tasks]

        k_eff = min(k, len(train_tasks) - 1)
        if k_eff < 1:
            print(f"  [{held_out}] Not enough training tasks for regression — skipping")
            continue

        predict_fn, W_kk, _, _ = low_rank_regression(
            train_tvecs, train_ivecs, k=k_eff, lam=args.ridge_lam
        )
        W_kk_matrices.append(W_kk)

        # Predict injection_vec for held-out task
        pred_vec = predict_fn(task_vecs[held_out].float()).cpu()
        emp_vec  = inj_vecs[held_out].float().cpu()
        pred_vecs_dict[held_out] = pred_vec

        cos = torch.nn.functional.cosine_similarity(
            pred_vec.unsqueeze(0), emp_vec.unsqueeze(0)
        ).item()
        loo_cosines.append(cos)
        print(f"  [{held_out:>12}] cos(pred, emp) = {cos:+.3f}  "
              f"||pred||={pred_vec.norm():.2f}  ||emp||={emp_vec.norm():.2f}")

        # ASR evaluation
        test_prompts = load_test_prompts(opi_ds, model, held_out, args.injection,
                                          args.n_test, args.seed)
        cor_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[held_out])
        inj_ids = to_first_token_ids(model, opi.ANSWER_STRINGS[args.injection])

        # Scale predicted to match empirical norm (same effective coef)
        pred_scaled = pred_vec / (pred_vec.norm() + 1e-8) * emp_vec.norm()

        rnd_vec  = random_unit_vec(pred_vec.shape[0], seed=42) * emp_vec.norm()

        asr_p = eval_asr(model, test_prompts["combine"], cor_ids, inj_ids,
                         pred_scaled, args.coef, args.layer, args.batch)
        asr_e = eval_asr(model, test_prompts["combine"], cor_ids, inj_ids,
                         emp_vec,     args.coef, args.layer, args.batch)
        asr_r = eval_asr(model, test_prompts["combine"], cor_ids, inj_ids,
                         rnd_vec,     args.coef, args.layer, args.batch)

        asr_predicted.append(asr_p)
        asr_empirical.append(asr_e)
        asr_random.append(asr_r)
        print(f"            ASR: predicted={asr_p:.3f}  empirical={asr_e:.3f}  "
              f"random={asr_r:.3f}  (coef={args.coef})")

    loo_tasks = [t for t in valid_tasks if t in pred_vecs_dict]

    # ── Print final table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'task':>12}  {'cos(p,e)':>9}  {'asr_pred':>9}  {'asr_emp':>8}  {'asr_rand':>9}")
    for t, cos, ap, ae, ar in zip(loo_tasks, loo_cosines, asr_predicted, asr_empirical, asr_random):
        print(f"{t:>12}  {cos:>9.3f}  {ap:>9.3f}  {ae:>8.3f}  {ar:>9.3f}")
    print(f"\nMean cos(pred, emp): {np.mean(loo_cosines):.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_loo_cosines(loo_tasks, loo_cosines, args.injection, args.layer, args.output_dir)
    plot_asr_comparison(
        loo_tasks, asr_predicted, asr_empirical, asr_random,
        args.coef, args.injection, args.layer, args.output_dir,
    )
    if W_kk_matrices:
        plot_singular_values(W_kk_matrices, loo_tasks, args.injection, args.layer, args.output_dir)
    plot_predicted_vs_actual_pca(
        pred_vecs_dict, {t: inj_vecs[t] for t in valid_tasks},
        valid_tasks, args.injection, args.layer, args.output_dir,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir,
                             f"exp10_loo_{args.injection}_L{args.layer}.pt")
    torch.save({
        "loo_cosines":   dict(zip(loo_tasks, loo_cosines)),
        "asr_predicted": dict(zip(loo_tasks, asr_predicted)),
        "asr_empirical": dict(zip(loo_tasks, asr_empirical)),
        "asr_random":    dict(zip(loo_tasks, asr_random)),
        "pred_vecs":     pred_vecs_dict,
        "injection":     args.injection,
        "layer":         args.layer,
        "coef":          args.coef,
        "model":         args.model,
    }, save_path)
    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()

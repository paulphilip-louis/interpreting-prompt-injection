"""
Experiment 8 — Task representation geometry.

Where and how are primary tasks encoded in the residual stream?

For each of the 5 primary tasks (sentiment, spam, hsol, rte, mrpc), we collect
N safe (uninjected) prompts and cache the last-token residual at each layer.
The per-task mean residual is the "task vector" at that layer.

Analysis
--------
  (a) Layer sweep: which layer best discriminates tasks?
      Metric: mean pairwise cosine distance across task vectors.
  (b) PCA 2D scatter at the best (or chosen) layer — 5 task mean vectors as
      points, plus per-sample scatter showing within/between-task variance.
  (c) Cosine similarity heatmap between task vectors.
  (d) Per-sample residuals plotted in the task-PCA plane — visualises how
      cleanly tasks cluster relative to within-task variance.

The mean task vectors are saved to disk for use in exp9 and exp10.

Usage
-----
    python experiments/task_geometry/exp8_task_geometry.py
        --layer 21           # only analyse this layer (fast)
        --all-layers         # sweep all layers (slow, needed for (a))
        --n 100              # prompts per task
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
from src.utils.steering import cache_resid
from src.utils.variables import DEVICE, MODEL_NAME

# The 5 tasks with categorical answers — usable as both primary and injected tasks
TASKS = list(opi.FORMAT.keys())  # ["sentiment", "spam", "mrpc", "hsol", "rte"]

TASK_COLORS = {
    "sentiment": "#e41a1c",
    "spam":      "#377eb8",
    "hsol":      "#4daf4a",
    "rte":       "#984ea3",
    "mrpc":      "#ff7f00",
}


# ---------------------------------------------------------------------------
# PCA helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def pca_fit(X: torch.Tensor, k: int = 2):
    """Fit PCA on X [N, d], return (components [k, d], mean [d])."""
    mean = X.mean(0)
    Xc = X - mean
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    return Vt[:k], mean   # [k, d], [d]


def pca_project(X: torch.Tensor, components: torch.Tensor, mean: torch.Tensor):
    """Project X [N, d] → [N, k]."""
    return ((X - mean) @ components.T).numpy()


def pca_variance_explained(X: torch.Tensor, k: int = 2):
    """Fraction of variance explained by top-k PCs."""
    Xc = X - X.mean(0)
    sv = torch.linalg.svdvals(Xc)
    return (sv[:k].pow(2).sum() / sv.pow(2).sum()).item()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_safe_prompts(opi_ds, model, task: str, n: int, seed: int) -> list[str]:
    """
    Collect up to `n` safe (uninjected) prompts for `task`.
    Uses the first valid injection (≠ task) just to get the normal_input rows.
    """
    ref_inj = next(inj for inj in opi.INJECTIONS if inj != task)
    prompts = opi.data_all_attack_types(
        opi_ds, model, task_type=task, injected_task=ref_inj, include_clean=True
    )["safe"]
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(prompts), generator=gen)[:n].tolist()
    return [prompts[i] for i in idx]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def between_task_mean_cosine_distance(vecs: dict) -> float:
    """Mean pairwise cosine distance between task vectors (1 − cosine_sim)."""
    tasks = list(vecs.keys())
    dists = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            a, b = vecs[tasks[i]], vecs[tasks[j]]
            cos = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            dists.append(1 - cos)
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_layer_sweep(layers, disc_scores, best_layer, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, disc_scores, marker="o", ms=4, lw=1.5)
    ax.axvline(best_layer, color="red", lw=0.8, ls="--", label=f"best L={best_layer}")
    ax.set(xlabel="layer", ylabel="mean pairwise cosine distance",
           title="Task discriminability across layers\n"
                 "(higher = task vectors more spread apart)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "exp8_layer_sweep.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_pca_means(mean_vecs, layer, out_dir, var_exp=None):
    """2D PCA scatter of the 5 task mean vectors."""
    tasks = list(mean_vecs.keys())
    X = torch.stack([mean_vecs[t] for t in tasks])
    comps, mean = pca_fit(X, k=2)
    coords = pca_project(X, comps, mean)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, t in enumerate(tasks):
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=TASK_COLORS[t], s=180, zorder=5, label=t)
        ax.annotate(t, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(7, 4), fontsize=10)
    title = f"Task vectors — PCA @ L={layer}"
    if var_exp is not None:
        title += f"\n(PC1+PC2 explain {var_exp:.1%} of variance)"
    ax.set(xlabel="PC 1", ylabel="PC 2", title=title)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp8_pca_means_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")
    return comps, mean   # for projecting samples later


def plot_pca_samples(sample_resids, mean_vecs, layer, comps, pca_mean, out_dir):
    """Per-sample scatter in the task-mean PCA plane, with centroids."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for task, resids in sample_resids.items():
        X = resids.float().cpu()   # [N, d]
        coords = pca_project(X, comps, pca_mean)
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=TASK_COLORS[task], alpha=0.25, s=20, label=f"{task} (samples)")
    # centroids on top
    tasks = list(mean_vecs.keys())
    X_means = torch.stack([mean_vecs[t] for t in tasks]).float().cpu()
    c_coords = pca_project(X_means, comps, pca_mean)
    for i, t in enumerate(tasks):
        ax.scatter(c_coords[i, 0], c_coords[i, 1],
                   color=TASK_COLORS[t], s=200, marker="*", zorder=10, edgecolors="black", lw=0.5)
        ax.annotate(t, (c_coords[i, 0], c_coords[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"Per-sample task residuals @ L={layer}\n"
                 f"(* = task centroid, dots = individual prompts)")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=TASK_COLORS[t],
                          label=t, ms=8) for t in TASKS]
    ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp8_pca_samples_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_cosine_heatmap(vecs, title, path):
    tasks = list(vecs.keys())
    n = len(tasks)
    mat = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            mat[i, j] = torch.nn.functional.cosine_similarity(
                vecs[a].unsqueeze(0), vecs[b].unsqueeze(0)
            ).item()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticklabels(tasks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_norms_per_layer(task_vecs_by_layer, tasks, layers, out_dir):
    """||task_vec||  vs layer for each task."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for t in tasks:
        norms = [task_vecs_by_layer[l][t].norm().item() for l in layers]
        ax.plot(layers, norms, color=TASK_COLORS[t], label=t, lw=1.5)
    ax.set(xlabel="layer", ylabel="||task vector||",
           title="Task vector magnitude per layer\n(mean residual of safe prompts)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "exp8_norms_per_layer.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model",       default=MODEL_NAME)
    p.add_argument("--n",           type=int, default=100,
                   help="Safe prompts per task.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--layer",       type=int, default=21,
                   help="Layer to analyse in depth (PCA, heatmap, sample scatter).")
    p.add_argument("--all-layers",  action="store_true",
                   help="Also sweep all layers for the discriminability plot and norm plot.")
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--output-dir",  default="results/task_geometry")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    n_layers = model.cfg.n_layers

    layers_to_cache = list(range(n_layers)) if args.all_layers else [args.layer]

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Loading OPI dataset …")
    opi_ds = opi.load_opi_dataset()

    # ── Collect residuals ──────────────────────────────────────────────────
    # {task: {layer: mean_resid [d_model]}}
    task_mean_vecs = {}
    # {layer: {task: mean_resid}}  — for layer sweep
    vecs_by_layer  = {l: {} for l in layers_to_cache}
    # {task: [N, d_model]}  — per-sample at args.layer (for sample scatter)
    sample_resids  = {}

    for task in tqdm(TASKS, desc="tasks"):
        prompts = load_safe_prompts(opi_ds, model, task, args.n, args.seed)
        print(f"  [{task}] {len(prompts)} safe prompts → caching …")
        _, resids = cache_resid(model, prompts,
                                batch_size=args.batch,
                                cache_layers=layers_to_cache)

        task_mean_vecs[task] = {}
        for l in layers_to_cache:
            mean_v = resids[l].float().mean(0)       # [d_model]
            task_mean_vecs[task][l] = mean_v.cpu()
            vecs_by_layer[l][task] = mean_v.cpu()

        sample_resids[task] = resids[args.layer].float().cpu()

    # ── Layer sweep ────────────────────────────────────────────────────────
    if args.all_layers:
        layers_sorted = sorted(layers_to_cache)
        disc_scores   = [between_task_mean_cosine_distance(vecs_by_layer[l])
                         for l in layers_sorted]
        best_layer    = layers_sorted[int(np.argmax(disc_scores))]
        print(f"\nBest discriminability layer: L={best_layer} "
              f"(mean pairwise cos-dist={max(disc_scores):.4f})")
        plot_layer_sweep(layers_sorted, disc_scores, best_layer, args.output_dir)
        plot_norms_per_layer(vecs_by_layer, TASKS, layers_sorted, args.output_dir)
    else:
        best_layer = args.layer

    # ── Depth analysis at best_layer ───────────────────────────────────────
    mean_vecs_at_layer = {t: task_mean_vecs[t][best_layer] for t in TASKS}
    X_means = torch.stack([mean_vecs_at_layer[t] for t in TASKS]).float()
    var_exp = pca_variance_explained(X_means, k=2)
    print(f"\nPCA @ L={best_layer}: top-2 PCs explain {var_exp:.1%} of variance")

    comps, pca_mean = plot_pca_means(
        mean_vecs_at_layer, best_layer, args.output_dir, var_exp
    )
    plot_pca_samples(
        sample_resids, mean_vecs_at_layer, best_layer, comps, pca_mean, args.output_dir
    )
    plot_cosine_heatmap(
        mean_vecs_at_layer,
        title=f"Task vector cosine similarity @ L={best_layer}",
        path=os.path.join(args.output_dir, f"exp8_cosine_L{best_layer}.png"),
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\nTask vector norms @ L={best_layer}:")
    for t in TASKS:
        v = mean_vecs_at_layer[t]
        print(f"  {t:>12}: ||v|| = {v.norm().item():.2f}")

    print(f"\nPairwise cosine similarities @ L={best_layer}:")
    for i, a in enumerate(TASKS):
        for j, b in enumerate(TASKS):
            if j <= i: continue
            cos = torch.nn.functional.cosine_similarity(
                mean_vecs_at_layer[a].unsqueeze(0),
                mean_vecs_at_layer[b].unsqueeze(0)
            ).item()
            print(f"  {a} ↔ {b}: {cos:+.3f}")

    # ── Save task vectors ──────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "task_vectors.pt")
    torch.save({
        "task_mean_vecs": task_mean_vecs,     # {task: {layer: [d_model]}}
        "best_layer":     best_layer,
        "tasks":          TASKS,
        "model":          args.model,
        "n":              args.n,
        "seed":           args.seed,
        "layers":         layers_to_cache,
    }, save_path)
    print(f"\nTask vectors saved → {save_path}")


if __name__ == "__main__":
    main()

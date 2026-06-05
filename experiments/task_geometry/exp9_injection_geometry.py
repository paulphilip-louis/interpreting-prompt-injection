"""
Experiment 9 — Injection direction geometry across primary tasks.

For each primary task T, we compute the combine−naive diff-of-means steering
vector (the injection-following direction) using a fixed injected task.
We then ask: what kind of transformation does the injection direction undergo
as the primary task changes?

Analysis
--------
  (a) Cosine similarity between injection_vecs across primary tasks — are they
      consistent, or task-specific?
  (b) Cross-matrix: cos(injection_vec[T_a], task_vec[T_b]) — is the injection
      direction aligned with the task representation?
  (c) Decomposition: for each task T, how much of injection_vec[T] lies along
      task_vec[T] vs orthogonal to it?
  (d) Joint PCA: fit PCA on task_vecs, project injection_vecs into that plane.
      Do injection directions live in the task subspace?
  (e) Biplot: task_vecs and injection_vecs as vectors from the origin in the
      shared PCA plane — shows the transformation geometry directly.
  (f) Layer sweep: how does cos(injection_vec[T], injection_vec[T_ref]) vary
      across layers? Does the relationship strengthen or weaken at specific layers?

Inputs
------
  Task vectors from exp8 (results/task_geometry/task_vectors.pt) if available,
  otherwise computed on the fly.

Usage
-----
    python experiments/task_geometry/exp9_injection_geometry.py \\
        --injection spam --layer 21 --n-train 100

    # Sweep all layers for the rotation-consistency plot:
    python experiments/task_geometry/exp9_injection_geometry.py \\
        --injection spam --all-layers
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
from src.utils.steering import cache_resid, diff_of_means
from src.utils.variables import DEVICE, MODEL_NAME
from experiments.task_geometry.exp8_task_geometry import (
    TASKS, TASK_COLORS, pca_fit, pca_project, load_safe_prompts,
    between_task_mean_cosine_distance,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def compute_injection_vecs(model, opi_ds, tasks, injection, layers, n_train, seed, batch):
    """
    For each primary task T (where T ≠ injection), compute the diff-of-means
    steering vector combine − naive at each requested layer.

    Returns {task: {layer: [d_model]}}
    """
    gen = torch.Generator().manual_seed(seed)
    inj_vecs = {}

    for task in tqdm(tasks, desc="injection vecs"):
        if task == injection:
            print(f"  Skipping {task} (same as injection)")
            continue

        prompts = opi.data_all_attack_types(
            opi_ds, model,
            task_type=task, injected_task=injection, include_clean=False,
        )
        n_avail = min(len(prompts["combine"]), len(prompts["naive"]))
        perm = torch.randperm(n_avail, generator=torch.Generator().manual_seed(seed))
        idx = perm[:n_train].tolist()

        combine_p = [prompts["combine"][i] for i in idx]
        naive_p   = [prompts["naive"][i]   for i in idx]

        _, res_c = cache_resid(model, combine_p, batch_size=batch, cache_layers=layers)
        _, res_n = cache_resid(model, naive_p,   batch_size=batch, cache_layers=layers)

        sv = diff_of_means(res_c, res_n, DEVICE)   # {layer: [d_model]}
        inj_vecs[task] = {l: sv[l].cpu() for l in layers}
        norms = {l: sv[l].norm().item() for l in layers}
        print(f"  [{task}] ||v|| @ L{layers[0]}={norms[layers[0]]:.2f}")

    return inj_vecs


def load_task_vecs(path, layer):
    """Load task_mean_vecs from exp8 output. Returns {task: tensor [d_model]}."""
    data = torch.load(path, map_location="cpu")
    return {t: data["task_mean_vecs"][t][layer] for t in data["tasks"]}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cosine_heatmap(mat, row_labels, col_labels, title, path,
                        vmin=-1, vmax=1, cmap="RdBu_r", fmt=".2f"):
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels)), max(3.5, len(row_labels) * 0.9)))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_decomposition_bars(inj_vecs, task_vecs, tasks, layer, injection, out_dir):
    """
    For each task T: decompose injection_vec[T] into:
      - component parallel to task_vec[T]  (|proj|)
      - component orthogonal to task_vec[T] (||v_perp||)
    Bar chart showing these two components per task.
    """
    parallel_norms = []
    perp_norms     = []
    total_norms    = []
    proj_fracs     = []

    for t in tasks:
        v  = inj_vecs[t][layer].float()
        u  = task_vecs[t].float()
        u_hat = u / (u.norm() + 1e-8)
        proj  = (v @ u_hat) * u_hat        # parallel component
        perp  = v - proj                   # orthogonal component
        parallel_norms.append(proj.norm().item())
        perp_norms.append(perp.norm().item())
        total_norms.append(v.norm().item())
        proj_fracs.append(proj.norm().item() / (v.norm().item() + 1e-8))

    x = np.arange(len(tasks))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    axes[0].bar(x - w/2, parallel_norms, w, label="‖parallel to task_vec‖",   color="steelblue")
    axes[0].bar(x + w/2, perp_norms,    w, label="‖orthogonal to task_vec‖",  color="tomato")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tasks, rotation=30, ha="right")
    axes[0].set(ylabel="L2 norm", title=f"Injection vec decomposition @ L={layer}\n"
                                         f"(injection={injection})")
    axes[0].legend(fontsize=8)

    # Fraction plot
    axes[1].bar(x, proj_fracs, color=[TASK_COLORS[t] for t in tasks])
    axes[1].set_xticks(x); axes[1].set_xticklabels(tasks, rotation=30, ha="right")
    axes[1].set(ylabel="fraction", ylim=(0, 1),
                title="Fraction of injection_vec parallel to task_vec")
    axes[1].axhline(0.5, color="gray", lw=0.8, ls="--")

    plt.suptitle(f"How much of the injection direction lies along the task direction?\n"
                 f"layer={layer}  injection={injection}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp9_decomp_L{layer}_{injection}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_joint_pca(task_vecs, inj_vecs, tasks, layer, injection, out_dir):
    """
    Fit PCA on task_vecs. Project both task_vecs and injection_vecs into this
    plane. Draws task_vecs as filled circles and injection_vecs as arrow tips
    from the origin, using the same colour per task.
    """
    tasks_with_inj = [t for t in tasks if t in inj_vecs]
    T_stack = torch.stack([task_vecs[t].float() for t in tasks_with_inj])
    comps, mean = pca_fit(T_stack, k=2)

    t_coords  = pca_project(T_stack, comps, mean)
    I_stack   = torch.stack([inj_vecs[t][layer].float() for t in tasks_with_inj])
    i_coords  = pca_project(I_stack, comps, mean)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, t in enumerate(tasks_with_inj):
        c = TASK_COLORS[t]
        # Task vector: filled circle
        ax.scatter(t_coords[idx, 0], t_coords[idx, 1],
                   color=c, s=200, zorder=5, marker="o", edgecolors="black", lw=0.8)
        ax.annotate(f"{t}\n(task)", (t_coords[idx, 0], t_coords[idx, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color=c)
        # Injection vector: cross/diamond
        ax.scatter(i_coords[idx, 0], i_coords[idx, 1],
                   color=c, s=200, zorder=5, marker="D", edgecolors="black", lw=0.8, alpha=0.7)
        ax.annotate(f"{t}\n(inj)", (i_coords[idx, 0], i_coords[idx, 1]),
                    textcoords="offset points", xytext=(5, -14), fontsize=8, color=c)
        # Line connecting them
        ax.plot([t_coords[idx, 0], i_coords[idx, 0]],
                [t_coords[idx, 1], i_coords[idx, 1]],
                color=c, lw=0.8, ls="--", alpha=0.5)

    ax.axhline(0, color="gray", lw=0.5, ls=":"); ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="PC 1 (task-vec space)", ylabel="PC 2 (task-vec space)",
           title=f"Joint PCA: ● task_vec  ◆ injection_vec\n"
                 f"PCA fitted on task_vecs  |  injection={injection}  L={layer}")
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0], [0], marker="o", ls="", color=TASK_COLORS[t], label=t, ms=9)
                 for t in tasks_with_inj]
    legend_el += [Line2D([0], [0], marker="o", ls="", color="gray", label="● task_vec", ms=9),
                  Line2D([0], [0], marker="D", ls="", color="gray", label="◆ inj_vec",  ms=9)]
    ax.legend(handles=legend_el, fontsize=8, loc="best")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp9_joint_pca_L{layer}_{injection}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_layer_sweep_consistency(inj_vecs_by_layer, tasks, layers, ref_task, injection, out_dir):
    """
    For each layer, plot cos(injection_vec[T], injection_vec[ref_task]).
    Shows whether the injection direction becomes more/less consistent across tasks
    at different depths.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    other_tasks = [t for t in tasks if t != ref_task and t in inj_vecs_by_layer]
    ref_vecs = inj_vecs_by_layer[ref_task]    # {layer: tensor}

    for t in other_tasks:
        cosines = []
        for l in layers:
            a = inj_vecs_by_layer[ref_task][l].float()
            b = inj_vecs_by_layer[t][l].float()
            cosines.append(torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item())
        ax.plot(layers, cosines, marker="o", ms=3, lw=1.5,
                color=TASK_COLORS[t], label=f"{t}")

    ax.set(xlabel="layer", ylabel=f"cos(inj_vec[T], inj_vec[{ref_task}])",
           title=f"Injection direction consistency across primary tasks\n"
                 f"(ref={ref_task}, injection={injection})")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp9_layer_sweep_consistency_{injection}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_biplot(task_vecs, inj_vecs, tasks, layer, injection, out_dir):
    """
    Biplot: both sets of vectors from the origin in a shared PCA space.
    Arrows show the direction and magnitude of each vector.
    Reveals whether injection_vec[T] is a rotation of task_vec[T] or
    points in a completely different direction.
    """
    tasks_with_inj = [t for t in tasks if t in inj_vecs]
    # Fit PCA on the union of all vectors
    all_vecs = torch.stack(
        [task_vecs[t].float() for t in tasks_with_inj] +
        [inj_vecs[t][layer].float() for t in tasks_with_inj]
    )
    comps, mean_all = pca_fit(all_vecs, k=2)

    # Project centered vectors (relative to origin, not mean-centered)
    origin = torch.zeros(all_vecs.shape[-1])
    o_proj = pca_project(origin.unsqueeze(0), comps, mean_all)[0]

    def proj(v):
        return pca_project(v.unsqueeze(0), comps, mean_all)[0] - o_proj

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter([0], [0], color="black", s=60, zorder=10)  # origin

    for t in tasks_with_inj:
        c = TASK_COLORS[t]
        tv = proj(task_vecs[t].float())
        iv = proj(inj_vecs[t][layer].float())
        ax.annotate("", xy=tv, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        ax.annotate("", xy=iv, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0, linestyle="dashed",
                                    alpha=0.65))
        ax.annotate(f"{t}(T)", tv, textcoords="offset points", xytext=(5, 3),
                    fontsize=9, color=c, fontweight="bold")
        ax.annotate(f"{t}(I)", iv, textcoords="offset points", xytext=(5, -12),
                    fontsize=9, color=c, alpha=0.75)

    ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set(xlabel="PC 1 (joint space)", ylabel="PC 2 (joint space)",
           title=f"Biplot: task_vec (solid) vs injection_vec (dashed)\n"
                 f"injection={injection}  L={layer}  (PCA fitted on all vectors)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp9_biplot_L{layer}_{injection}.png")
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
    p.add_argument("--injection",    default="spam", choices=opi.INJECTIONS,
                   help="Injected task to use for injection vectors.")
    p.add_argument("--layer",        type=int, default=21)
    p.add_argument("--all-layers",   action="store_true",
                   help="Sweep all layers for the consistency plot.")
    p.add_argument("--n-train",      type=int, default=100,
                   help="Prompts per task for steering vector computation.")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--batch",        type=int, default=4)
    p.add_argument("--task-vectors", default="results/task_geometry/task_vectors.pt",
                   help="Path to task_vectors.pt from exp8. If absent, computed on the fly.")
    p.add_argument("--output-dir",   default="results/task_geometry")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    n_layers = model.cfg.n_layers

    layers = list(range(n_layers)) if args.all_layers else [args.layer]

    # ── Task vectors ───────────────────────────────────────────────────────
    if os.path.exists(args.task_vectors):
        print(f"Loading task vectors from {args.task_vectors} …")
        task_vecs_at_layer = load_task_vecs(args.task_vectors, args.layer)
        # Also load for all layers if needed for sweep
        if args.all_layers:
            tv_data = torch.load(args.task_vectors, map_location="cpu")
            has_all_layers = (set(layers) <= set(tv_data["layers"]))
            if not has_all_layers:
                print("  task_vectors.pt doesn't have all layers — recomputing …")
                task_vecs_at_layer = None
        else:
            task_vecs_at_layer = task_vecs_at_layer
    else:
        print("task_vectors.pt not found — computing on the fly …")
        task_vecs_at_layer = None

    if task_vecs_at_layer is None:
        print("Computing task vectors …")
        opi_ds = opi.load_opi_dataset()
        raw_task_vecs = {}
        for task in tqdm(TASKS, desc="task vecs"):
            prompts = load_safe_prompts(opi_ds, model, task, args.n_train, args.seed)
            _, resids = cache_resid(model, prompts, batch_size=args.batch, cache_layers=layers)
            raw_task_vecs[task] = {l: resids[l].float().mean(0).cpu() for l in layers}
        task_vecs_at_layer = {t: raw_task_vecs[t][args.layer] for t in TASKS}

    # ── OPI dataset ────────────────────────────────────────────────────────
    print("Loading OPI dataset …")
    opi_ds = opi.load_opi_dataset()

    # ── Injection vectors ──────────────────────────────────────────────────
    print(f"\nComputing injection vectors (injection={args.injection}) …")
    # {task: {layer: [d_model]}}
    inj_vecs = compute_injection_vecs(
        model, opi_ds,
        tasks=TASKS,
        injection=args.injection,
        layers=layers,
        n_train=args.n_train,
        seed=args.seed,
        batch=args.batch,
    )
    valid_tasks = [t for t in TASKS if t in inj_vecs]

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"L={args.layer}  injection={args.injection}")
    print(f"{'='*60}")
    print("\nInjection vector norms:")
    for t in valid_tasks:
        print(f"  {t:>12}: {inj_vecs[t][args.layer].norm().item():.3f}")

    # Self-alignment: cos(inj_vec[T], task_vec[T])
    print("\nSelf-alignment cos(injection_vec[T], task_vec[T]):")
    for t in valid_tasks:
        iv = inj_vecs[t][args.layer].float()
        tv = task_vecs_at_layer[t].float()
        cos = torch.nn.functional.cosine_similarity(
            iv.unsqueeze(0), tv.unsqueeze(0)
        ).item()
        print(f"  {t:>12}: {cos:+.3f}")

    # ── Heatmap: injection_vec[a] vs injection_vec[b] ─────────────────────
    mat_ii = np.zeros((len(valid_tasks), len(valid_tasks)))
    for i, a in enumerate(valid_tasks):
        for j, b in enumerate(valid_tasks):
            mat_ii[i, j] = torch.nn.functional.cosine_similarity(
                inj_vecs[a][args.layer].float().unsqueeze(0),
                inj_vecs[b][args.layer].float().unsqueeze(0),
            ).item()
    plot_cosine_heatmap(
        mat_ii, valid_tasks, valid_tasks,
        title=f"Injection vector similarity across primary tasks\n"
              f"cos(inj_vec[row], inj_vec[col])  L={args.layer}  inj={args.injection}",
        path=os.path.join(args.output_dir,
                          f"exp9_inj_cosine_L{args.layer}_{args.injection}.png"),
    )

    # ── Heatmap: injection_vec[a] vs task_vec[b] ──────────────────────────
    task_list = [t for t in TASKS if t in task_vecs_at_layer]
    mat_it = np.zeros((len(valid_tasks), len(task_list)))
    for i, a in enumerate(valid_tasks):
        for j, b in enumerate(task_list):
            mat_it[i, j] = torch.nn.functional.cosine_similarity(
                inj_vecs[a][args.layer].float().unsqueeze(0),
                task_vecs_at_layer[b].float().unsqueeze(0),
            ).item()
    plot_cosine_heatmap(
        mat_it, valid_tasks, task_list,
        title=f"Cross: cos(inj_vec[row], task_vec[col])\n"
              f"L={args.layer}  injection={args.injection}",
        path=os.path.join(args.output_dir,
                          f"exp9_cross_cosine_L{args.layer}_{args.injection}.png"),
    )

    # ── Decomposition bars ────────────────────────────────────────────────
    plot_decomposition_bars(
        inj_vecs, task_vecs_at_layer, valid_tasks, args.layer, args.injection, args.output_dir
    )

    # ── Joint PCA ─────────────────────────────────────────────────────────
    plot_joint_pca(
        task_vecs_at_layer, inj_vecs, valid_tasks, args.layer, args.injection, args.output_dir
    )

    # ── Biplot (vectors from origin) ───────────────────────────────────────
    plot_biplot(
        task_vecs_at_layer, inj_vecs, valid_tasks, args.layer, args.injection, args.output_dir
    )

    # ── Layer sweep: consistency ───────────────────────────────────────────
    if args.all_layers:
        ref_task = valid_tasks[0]
        print(f"\nLayer sweep: injection consistency (ref={ref_task}) …")
        # Need inj_vecs indexed by task then layer
        inj_by_task = {t: inj_vecs[t] for t in valid_tasks}
        plot_layer_sweep_consistency(
            inj_by_task, valid_tasks, sorted(layers), ref_task, args.injection, args.output_dir
        )

    # ── Save injection vectors ─────────────────────────────────────────────
    save_path = os.path.join(args.output_dir,
                             f"injection_vectors_{args.injection}_L{args.layer}.pt")
    torch.save({
        "inj_vecs":  {t: {l: inj_vecs[t][l] for l in layers} for t in valid_tasks},
        "layer":     args.layer,
        "injection": args.injection,
        "tasks":     valid_tasks,
        "model":     args.model,
        "n_train":   args.n_train,
        "seed":      args.seed,
    }, save_path)
    print(f"\nInjection vectors saved → {save_path}")


if __name__ == "__main__":
    main()

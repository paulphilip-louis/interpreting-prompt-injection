"""
Experiment 11 — Task manifold and v_combine geometry.

For each primary task T (with fixed injected_task), we extract three families
of residuals at a given layer:

  (a) instruction-end  — residual at the LAST TOKEN of the instruction span,
                         before the model sees any user content.  Pure task identity.
  (b) last-token safe  — residual at the final token of uninjected prompts.
  (c) last-token naive — last token of naive (no-trigger) attack prompts.
  (d) last-token combine — last token of combine (hard-trigger) attack prompts.

v_combine[T] = mean(d) − mean(c)  is the injection-following direction for task T.

Hypothesis (task manifold): the 5 primary tasks define a low-dimensional manifold
in residual space.  v_combine "slides" along this manifold — its component tangent
to the manifold changes systematically with the task, while its normal component
may be more consistent.

Visualisations
--------------
  3D PCA  — instruction-end residuals (task clusters in the pure-instruction space)
  3D PCA  — last-token residuals, all four conditions, colored by task / shaped by cond
  3D PCA  — centroids per (task, condition) with v_combine as 3D arrows
  t-SNE   — all residuals pooled, colored by task, marker by condition
  Tangent — project v_combine[T] onto the 1st and 2nd PCs of the task-centroid manifold;
            compare the tangent and normal components across tasks

Usage
-----
    python experiments/task_geometry/exp11_task_manifold.py \\
        --injection spam --layer 21 --n 50 --batch 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sklearn.manifold import TSNE

import src.data.opi as opi
from src.utils.steering import cache_resid, get_instruction_span
from src.utils.variables import DEVICE, MODEL_NAME
from experiments.task_geometry.exp8_task_geometry import TASKS, TASK_COLORS, pca_fit, pca_project


# ── colour / marker scheme ────────────────────────────────────────────────────
COND_MARKER = {
    "instend":  ("^", "solid",  0.90),   # triangle-up, opaque
    "safe":     ("o", "solid",  0.55),   # circle
    "naive":    ("s", "dashed", 0.55),   # square
    "combine":  ("D", "solid",  0.80),   # diamond
}
COND_LABEL = {
    "instend": "instr-end",
    "safe":    "safe (last tok)",
    "naive":   "naive (last tok)",
    "combine": "combine (last tok)",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_instruction_text(opi_ds, task: str) -> str:
    """Return the full instruction string exactly as it appears in formatted prompts."""
    row = next(r for r in opi_ds if r["task_type"] == task)
    old, new = opi.FORMAT[task]
    return row["instruction"].replace(old, new)


@torch.no_grad()
def cache_instend(model, prompts: list, instruction_text: str, layer: int) -> torch.Tensor:
    """
    Extract residual at the instruction-end token for each prompt.
    Processes one at a time (no padding artefacts).
    Returns [N, d_model].
    """
    device  = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    resids  = []

    # Find instruction span from the first prompt (same instruction for all)
    inst_start, inst_end = get_instruction_span(model, prompts[0], instruction_text)
    pos = inst_end - 1   # last token of instruction span (0-indexed, no padding)

    for prompt in tqdm(prompts, desc=f"  instend L{layer}", leave=False):
        enc = model.tokenizer(prompt, return_tensors="pt").to(device)
        _, cache = model.run_with_cache(
            enc.input_ids,
            names_filter=lambda n: n == hook_name,
        )
        resids.append(cache[hook_name][0, pos, :].cpu())
        del cache

    return torch.stack(resids)   # [N, d_model]


def cache_lasttoken(model, prompts: list, layer: int, batch: int) -> torch.Tensor:
    """Extract last-token residuals in batches. Returns [N, d_model]."""
    _, resids = cache_resid(model, prompts, batch_size=batch, cache_layers=[layer])
    return resids[layer].float().cpu()


def pca3(X: torch.Tensor):
    """Fit PCA and return 3-D projection. X: [N, d] → ([N, 3], comps[3,d], mean[d])."""
    comps, mean = pca_fit(X, k=3)
    coords = pca_project(X, comps, mean)   # [N, 3]
    return coords, comps, mean


def tsne2(X: np.ndarray, perplexity: float = 30.0, seed: int = 0) -> np.ndarray:
    """sklearn t-SNE to 2D."""
    return TSNE(n_components=2, perplexity=min(perplexity, X.shape[0] - 1),
                random_state=seed, init="pca").fit_transform(X)


# ── plotting ──────────────────────────────────────────────────────────────────

def scatter3(ax, coords, color, marker, alpha, label, size=22):
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=color, marker=marker, alpha=alpha, s=size, label=label,
               edgecolors="none")


def plot_3d_by_task(coords_dict: dict, title: str, path: str, elev: int = 25, azim: int = 45):
    """
    coords_dict: {task: np.ndarray [N, 3]}
    One colour per task, all samples as points.
    """
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for task, coords in coords_dict.items():
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=TASK_COLORS[task], label=task, s=18, alpha=0.55, edgecolors="none")
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_3d_conditions(all_coords, all_tasks, all_conds, title, path,
                       centroids=None, v_arrows=None, elev=25, azim=45):
    """
    Scatter of all residuals colored by task, marker by condition.
    Optionally overlay task centroids and v_combine arrows.
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Draw samples
    for cond, (marker, _, alpha) in COND_MARKER.items():
        mask = np.array(all_conds) == cond
        if not mask.any():
            continue
        c_arr = np.array(all_coords)[mask]
        t_arr = np.array(all_tasks)[mask]
        colors = [TASK_COLORS[t] for t in t_arr]
        ax.scatter(c_arr[:, 0], c_arr[:, 1], c_arr[:, 2],
                   c=colors, marker=marker, s=18, alpha=alpha, edgecolors="none")

    # Draw centroids as large stars
    if centroids is not None:
        for (task, cond), c3 in centroids.items():
            ax.scatter(*c3, c=TASK_COLORS[task], s=180, marker="*",
                       edgecolors="black", lw=0.5, zorder=10)

    # Draw v_combine arrows (naive → combine centroid)
    if v_arrows is not None:
        for task, (start3, end3) in v_arrows.items():
            ax.quiver(*start3, *(end3 - start3),
                      color=TASK_COLORS[task], lw=2.0, arrow_length_ratio=0.25)
            ax.text(*(end3 + 0.02), task, fontsize=7, color=TASK_COLORS[task])

    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=title)
    ax.view_init(elev=elev, azim=azim)

    # Legend: task by colour, condition by marker
    from matplotlib.lines import Line2D
    leg_task = [Line2D([0],[0], marker="o", ls="", color=TASK_COLORS[t], label=t, ms=7)
                for t in TASKS if t in dict(zip(all_tasks, all_tasks))]
    leg_cond = [Line2D([0],[0], marker=m, ls="", color="gray",
                       label=COND_LABEL[c], ms=7, alpha=al)
                for c, (m, _, al) in COND_MARKER.items()]
    ax.legend(handles=leg_task + leg_cond, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_tsne(all_coords2d, all_tasks, all_conds, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))

    for cond, (marker, ls, alpha) in COND_MARKER.items():
        mask = np.array(all_conds) == cond
        if not mask.any():
            continue
        c_arr  = all_coords2d[mask]
        t_arr  = np.array(all_tasks)[mask]
        colors = [TASK_COLORS[t] for t in t_arr]
        ax.scatter(c_arr[:, 0], c_arr[:, 1],
                   c=colors, marker=marker, s=28, alpha=alpha,
                   edgecolors="none", label=COND_LABEL[cond])

    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
    from matplotlib.lines import Line2D
    leg_task = [Line2D([0],[0], marker="o", ls="", color=TASK_COLORS[t], label=t, ms=8)
                for t in TASKS]
    leg_cond = [Line2D([0],[0], marker=m, ls="", color="gray",
                       label=COND_LABEL[c], ms=8)
                for c, (m, _, _) in COND_MARKER.items()]
    ax.legend(handles=leg_task + leg_cond, fontsize=8, loc="best",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_manifold_tangent(tasks, tang_comps, norm_comps, inj_norms, injection, layer, out_dir):
    """
    For each task: show how much of v_combine lies tangent vs normal to the
    task-centroid manifold (spanned by first 2 PCs of safe centroids).
    """
    x   = np.arange(len(tasks))
    w   = 0.30
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = [TASK_COLORS[t] for t in tasks]

    # Stacked bar: tangent vs normal magnitudes
    axes[0].bar(x - w/2, tang_comps, w, label="tangent to manifold",  color="steelblue")
    axes[0].bar(x + w/2, norm_comps, w, label="normal to manifold",   color="tomato")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tasks, rotation=25, ha="right")
    axes[0].set(ylabel="L2 norm",
                title=f"v_combine decomposition on task manifold\n"
                      f"injection={injection}  L={layer}")
    axes[0].legend(fontsize=9)

    # Fraction tangent
    fracs = np.array(tang_comps) / (np.array(tang_comps) + np.array(norm_comps) + 1e-8)
    axes[1].bar(x, fracs, color=colors, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(tasks, rotation=25, ha="right")
    axes[1].set(ylabel="fraction tangent", ylim=(0, 1),
                title="Fraction of v_combine tangent to task manifold")
    axes[1].axhline(0.5, color="gray", lw=0.8, ls="--")
    for i, (frac, task) in enumerate(zip(fracs, tasks)):
        axes[1].text(i, frac + 0.02, f"{frac:.2f}", ha="center", fontsize=9)

    plt.suptitle(f"Does v_combine slide along the task manifold?\n"
                 f"(tangent = parallel to the manifold spanned by task safe-centroids)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp11_manifold_tangent_{injection}_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


def plot_2d_manifold_arrows(safe_centroids, naive_centroids, combine_centroids,
                             tasks, injection, layer, out_dir):
    """
    Project safe centroids to 2D (their own PCA plane).
    Show naive and combine centroids in the same plane.
    Draw v_combine as arrows from naive → combine centroid.
    """
    # Fit PCA on safe centroids
    X_safe = torch.stack([safe_centroids[t].float() for t in tasks])
    comps, mean = pca_fit(X_safe, k=2)

    def proj(vecs_dict):
        X = torch.stack([vecs_dict[t].float() for t in tasks])
        return pca_project(X, comps, mean)   # [5, 2]

    sc = proj(safe_centroids)
    nc = proj(naive_centroids)
    cc = proj(combine_centroids)

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, t in enumerate(tasks):
        c = TASK_COLORS[t]
        # safe centroid: filled circle
        ax.scatter(sc[i, 0], sc[i, 1], color=c, s=200, marker="o",
                   edgecolors="black", lw=0.8, zorder=5)
        # naive centroid: open square
        ax.scatter(nc[i, 0], nc[i, 1], color=c, s=160, marker="s",
                   edgecolors=c, facecolors="none", lw=1.5, zorder=5)
        # combine centroid: filled diamond
        ax.scatter(cc[i, 0], cc[i, 1], color=c, s=160, marker="D",
                   edgecolors="black", lw=0.5, alpha=0.8, zorder=5)
        # v_combine arrow: naive → combine
        dx, dy = cc[i, 0] - nc[i, 0], cc[i, 1] - nc[i, 1]
        ax.annotate("", xy=(cc[i, 0], cc[i, 1]), xytext=(nc[i, 0], nc[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        ax.annotate(t, (sc[i, 0], sc[i, 1]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9, color=c)

    ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set(xlabel="PC1 of safe centroids", ylabel="PC2 of safe centroids",
           title=f"Task manifold (safe ●) with v_combine arrows (□ naive → ◆ combine)\n"
                 f"injection={injection}  L={layer}")
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker="o", ls="", color="gray", ms=9, label="● safe centroid"),
           Line2D([0],[0], marker="s", ls="", color="gray", ms=9,
                  fillstyle="none", label="□ naive centroid"),
           Line2D([0],[0], marker="D", ls="", color="gray", ms=9, alpha=0.8,
                  label="◆ combine centroid")]
    ax.legend(handles=leg, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"exp11_manifold_arrows_{injection}_L{layer}.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model",      default=MODEL_NAME)
    p.add_argument("--injection",  default="spam", choices=opi.INJECTIONS)
    p.add_argument("--layer",      type=int, default=21)
    p.add_argument("--n",          type=int, default=50,
                   help="Prompts per task per condition.")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--batch",      type=int, default=4)
    p.add_argument("--output-dir", default="results/task_geometry")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …")
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    print("Loading OPI dataset …")
    opi_ds = opi.load_opi_dataset()

    valid_tasks = [t for t in TASKS if t != args.injection]

    # ── Collect residuals ──────────────────────────────────────────────────
    # {task: {cond: tensor [N, d_model]}}
    resid = {t: {} for t in valid_tasks}

    for task in valid_tasks:
        print(f"\n[{task}]")
        inst_text = get_instruction_text(opi_ds, task)
        prompts   = opi.data_all_attack_types(
            opi_ds, model,
            task_type=task, injected_task=args.injection, include_clean=True,
        )
        gen = torch.Generator().manual_seed(args.seed)
        n   = min(args.n, len(prompts["naive"]))
        idx = torch.randperm(n, generator=gen)[:n].tolist()

        safe_p    = [prompts["safe"][i]    for i in idx]
        naive_p   = [prompts["naive"][i]   for i in idx]
        combine_p = [prompts["combine"][i] for i in idx]

        print(f"  instruction-end residuals …")
        resid[task]["instend"]  = cache_instend(model, safe_p, inst_text, args.layer)

        print(f"  last-token residuals (safe / naive / combine) …")
        resid[task]["safe"]    = cache_lasttoken(model, safe_p,    args.layer, args.batch)
        resid[task]["naive"]   = cache_lasttoken(model, naive_p,   args.layer, args.batch)
        resid[task]["combine"] = cache_lasttoken(model, combine_p, args.layer, args.batch)

    # ── Centroids and v_combine ────────────────────────────────────────────
    safe_cen    = {t: resid[t]["safe"].mean(0)    for t in valid_tasks}
    naive_cen   = {t: resid[t]["naive"].mean(0)   for t in valid_tasks}
    combine_cen = {t: resid[t]["combine"].mean(0) for t in valid_tasks}
    v_combine   = {t: combine_cen[t] - naive_cen[t] for t in valid_tasks}

    print(f"\nv_combine norms @ L={args.layer}:")
    for t in valid_tasks:
        print(f"  {t:>12}: {v_combine[t].norm().item():.3f}")

    # ── 3D PCA — instruction-end only ─────────────────────────────────────
    X_instend = torch.cat([resid[t]["instend"].float() for t in valid_tasks])
    labels_t  = sum([[t] * len(resid[t]["instend"]) for t in valid_tasks], [])
    coords3_ie, comps_ie, mean_ie = pca3(X_instend)

    plot_3d_by_task(
        {t: coords3_ie[np.array(labels_t) == t] for t in valid_tasks},
        title=f"Instruction-end residuals @ L={args.layer}\n(colored by primary task)",
        path=os.path.join(args.output_dir,
                          f"exp11_3dpca_instend_{args.injection}_L{args.layer}.png"),
    )

    # ── 3D PCA — all conditions, joint ────────────────────────────────────
    # Fit PCA on safe residuals, project all conditions into same space
    X_safe_all = torch.cat([resid[t]["safe"].float() for t in valid_tasks])
    _, comps_safe, mean_safe = pca3(X_safe_all)

    all_coords3 = []
    all_tasks_l = []
    all_conds_l = []
    for cond in ("instend", "safe", "naive", "combine"):
        for t in valid_tasks:
            X = resid[t][cond].float()
            c3 = pca_project(X, comps_safe, mean_safe)
            all_coords3.extend(c3.tolist())
            all_tasks_l.extend([t] * len(X))
            all_conds_l.extend([cond] * len(X))

    all_coords3 = np.array(all_coords3)

    # Centroids in PCA space + v_combine arrows
    centroids_3d = {}
    for t in valid_tasks:
        for cond in ("safe", "naive", "combine"):
            c = torch.tensor(
                pca_project(resid[t][cond].float().mean(0).unsqueeze(0),
                            comps_safe, mean_safe)[0]
            )
            centroids_3d[(t, cond)] = c.numpy()

    v_arrows_3d = {
        t: (centroids_3d[(t, "naive")],
            centroids_3d[(t, "combine")])
        for t in valid_tasks
    }

    plot_3d_conditions(
        all_coords3, all_tasks_l, all_conds_l,
        title=f"Last-token residuals @ L={args.layer}\n"
              f"color=task, marker=condition, ★=centroid, arrow=v_combine",
        path=os.path.join(args.output_dir,
                          f"exp11_3dpca_all_{args.injection}_L{args.layer}.png"),
        centroids=centroids_3d,
        v_arrows=v_arrows_3d,
    )

    # ── t-SNE — all residuals pooled ──────────────────────────────────────
    print("\nRunning t-SNE …")
    X_all   = torch.cat([resid[t][cond].float()
                         for t in valid_tasks
                         for cond in ("instend", "safe", "naive", "combine")]).numpy()
    labels_task = [t    for t in valid_tasks for cond in ("instend","safe","naive","combine")
                       for _ in range(len(resid[t][cond]))]
    labels_cond = [cond for t in valid_tasks for cond in ("instend","safe","naive","combine")
                       for _ in range(len(resid[t][cond]))]

    perp  = min(30.0, X_all.shape[0] // 4)
    tsne2d = tsne2(X_all, perplexity=perp)

    plot_tsne(
        tsne2d, labels_task, labels_cond,
        title=f"t-SNE of all residuals @ L={args.layer}\n"
              f"color=task, marker=condition  (injection={args.injection})",
        path=os.path.join(args.output_dir,
                          f"exp11_tsne_{args.injection}_L{args.layer}.png"),
    )

    # ── v_combine in proper reference frame ───────────────────────────────
    # v_combine[T] is a difference vector; comparing it to absolute residuals
    # is meaningless.  Instead we center everything per-task relative to the
    # naive baseline: x_rel = resid[T][cond][i] − naive_centroid[T].
    # In this relative frame:
    #   naive samples ≈ 0 (by construction)
    #   combine samples are displaced by v_combine
    #   safe samples show the clean-task deviation from naive
    #   v_combine[T] centroid sits exactly at combine_centroid − naive_centroid

    # ── (a) Centroid-only PCA (all 8 centroids: naive + combine per task) ──
    cen_vecs  = ([naive_cen[t]   for t in valid_tasks] +
                 [combine_cen[t] for t in valid_tasks])
    cen_tasks = list(valid_tasks) + list(valid_tasks)
    cen_conds = ["naive"] * len(valid_tasks) + ["combine"] * len(valid_tasks)

    X_cen = torch.stack(cen_vecs).float()
    comps_cen, mean_cen = pca_fit(X_cen, k=2)
    coords_cen = pca_project(X_cen, comps_cen, mean_cen)   # [8, 2]

    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(valid_tasks)
    for i, t in enumerate(valid_tasks):
        c = TASK_COLORS[t]
        nc = coords_cen[i]          # naive centroid
        cc = coords_cen[i + n]      # combine centroid
        ax.scatter(*nc, color=c, marker="s", s=160, zorder=5,
                   edgecolors="black", lw=0.6)
        ax.scatter(*cc, color=c, marker="D", s=160, zorder=5,
                   edgecolors="black", lw=0.6, alpha=0.85)
        ax.annotate("", xy=cc, xytext=nc,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        ax.annotate(t, nc, textcoords="offset points",
                    xytext=(-22, 5), fontsize=9, color=c)
    ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set(xlabel="PC1", ylabel="PC2",
           title=f"Task manifold — centroids only\n"
                 f"□ naive  ◆ combine  arrow = v_combine[task]\n"
                 f"injection={args.injection}  L={args.layer}")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker="s", ls="", color="gray",
                               ms=9, label="□ naive centroid"),
                        Line2D([0],[0], marker="D", ls="", color="gray",
                               ms=9, alpha=0.85, label="◆ combine centroid")] +
                       [Line2D([0],[0], marker="o", ls="", color=TASK_COLORS[t],
                               ms=8, label=t) for t in valid_tasks],
              fontsize=8)
    plt.tight_layout()
    cen_path = os.path.join(args.output_dir,
                            f"exp11_centroid_pca_{args.injection}_L{args.layer}.png")
    plt.savefig(cen_path, dpi=120); plt.close()
    print(f"  Saved → {cen_path}")

    # ── (b) Relative-to-naive PCA/t-SNE ───────────────────────────────────
    # Center each task's samples by subtracting that task's naive centroid.
    X_rel, rel_tasks, rel_conds = [], [], []
    for t in valid_tasks:
        nc = naive_cen[t].float()
        for cond in ("safe", "naive", "combine"):
            X_c = resid[t][cond].float() - nc   # [N, d]
            X_rel.append(X_c)
            rel_tasks.extend([t] * len(X_c))
            rel_conds.extend([cond] * len(X_c))
    X_rel = torch.cat(X_rel)   # [N_total, d]

    # v_combine centroids in relative space = combine_centroid − naive_centroid
    vc_rel = torch.stack([v_combine[t].float() for t in valid_tasks])  # [4, d]

    # Fit PCA on relative residuals, project everything
    comps_rel, mean_rel = pca_fit(X_rel, k=2)
    coords_rel  = pca_project(X_rel,    comps_rel, mean_rel)
    coords_vc_r = pca_project(vc_rel,   comps_rel, mean_rel)

    rel_tasks_arr = np.array(rel_tasks)
    rel_conds_arr = np.array(rel_conds)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap_cond = {"safe": ("o", 0.28), "naive": ("s", 0.22), "combine": ("D", 0.38)}
    for cond, (marker, alpha) in cmap_cond.items():
        mask   = rel_conds_arr == cond
        colors = [TASK_COLORS[t] for t in rel_tasks_arr[mask]]
        ax.scatter(coords_rel[mask, 0], coords_rel[mask, 1],
                   c=colors, marker=marker, s=18, alpha=alpha, edgecolors="none")
    for i, t in enumerate(valid_tasks):
        ax.scatter(coords_vc_r[i, 0], coords_vc_r[i, 1],
                   c=TASK_COLORS[t], marker="p", s=280, zorder=10,
                   edgecolors="black", lw=0.8)
        ax.annotate(t, (coords_vc_r[i, 0], coords_vc_r[i, 1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=TASK_COLORS[t], fontweight="bold")
    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set(xlabel="PC1 (relative space)", ylabel="PC2 (relative space)",
           title=f"Relative-to-naive PCA  (⬠ = v_combine[task])\n"
                 f"each task centered at its naive baseline → origin\n"
                 f"injection={args.injection}  L={args.layer}")
    leg_t = [Line2D([0],[0], marker="o", ls="", color=TASK_COLORS[t], label=t, ms=8)
             for t in valid_tasks]
    leg_c = [Line2D([0],[0], marker=m, ls="", color="gray", label=c, ms=7, alpha=a)
             for c, (m, a) in cmap_cond.items()]
    leg_c += [Line2D([0],[0], marker="p", ls="", color="gray",
                     label="v_combine centroid", ms=10, markeredgecolor="black")]
    ax.legend(handles=leg_t + leg_c, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    rel_pca_path = os.path.join(args.output_dir,
                                f"exp11_relative_pca_{args.injection}_L{args.layer}.png")
    plt.savefig(rel_pca_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {rel_pca_path}")

    # t-SNE in relative space (include v_combine centroids in the pool)
    print("Running t-SNE in relative space …")
    X_tsne_pool = torch.cat([X_rel, vc_rel])
    tsne_tasks  = rel_tasks + list(valid_tasks)
    tsne_conds  = rel_conds + ["vcombine"] * len(valid_tasks)

    perp_r  = min(30.0, X_tsne_pool.shape[0] // 4)
    coords_tsne_r = tsne2(X_tsne_pool.numpy(), perplexity=perp_r)

    n_rel = len(X_rel)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cond, (marker, alpha) in cmap_cond.items():
        mask   = np.array(tsne_conds[:n_rel]) == cond
        colors = [TASK_COLORS[t] for t in np.array(tsne_tasks[:n_rel])[mask]]
        ax.scatter(coords_tsne_r[:n_rel][mask, 0], coords_tsne_r[:n_rel][mask, 1],
                   c=colors, marker=marker, s=20, alpha=alpha, edgecolors="none")
    for i, t in enumerate(valid_tasks):
        ax.scatter(coords_tsne_r[n_rel + i, 0], coords_tsne_r[n_rel + i, 1],
                   c=TASK_COLORS[t], marker="p", s=290, zorder=10,
                   edgecolors="black", lw=0.8)
        ax.annotate(t, coords_tsne_r[n_rel + i],
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=TASK_COLORS[t], fontweight="bold")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title=f"t-SNE in relative-to-naive space  (⬠ = v_combine)\n"
                 f"injection={args.injection}  L={args.layer}")
    ax.legend(handles=leg_t + leg_c, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    rel_tsne_path = os.path.join(args.output_dir,
                                 f"exp11_relative_tsne_{args.injection}_L{args.layer}.png")
    plt.savefig(rel_tsne_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {rel_tsne_path}")

    # ── 2D manifold arrows plot ────────────────────────────────────────────
    plot_2d_manifold_arrows(
        safe_cen, naive_cen, combine_cen,
        valid_tasks, args.injection, args.layer, args.output_dir,
    )

    # ── Tangent / normal decomposition ────────────────────────────────────
    # Fit PCA on safe centroids → manifold basis
    X_scen = torch.stack([safe_cen[t].float() for t in valid_tasks])
    comps_man, mean_man = pca_fit(X_scen, k=2)   # [2, d_model]

    tang_comps = []
    norm_comps = []
    for t in valid_tasks:
        v = v_combine[t].float()
        # Project v onto manifold tangent plane (spanned by comps_man)
        proj_tan = (comps_man @ v).unsqueeze(1) * comps_man   # [2, d] sum
        tang_vec = proj_tan.sum(0)                             # [d]
        norm_vec = v - tang_vec
        tang_comps.append(tang_vec.norm().item())
        norm_comps.append(norm_vec.norm().item())

    print(f"\nv_combine tangent/normal decomposition (manifold = top-2 PCs of safe centroids):")
    print(f"  {'task':>12}  {'tangent':>8}  {'normal':>8}  {'frac_tan':>9}")
    for t, tan, nor in zip(valid_tasks, tang_comps, norm_comps):
        frac = tan / (tan + nor + 1e-8)
        print(f"  {t:>12}  {tan:>8.3f}  {nor:>8.3f}  {frac:>9.3f}")

    plot_manifold_tangent(
        valid_tasks, tang_comps, norm_comps,
        [v_combine[t].norm().item() for t in valid_tasks],
        args.injection, args.layer, args.output_dir,
    )

    # ── Print pairwise cosines of v_combine ───────────────────────────────
    print(f"\nPairwise cosine of v_combine across tasks:")
    for i, a in enumerate(valid_tasks):
        for j, b in enumerate(valid_tasks):
            if j <= i: continue
            cos = torch.nn.functional.cosine_similarity(
                v_combine[a].float().unsqueeze(0),
                v_combine[b].float().unsqueeze(0)
            ).item()
            print(f"  {a} ↔ {b}: {cos:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

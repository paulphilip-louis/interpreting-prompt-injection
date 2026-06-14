"""
Experiment 4 — Comparison with optimised (neural-exec) triggers.

Figures produced:
  fig10 — ASR baselines: naive / combine / neural_exec / random (with CI)
  fig11 — Per-layer norm: v_combine vs v_neural_exec
  fig12 — Cosine similarity v_combine vs v_neural_exec across layers, per task
  fig13 — PCA of per-task {combine, neural_exec} vectors at peak layer
  fig14 — Steering with v_neural_exec vs v_combine: coef sweep at peak layer
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from src.utils.attention_tracker import load_model
from src.data.opi import load_opi_per_task, INJECTIONS
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook, save_results,
)
from src.utils.utils import cosine_similarity
from paper_exp.style import apply as apply_style, savefig, COLORS, TASK_LABELS

apply_style()

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TASK = "sentiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 75
N_TEST = 75
PEAK_LAYER = 21
N_LAYERS = 28
COEFS = np.arange(-1, 5, 0.5).tolist()

RESULTS_DIR = "results/exp4"
CACHE_DIR = "results/cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_or_cache_resids(name, model, prompt_list, layers):
    path = os.path.join(CACHE_DIR, f"{name}.pt")
    if os.path.exists(path):
        print(f"  Loading cached {name}")
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"  Computing {name}...")
    _, resids = cache_resid(model, prompt_list, BATCH, cache_layers=layers)
    torch.save(resids, path)
    return resids


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    n = len(values)
    means = np.array([np.random.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def per_example_asr(logits, cor_ids, inj_ids):
    p = logits.softmax(-1)
    p_inj = p[:, inj_ids].sum(-1)
    p_cor = p[:, cor_ids].sum(-1)
    return ((p_inj - p_cor) + 1) / 2


# ── Load model & data ────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_NAME)

print("Loading data...")
prompts = load_opi_per_task(model, TASK)

all_layers = list(range(N_LAYERS))

# Cache train-split residuals for vector extraction
print("Caching train residuals...")
conditions = ["naive", "combine", "neural_exec", "random"]
task_resids = {}
for inj in INJECTIONS:
    task_resids[inj] = {}
    for cond in conditions:
        train_p = prompts[inj]["prompts"][cond][:N_TRAIN]
        task_resids[inj][cond] = load_or_cache_resids(f"{inj}_{cond}_train", model, train_p, all_layers)

# Steering vectors (from train split)
v_combine = {}
v_neural_exec = {}
for inj in INJECTIONS:
    v_combine[inj] = diff_of_means(task_resids[inj]["combine"], task_resids[inj]["naive"], DEVICE)
    v_neural_exec[inj] = diff_of_means(task_resids[inj]["neural_exec"], task_resids[inj]["naive"], DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — ASR baselines: naive / combine / neural_exec / random (with CI)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 10: ASR baselines per task ──")
baseline_data = {}
for inj in INJECTIONS:
    baseline_data[inj] = {}
    for cond in conditions:
        p = prompts[inj]["prompts"][cond][N_TRAIN:N_TRAIN + N_TEST]
        logits, _ = cache_resid(model, p, BATCH)
        per_ex = per_example_asr(logits, prompts[inj]["cor_ids"], prompts[inj]["inj_ids"]).numpy()
        mean, ci_lo, ci_hi = bootstrap_ci(per_ex)
        baseline_data[inj][cond] = {"asr": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}

fig, axes = plt.subplots(1, len(INJECTIONS), figsize=(3.5 * len(INJECTIONS), 3.5), sharey=True)
cond_colors = {"naive": COLORS["naive"], "combine": COLORS["combine"],
               "neural_exec": COLORS["neural_exec"], "random": COLORS["random_ctrl"]}
for ax, inj in zip(axes, INJECTIONS):
    x = range(len(conditions))
    means = [baseline_data[inj][c]["asr"] for c in conditions]
    ci_lo = [baseline_data[inj][c]["ci_lo"] for c in conditions]
    ci_hi = [baseline_data[inj][c]["ci_hi"] for c in conditions]
    yerr = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, ci_lo, ci_hi)]).T
    ax.bar(x, means, yerr=yerr, capsize=4,
           color=[cond_colors[c] for c in conditions], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    ax.set_title(TASK_LABELS[inj])
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("ASR")
fig.suptitle("Baseline ASR by trigger type", y=1.02)
savefig(fig, "fig10_neural_exec_baselines")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Per-layer norm: v_combine vs v_neural_exec
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 11: per-layer norms ──")
fig, axes = plt.subplots(1, len(INJECTIONS), figsize=(3.5 * len(INJECTIONS), 3), sharey=True)
for ax, inj in zip(axes, INJECTIONS):
    norms_c = [v_combine[inj][l].norm().item() for l in all_layers]
    norms_n = [v_neural_exec[inj][l].norm().item() for l in all_layers]
    ax.plot(all_layers, norms_c, label="combine", color=COLORS["combine"])
    ax.plot(all_layers, norms_n, label="neural_exec", color=COLORS["neural_exec"])
    ax.set(xlabel="Layer", title=TASK_LABELS[inj])
    ax.legend(fontsize=7)
axes[0].set_ylabel("||v||")
fig.suptitle("Steering vector norms across layers", y=1.02)
savefig(fig, "fig11_norms")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Cosine similarity v_combine vs v_neural_exec across layers
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 12: cosine similarity across layers ──")
fig, ax = plt.subplots(figsize=(7, 4))
for inj in INJECTIONS:
    cos_vals = [cosine_similarity(v_combine[inj][l], v_neural_exec[inj][l]) for l in all_layers]
    ax.plot(all_layers, cos_vals, color=COLORS[inj], label=TASK_LABELS[inj])
ax.axhline(0, ls=":", color="gray", lw=0.8)
ax.set(xlabel="Layer", ylabel="Cosine similarity",
       title="v_combine vs v_neural_exec alignment")
ax.legend()
savefig(fig, "fig12_cosine_combine_vs_neural_exec")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 — PCA of per-task {combine, neural_exec} vectors at peak layer
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 13: PCA of combine + neural_exec vectors ──")
labels, vecs = [], []
for inj in INJECTIONS:
    for cond, v_dict in [("combine", v_combine), ("neural_exec", v_neural_exec)]:
        labels.append((inj, cond))
        vecs.append(v_dict[inj][PEAK_LAYER].cpu().numpy())

X = np.stack(vecs)
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(6, 5))
marker_map = {"combine": "o", "neural_exec": "s"}
for i, (inj, cond) in enumerate(labels):
    ax.scatter(coords[i, 0], coords[i, 1], c=COLORS[inj],
               marker=marker_map[cond], s=80, zorder=5, edgecolors="black", linewidths=0.5)

# Draw arrows from combine to neural_exec for each task
for inj in INJECTIONS:
    idx_c = labels.index((inj, "combine"))
    idx_n = labels.index((inj, "neural_exec"))
    ax.annotate("", xy=coords[idx_n], xytext=coords[idx_c],
                arrowprops=dict(arrowstyle="->", color=COLORS[inj], lw=1.5))
    ax.annotate(TASK_LABELS[inj], coords[idx_c], fontsize=8,
                textcoords="offset points", xytext=(-10, 8))

# Legend for marker shapes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="gray", label="combine", linestyle="None", ms=8),
    Line2D([0], [0], marker="s", color="gray", label="neural_exec", linestyle="None", ms=8),
]
ax.legend(handles=legend_elements)
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
       title=f"PCA of steering vectors (L={PEAK_LAYER})")
savefig(fig, "fig13_pca_combine_neural_exec")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 14 — Steering with v_neural_exec vs v_combine: coef sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 14: steering comparison (combine vs neural_exec vectors) ──")
INJ = "spam"
naive_p = prompts[INJ]["prompts"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
cor_ids = prompts[INJ]["cor_ids"]
inj_ids = prompts[INJ]["inj_ids"]

steer_results = {}
for label, v_dict in [("combine", v_combine), ("neural_exec", v_neural_exec)]:
    vec = v_dict[INJ][PEAK_LAYER]
    asr_curve = []
    for c in tqdm(COEFS, desc=f"fig14 {label}"):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(vec, c))]
        logits, _ = cache_resid(model, naive_p, BATCH, fwd_hooks=hooks)
        m = compute_metrics(logits, cor_ids, inj_ids)
        asr_curve.append(m["asr"])
    steer_results[label] = asr_curve

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(COEFS, steer_results["combine"], color=COLORS["combine"], label="v_combine")
ax.plot(COEFS, steer_results["neural_exec"], color=COLORS["neural_exec"],
        ls="--", label="v_neural_exec")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Steering naive prompts: combine vs neural_exec vector (L={PEAK_LAYER}, {INJ})")
ax.set_ylim(-0.05, 1.05)
ax.legend()
savefig(fig, "fig14_steering_comparison")
save_results({"coefs": COEFS, **steer_results},
             f"{RESULTS_DIR}/fig14.json", layer=PEAK_LAYER, injection=INJ)


print("\n✓ Experiment 4 complete.")

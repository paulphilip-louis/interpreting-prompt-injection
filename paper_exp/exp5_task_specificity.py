"""
Experiment 5 — Task-specificity.

Figures produced:
  fig15 — PCA of all 6 per-task steering vectors (free-form vs fixed-form)
  fig16 — PCA variance breakdown within fixed-form cluster
  fig17 — Cross-main-task transfer: v_combine extracted with different main tasks
  fig18 — PC-space visualization of v_combine per main task
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from datasets import load_dataset

from src.utils.attention_tracker import load_model
from src.data.opi import (load_opi_per_task, INJECTIONS, FORMAT, ANSWER_STRINGS,
                          load_opi_dataset, data_all_attack_types)
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook, save_results,
)
from src.utils.utils import to_first_token_ids, cosine_similarity
from paper_exp.style import apply as apply_style, savefig, COLORS, TASK_LABELS

apply_style()

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PRIMARY_TASK = "sentiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 75
N_TEST = 75
PEAK_LAYER = 21
N_LAYERS = 28

ALL_INJ_TASKS = ["spam", "hsol", "rte", "mrpc", "gigaword", "jfleg"]
FIXED_FORM = ["spam", "hsol", "rte", "mrpc"]
FREE_FORM = ["gigaword", "jfleg"]

# For fig17-18: fix injected task, vary main task
FIXED_INJ = "spam"
MAIN_TASKS = ["sentiment", "hsol", "rte"]

COEFS = np.arange(-1, 5, 0.5).tolist()

RESULTS_DIR = "results/exp5"
CACHE_DIR = "results/cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TASK_LABELS_EXT = {
    **TASK_LABELS,
    "gigaword": "Summarization (GW)",
    "jfleg": "Grammar (JFLEG)",
}


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


def load_freeform_prompts(model, task_type, injected_task):
    """Load prompts for free-form injected tasks without FORMAT replacement on the injection."""
    opi_ds = load_opi_dataset()
    old_task, new_task = FORMAT[task_type]
    prompts = {}
    for attack_type in ["naive", "combine"]:
        filtered = opi_ds.filter(
            lambda row, at=attack_type: (
                row["task_type"] == task_type
                and row["attack_type"] == at
                and row["injected_task"] == injected_task
            )
        )
        chat_prompts = []
        for row in filtered:
            messages = [
                {"role": "system", "content": row["instruction"].replace(old_task, new_task)},
                {"role": "user", "content": row["attack_input"]},
            ]
            chat_prompts.append(model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        prompts[attack_type] = chat_prompts
    return prompts


# ── Load model & data ────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_NAME)

# Get steering vectors for all 6 injected tasks at all layers
print("Computing steering vectors for all injected tasks...")
all_layers = list(range(N_LAYERS))
task_vecs_all = {}

prompts_fixed = load_opi_per_task(model, PRIMARY_TASK)

for inj in ALL_INJ_TASKS:
    if inj in FIXED_FORM:
        naive_p = prompts_fixed[inj]["prompts"]["naive"][:N_TRAIN]
        combine_p = prompts_fixed[inj]["prompts"]["combine"][:N_TRAIN]
    else:
        ff_prompts = load_freeform_prompts(model, PRIMARY_TASK, inj)
        naive_p = ff_prompts["naive"][:N_TRAIN]
        combine_p = ff_prompts["combine"][:N_TRAIN]

    r_naive = load_or_cache_resids(f"{inj}_naive_train", model, naive_p, all_layers)
    r_combine = load_or_cache_resids(f"{inj}_combine_train", model, combine_p, all_layers)
    task_vecs_all[inj] = diff_of_means(r_combine, r_naive, DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 15 — PCA of all 6 per-task steering vectors
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 15: PCA of all 6 task vectors ──")
vecs_at_peak = []
labels = []
for inj in ALL_INJ_TASKS:
    vecs_at_peak.append(task_vecs_all[inj][PEAK_LAYER].cpu().numpy())
    labels.append(inj)

X = np.stack(vecs_at_peak)
pca = PCA(n_components=3)
coords = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(6, 5))
for i, inj in enumerate(labels):
    marker = "s" if inj in FREE_FORM else "o"
    color = COLORS.get(inj, "#666")
    ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker, s=100,
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate(TASK_LABELS_EXT[inj], (coords[i, 0], coords[i, 1]),
                textcoords="offset points", xytext=(8, 5), fontsize=9)

from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", ms=8, label="Fixed-form"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None", ms=8, label="Free-form"),
]
ax.legend(handles=legend)
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
       title=f"PCA of per-task steering vectors (L={PEAK_LAYER})")
savefig(fig, "fig15_pca_all_tasks")

print(f"  Variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}, "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}, PC3={pca.explained_variance_ratio_[2]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 16 — PCA variance within fixed-form cluster
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 16: fixed-form PCA variance ──")
X_fixed = np.stack([task_vecs_all[inj][PEAK_LAYER].cpu().numpy() for inj in FIXED_FORM])
pca_fixed = PCA(n_components=3)
pca_fixed.fit(X_fixed)
var_ratios = pca_fixed.explained_variance_ratio_

print(f"  Fixed-form variance: PC1={var_ratios[0]:.3f}, PC2={var_ratios[1]:.3f}, PC3={var_ratios[2]:.3f}")

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(range(1, 4), var_ratios, color=[COLORS["spam"], COLORS["hsol"], COLORS["rte"]])
ax.set(xlabel="Principal component", ylabel="Variance explained",
       title="PCA within fixed-form tasks")
ax.set_xticks(range(1, 4))
for i, v in enumerate(var_ratios):
    ax.text(i + 1, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
savefig(fig, "fig16_fixedform_pca_variance")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 17 — Cross-main-task transfer
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 17: cross-main-task transfer ──")
# Extract v_combine for the fixed injection (spam) under each main task
main_task_vecs = {}
main_task_data = {}
opi_ds = load_opi_dataset()
for mt in MAIN_TASKS:
    print(f"  Main task: {mt}")
    mt_all = data_all_attack_types(opi_ds, model, task_type=mt,
                                   injected_task=FIXED_INJ, include_clean=True)
    cor_ids = to_first_token_ids(model, ANSWER_STRINGS[mt])
    inj_ids = to_first_token_ids(model, ANSWER_STRINGS[FIXED_INJ])

    naive_p = mt_all["naive"][:N_TRAIN]
    combine_p = mt_all["combine"][:N_TRAIN]
    r_naive = load_or_cache_resids(f"{FIXED_INJ}_naive_{mt}_train", model, naive_p, [PEAK_LAYER])
    r_combine = load_or_cache_resids(f"{FIXED_INJ}_combine_{mt}_train", model, combine_p, [PEAK_LAYER])

    v = (r_combine[PEAK_LAYER].mean(0) - r_naive[PEAK_LAYER].mean(0)).to(DEVICE)
    main_task_vecs[mt] = v
    main_task_data[mt] = {"prompts": mt_all, "cor_ids": cor_ids, "inj_ids": inj_ids}

# Cross-main-task steering: train on main_task_A, test on main_task_B
print("  Running cross-main-task steering...")
transfer_results = {}
for train_mt in main_task_vecs:
    transfer_results[train_mt] = {}
    v_train = main_task_vecs[train_mt]
    for test_mt in main_task_vecs:
        test_naive = main_task_data[test_mt]["prompts"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
        cor_ids = main_task_data[test_mt]["cor_ids"]
        inj_ids = main_task_data[test_mt]["inj_ids"]

        asr_curve = []
        for c in tqdm(COEFS, desc=f"  {train_mt}→{test_mt}", leave=False):
            hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(v_train, c))]
            logits, _ = cache_resid(model, test_naive, BATCH, fwd_hooks=hooks)
            m = compute_metrics(logits, cor_ids, inj_ids)
            asr_curve.append(m["asr"])
        transfer_results[train_mt][test_mt] = asr_curve

# Plot: one subplot per train main task
n_mt = len(main_task_vecs)
mt_list = list(main_task_vecs.keys())
mt_colors = {"sentiment": COLORS["spam"], "hsol": COLORS["hsol"], "rte": COLORS["rte"]}

fig, axes = plt.subplots(1, n_mt, figsize=(5 * n_mt, 4), sharey=True)
if n_mt == 1:
    axes = [axes]
for ax, train_mt in zip(axes, mt_list):
    for test_mt in mt_list:
        ls = "-" if train_mt == test_mt else "--"
        ax.plot(COEFS, transfer_results[train_mt][test_mt],
                color=mt_colors.get(test_mt, "#666"), ls=ls, label=f"test: {test_mt}")
    ax.set(xlabel="Coefficient", title=f"Train: {train_mt}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
axes[0].set_ylabel("ASR")
fig.suptitle(f"Cross-main-task steering transfer (inj={FIXED_INJ}, L={PEAK_LAYER})", y=1.02)
savefig(fig, "fig17_cross_main_task_transfer")
save_results(transfer_results, f"{RESULTS_DIR}/fig17.json",
             coefs=COEFS, layer=PEAK_LAYER, injection=FIXED_INJ, main_tasks=mt_list)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 18 — PC-space visualization of v_combine per main task
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 18: v_combine per main task in PC space ──")
mt_vecs_np = np.stack([main_task_vecs[mt].cpu().numpy() for mt in mt_list])
pca_mt = PCA(n_components=2)
coords_mt = pca_mt.fit_transform(mt_vecs_np)

fig, ax = plt.subplots(figsize=(5, 4))
for i, mt in enumerate(mt_list):
    ax.scatter(coords_mt[i, 0], coords_mt[i, 1], c=mt_colors.get(mt, "#666"),
               s=100, edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate(mt, (coords_mt[i, 0], coords_mt[i, 1]),
                textcoords="offset points", xytext=(8, 5), fontsize=10)
ax.set(xlabel=f"PC1 ({pca_mt.explained_variance_ratio_[0]:.0%})",
       ylabel=f"PC2 ({pca_mt.explained_variance_ratio_[1]:.0%})",
       title=f"v_combine per main task in PC space (inj={FIXED_INJ}, L={PEAK_LAYER})")
savefig(fig, "fig18_main_task_pca")


print("\n✓ Experiment 5 complete.")

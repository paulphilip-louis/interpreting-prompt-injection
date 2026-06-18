"""
Experiment 2 — Cross-task transfer and layer selection.

Figures produced:
  fig7  — LOO cross-task transfer: coefficient sweep per held-out task
          (ASR with bootstrap band; baselines as edge markers, not full lines)
  tab1  — Cosine similarity matrix between per-task steering vectors,
          over ALL injection tasks, grouped by family (free-form | spam/hsol |
          rte/mrpc). Off-diagonal mean in the title.

Conventions:
  • Left padding forced (real last token at [:, -1]).
  • Cache keys encode model, n, layer range, padding side.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import einsum

from src.utils.attention_tracker import load_model
from src.data.opi import load_opi_per_task, INJECTIONS
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook, save_results,
)
from src.utils.utils import cosine_similarity
from paper_exp.style import apply as apply_style, savefig, COLORS, TASK_LABELS
from paper_exp.constants import MODEL_NAME

apply_style()

# ── Load model ─────────────────────────────────────────────────────────
MODEL_TAG = MODEL_NAME.split("/")[-1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = load_model(MODEL_NAME)

# Force LEFT padding so the last position is always the real last token.
model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
PAD_TAG = "padL"

# Does the forward path prepend a BOS? (TransformerLens cfg). Token spans for
# fig6 are shifted by this. Verify this matches cache_resid's tokenization
# (add_special_tokens=False + cfg.default_prepend_bos).
BOS_OFFSET = int(getattr(model.cfg, "default_prepend_bos", False))

# ── Load data ─────────────────────────────────────────────────────────
TASK = "sentiment"
INJ = "spam"
ALL_INJ = INJECTIONS

print("Loading data...")
prompts = load_opi_per_task(model, TASK)
cor_ids = prompts[INJ]["cor_ids"]
inj_ids = prompts[INJ]["inj_ids"]

# ── Experimental settings ─────────────────────────────────────────────
BATCH = 4
N_TRAIN = 75
N_TEST = 75
N_RANDOM_SEEDS = 10
BOOT_SEED = 0
PEAK_LAYER = 24
N_LAYERS = model.cfg.n_layers
N_SPAN_EXAMPLES = 50          # fig6 runs at batch=1; cap the number of prompts

COEFS = np.arange(-0.5, 3.5, 0.5).tolist()

CACHE_LAYERS = list(range(N_LAYERS))

TRIGGERS = ["safe", "naive", "escape", "ignore", "combine", "neural_exec", "random"]

# Table ordering: keep families adjacent (free-form | spam/hsol | rte/mrpc).
COSINE_ORDER = ["gigaword", "jfleg", "spam", "hsol", "rte", "mrpc"]

RESULTS_DIR = "results_llama/exp2"
CACHE_DIR = "results_llama/cache"
FIG_DIR = "paper_exp/figures_llama"
for d in (RESULTS_DIR, CACHE_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

ALL_LAYERS = list(range(N_LAYERS))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _layers_sig(layers):
    return f"L{min(layers)}-{max(layers)}x{len(layers)}"


def load_or_cache_resids(name, model, prompt_list, layers, n):
    key = f"{MODEL_TAG}__{name}__n{n}__{_layers_sig(layers)}__{PAD_TAG}"
    path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        print(f"    loading cache {key}")
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"    computing {key} ...")
    _, resids = cache_resid(model, prompt_list, BATCH, cache_layers=layers)
    torch.save(resids, path)
    return resids


def per_example_asr(logits, cor_ids, inj_ids):
    p = logits.softmax(-1)
    return ((p[:, inj_ids].sum(-1) - p[:, cor_ids].sum(-1)) + 1) / 2


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    a = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(means, 100 * a)), \
        float(np.percentile(means, 100 * (1 - a)))


def steering_vec_at(task, layers, n=N_TRAIN):
    """Diff-of-means vector at PEAK_LAYER for `task` (cached residuals)."""
    r_naive = load_or_cache_resids(f"{task}_naive_train", model,
                                   prompts[task]["prompts"]["naive"][:n], layers, n)
    r_combine = load_or_cache_resids(f"{task}_combine_train", model,
                                     prompts[task]["prompts"]["combine"][:n], layers, n)
    return diff_of_means(r_combine, r_naive, DEVICE)[PEAK_LAYER]


# ── Steering vectors + baselines (fig7 tasks = INJECTIONS) ─────────────────────
print("Caching train residuals & vectors...")
task_vecs = {inj: steering_vec_at(inj, CACHE_LAYERS) for inj in INJECTIONS}

print("Computing baselines...")
baselines = {}
for inj in INJECTIONS:
    print(f"Baseline for {inj}...")
    test_naive = prompts[inj]["prompts"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
    test_combine = prompts[inj]["prompts"]["combine"][N_TRAIN:N_TRAIN + N_TEST]
    ln, _ = cache_resid(model, test_naive, BATCH)
    lc, _ = cache_resid(model, test_combine, BATCH)
    baselines[inj] = {
        "naive": compute_metrics(ln, prompts[inj]["cor_ids"], prompts[inj]["inj_ids"])["asr"],
        "combine": compute_metrics(lc, prompts[inj]["cor_ids"], prompts[inj]["inj_ids"])["asr"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — LOO cross-task transfer
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 7: LOO cross-task transfer ──")
loo = {}
for held_out in INJECTIONS:
    train = [t for t in INJECTIONS if t != held_out]
    v_train = torch.stack([task_vecs[t] for t in train]).mean(0).to(DEVICE)
    target = prompts[held_out]["prompts"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
    cor, inj = prompts[held_out]["cor_ids"], prompts[held_out]["inj_ids"]

    asr_m, asr_lo, asr_hi, ld = [], [], [], []
    for c in tqdm(COEFS, desc=f"fig7 LOO-{held_out}"):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(v_train, c))]
        logits, _ = cache_resid(model, target, BATCH, fwd_hooks=hooks)
        per_ex = per_example_asr(logits, cor, inj).numpy()
        m, lo, hi = bootstrap_ci(per_ex)
        asr_m.append(m); asr_lo.append(lo); asr_hi.append(hi)
        ld.append(compute_metrics(logits, cor, inj)["mean_logit_diff"])
    loo[held_out] = {"asr": asr_m, "asr_lo": asr_lo, "asr_hi": asr_hi, "ld": ld}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for inj in INJECTIONS:
    col, lab = COLORS[inj], TASK_LABELS[inj]
    axes[0].plot(COEFS, loo[inj]["asr"], color=col, label=lab)
    axes[0].fill_between(COEFS, loo[inj]["asr_lo"], loo[inj]["asr_hi"], color=col, alpha=0.15)
    # baselines as edge markers (not full-width lines): naive ○ at left, combine ★ at right
    axes[0].plot(COEFS[0], baselines[inj]["naive"], marker="o", mfc="none", mec=col, ms=6, zorder=5)
    axes[0].plot(COEFS[-1], baselines[inj]["combine"], marker="*", color=col, ms=11, zorder=5)
    axes[1].plot(COEFS, loo[inj]["ld"], color=col, label=lab)
axes[0].plot([], [], marker="o", mfc="none", mec="gray", ls="none", label="naive baseline")
axes[0].plot([], [], marker="*", color="gray", ls="none", label="combine baseline")
axes[0].set(xlabel="Steering coefficient", ylabel="ASR",
            title=f"LOO cross-task transfer (L={PEAK_LAYER})", ylim=(-0.05, 1.05))
axes[0].legend(fontsize=8)
axes[1].set(xlabel="Steering coefficient", ylabel="Mean logit diff",
            title="LOO cross-task logit difference")
axes[1].legend(fontsize=8)
savefig(fig, "fig7_loo_transfer")
save_results({inj: loo[inj] for inj in INJECTIONS},
             f"{RESULTS_DIR}/fig7.json", coefs=COEFS, layer=PEAK_LAYER, boot_seed=BOOT_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — full diff-of-means vector vs only shared component
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 7: full vs shared ──")
u_shared = torch.stack([task_vecs[inj] for inj in INJECTIONS]).mean(0).to(DEVICE)
u_shared = u_shared / u_shared.norm()
full_res = {}
shared_res = {}
task_spec_res = {}

for injection in INJECTIONS:
    full_vec = task_vecs[injection]
    shared_comp = torch.dot(task_vecs[injection], u_shared) * u_shared
    task_spec = full_vec - shared_comp

    target = prompts[injection]["prompts"]["naive"][N_TRAIN:N_TRAIN+N_TEST]
    cor, inj = prompts[injection]["cor_ids"], prompts[injection]["inj_ids"]

    print("—— Steering with full diff-of-means vector ——")
    asr_m, asr_lo, asr_hi, ld = [], [], [], []
    for c in tqdm(COEFS, desc=f"fig8 full vs shared-{injection}"):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(full_vec, c))]
        logits, _ = cache_resid(model, target, BATCH, fwd_hooks=hooks)
        per_ex = per_example_asr(logits, cor, inj).numpy()
        m, lo, hi = bootstrap_ci(per_ex)
        asr_m.append(m); asr_lo.append(lo); asr_hi.append(hi)
        ld.append(compute_metrics(logits, cor, inj)["mean_logit_diff"])
    full_res[injection] = {"asr": asr_m, "asr_lo": asr_lo, "asr_hi": asr_hi, "ld": ld}

    print("—— Steering with shared component ——")
    asr_m, asr_lo, asr_hi, ld = [], [], [], []
    for c in tqdm(COEFS, desc=f"fig7 LOO-{held_out}"):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(shared_comp, c))]
        logits, _ = cache_resid(model, target, BATCH, fwd_hooks=hooks)
        per_ex = per_example_asr(logits, cor, inj).numpy()
        m, lo, hi = bootstrap_ci(per_ex)
        asr_m.append(m); asr_lo.append(lo); asr_hi.append(hi)
        ld.append(compute_metrics(logits, cor, inj)["mean_logit_diff"])
    shared_res[injection] = {"asr": asr_m, "asr_lo": asr_lo, "asr_hi": asr_hi, "ld": ld}

    print("—— Steering with task-specific component ——")
    asr_m, asr_lo, asr_hi, ld = [], [], [], []
    for c in tqdm(COEFS, desc=f"fig7 LOO-{held_out}"):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(task_spec, c))]
        logits, _ = cache_resid(model, target, BATCH, fwd_hooks=hooks)
        per_ex = per_example_asr(logits, cor, inj).numpy()
        m, lo, hi = bootstrap_ci(per_ex)
        asr_m.append(m); asr_lo.append(lo); asr_hi.append(hi)
        ld.append(compute_metrics(logits, cor, inj)["mean_logit_diff"])
    task_spec_res[injection] = {"asr": asr_m, "asr_lo": asr_lo, "asr_hi": asr_hi, "ld": ld}

STYLES = {"full": "-", "shared": "--", "task_spec": ":"}
STYLE_LABELS = {"full": "Full v", "shared": "Shared component", "task_spec": "Task-specific component"}

fig, axes = plt.subplots(len(INJECTIONS), 2, figsize=(11, 3.5 * len(INJECTIONS)),
                         sharex=True, sharey="col")

for idx, inj in enumerate(INJECTIONS):
    col = COLORS[inj]
    ax_asr, ax_ld = axes[idx][0], axes[idx][1]

    for key, res in [("full", full_res), ("shared", shared_res), ("task_spec", task_spec_res)]:
        ls = STYLES[key]
        label = STYLE_LABELS[key] if idx == 0 else None   # legend only on top row
        ax_asr.plot(COEFS, res[inj]["asr"], color=col, ls=ls, label=label)
        ax_asr.fill_between(COEFS, res[inj]["asr_lo"], res[inj]["asr_hi"],
                            color=col, alpha=0.10)
        ax_ld.plot(COEFS, res[inj]["ld"], color=col, ls=ls)

    # Baselines as edge markers
    ax_asr.plot(COEFS[0], baselines[inj]["naive"], marker="o", mfc="none", mec=col, ms=6, zorder=5)
    ax_asr.plot(COEFS[-1], baselines[inj]["combine"], marker="*", color=col, ms=11, zorder=5)
    ax_asr.set_ylim(-0.05, 1.05)
    ax_asr.set_ylabel("ASR")
    ax_ld.set_ylabel("Mean logit diff")
    ax_asr.set_title(f"{TASK_LABELS[inj]}", fontsize=10)

# Shared legend on top row only
axes[0][0].legend(fontsize=8)
# Marker legend entries (once)
axes[0][0].plot([], [], marker="o", mfc="none", mec="gray", ls="none", label="naive baseline")
axes[0][0].plot([], [], marker="*", color="gray", ls="none", label="combine baseline")
axes[0][0].legend(fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("Steering coefficient")

fig.suptitle(f"Full vs shared vs task-specific steering (L={PEAK_LAYER})", y=1.01)
savefig(fig, "fig8_shared_vs_taskspecific")
save_results({"full": full_res, "shared": shared_res, "task_spec": task_spec_res},
             f"{RESULTS_DIR}/fig8.json", coefs=COEFS, layer=PEAK_LAYER, boot_seed=BOOT_SEED)
# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Cosine matrix over ALL injection tasks, grouped by family
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Table 1: cosine matrix (all tasks) ──")
cosine_tasks = [t for t in COSINE_ORDER if t in prompts]
missing = [t for t in COSINE_ORDER if t not in prompts]
if missing:
    print(f"  ⚠ not in dataset, skipped: {missing}")

# Free-form vectors may not be cached yet; cache just PEAK_LAYER for those.
cos_vecs = {}
for t in cosine_tasks:
    cos_vecs[t] = task_vecs[t] if t in task_vecs else steering_vec_at(t, [PEAK_LAYER])

n = len(cosine_tasks)
cos = np.zeros((n, n))
for i, t1 in enumerate(cosine_tasks):
    for j, t2 in enumerate(cosine_tasks):
        cos[i, j] = cosine_similarity(cos_vecs[t1], cos_vecs[t2])

off = cos[np.triu_indices(n, k=1)]
off_mean, off_std = float(off.mean()), float(off.std())
print("        " + "  ".join(f"{t:>8}" for t in cosine_tasks))
for i, t in enumerate(cosine_tasks):
    print(f"{t:>8} " + "  ".join(f"{cos[i,j]:8.3f}" for j in range(n)))
print(f"\nOff-diagonal mean: {off_mean:.3f} ± {off_std:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cos, cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in cosine_tasks], rotation=45, ha="right")
ax.set_yticklabels([TASK_LABELS.get(t, t) for t in cosine_tasks])
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{cos[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if cos[i, j] > 0.65 else "black")
fig.colorbar(im, ax=ax, label="Cosine similarity")
ax.set_title(f"Per-task steering-vector cosine (L={PEAK_LAYER})\n"
             f"off-diagonal mean = {off_mean:.3f} ± {off_std:.3f}")
savefig(fig, "tab1_cosine_matrix")
save_results({"matrix": cos.tolist(), "tasks": cosine_tasks,
              "off_diag_mean": off_mean, "off_diag_std": off_std},
             f"{RESULTS_DIR}/tab1.json", layer=PEAK_LAYER)


print("\n✓ Experiment 2 complete.")
"""
Experiment 5 — Task-specificity (main-task dependency of the injection direction).

Injected task is FIXED (spam); we vary the MAIN task the model is legitimately
performing, and ask how v_combine depends on it.

Figures produced:
  fig15 — Cosine matrix of v_combine across main tasks (injection fixed=spam)
  fig16 — Cross-main-task SUFFICIENCY: steer naive prompts with v from main
          task A, evaluate on main task B  (one subplot per test task, bands)
  fig17 — Cross-main-task NECESSITY: ablate (negative-steer) combine prompts
          with v from main task A, evaluate on B  (one subplot per test task)

Diagonal (train==test) is the within-main-task positive control (train/test
split); off-diagonal is the cross-main-task transfer test. Weak off-diagonal ⇒
the direction is main-task-specific.

Note: necessity here is additive negative steering at the peak layer (the
cross-main-task analog of exp3 fig9). The full multi-layer projection clamp lives
in exp3.

Conventions: left padding forced; cache keys encode model/n/layers/padding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.attention_tracker import load_model
from src.data.opi import (ANSWER_STRINGS, load_opi_dataset, data_all_attack_types)
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook, save_results,
)
from src.utils.utils import to_first_token_ids, cosine_similarity
from paper_exp.style import apply as apply_style, savefig

apply_style()

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = MODEL_NAME.split("/")[-1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 75
N_TEST = 75
BOOT_SEED = 0
PEAK_LAYER = 21

FIXED_INJ = "spam"
MAIN_TASKS = ["sentiment", "hsol", "rte", "mrpc"]   # legitimate task; injection fixed

COEFS_SUFF = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]      # add direction to naive
COEFS_ABL = [0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0] # subtract from combine

# Dedicated palette for MAIN tasks (do NOT reuse injected-task colours).
MT_COLORS = {"sentiment": "#3a7ca5", "hsol": "#d1495b",
             "rte": "#66a182", "mrpc": "#e09f3e"}
MT_LABEL = {"sentiment": "Sentiment", "hsol": "HateSpeech", "rte": "RTE", "mrpc": "MRPC"}

RESULTS_DIR = "results/exp5"
CACHE_DIR = "results/cache"
for d in (RESULTS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)


# ── Load model & data ─────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_NAME)

model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
PAD_TAG = "padL"

opi_ds = load_opi_dataset()


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


def steering_curve(target_prompts, vec, coefs, cor_ids, inj_ids, desc=""):
    m_, lo_, hi_ = [], [], []
    for c in tqdm(coefs, desc=desc, leave=False):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(vec, c))]
        logits, _ = cache_resid(model, target_prompts, BATCH, fwd_hooks=hooks)
        mm, lo, hi = bootstrap_ci(per_example_asr(logits, cor_ids, inj_ids).numpy())
        m_.append(mm); lo_.append(lo); hi_.append(hi)
    return {"asr": m_, "lo": lo_, "hi": hi_}


def make_grid(n):
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


# ── Extract main-task-conditioned vectors, data, baselines ────────────────────
print(f"Extracting v_combine per main task (injection={FIXED_INJ})...")
main_vec, main_data, baselines = {}, {}, {}
for mt in MAIN_TASKS:
    print(f"  main task: {mt}")
    mt_all = data_all_attack_types(opi_ds, model, task_type=mt,
                                   injected_task=FIXED_INJ, include_clean=True)
    cor_ids = to_first_token_ids(model, ANSWER_STRINGS[mt])
    inj_ids = to_first_token_ids(model, ANSWER_STRINGS[FIXED_INJ])

    r_naive = load_or_cache_resids(f"{FIXED_INJ}_naive_{mt}_train", model,
                                   mt_all["naive"][:N_TRAIN], [PEAK_LAYER], N_TRAIN)
    r_combine = load_or_cache_resids(f"{FIXED_INJ}_combine_{mt}_train", model,
                                     mt_all["combine"][:N_TRAIN], [PEAK_LAYER], N_TRAIN)
    main_vec[mt] = diff_of_means(r_combine, r_naive, DEVICE)[PEAK_LAYER]
    main_data[mt] = {"all": mt_all, "cor_ids": cor_ids, "inj_ids": inj_ids}

    naive_test = mt_all["naive"][N_TRAIN:N_TRAIN + N_TEST]
    combine_test = mt_all["combine"][N_TRAIN:N_TRAIN + N_TEST]
    ln, _ = cache_resid(model, naive_test, BATCH)
    lc, _ = cache_resid(model, combine_test, BATCH)
    baselines[mt] = {
        "naive": compute_metrics(ln, cor_ids, inj_ids)["asr"],
        "combine": compute_metrics(lc, cor_ids, inj_ids)["asr"],
    }
    print(f"    baseline naive={baselines[mt]['naive']:.3f} combine={baselines[mt]['combine']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 15 — Cosine matrix of v_combine across main tasks (injection fixed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 15: cosine across main tasks ──")
n = len(MAIN_TASKS)
cos = np.zeros((n, n))
for i, a in enumerate(MAIN_TASKS):
    for j, b in enumerate(MAIN_TASKS):
        cos[i, j] = cosine_similarity(main_vec[a], main_vec[b])
off = cos[np.triu_indices(n, k=1)]
off_mean, off_std = float(off.mean()), float(off.std())
print(f"  off-diagonal cosine mean: {off_mean:.3f} ± {off_std:.3f}")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cos, cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([MT_LABEL[t] for t in MAIN_TASKS], rotation=45, ha="right")
ax.set_yticklabels([MT_LABEL[t] for t in MAIN_TASKS])
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{cos[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if cos[i, j] > 0.65 else "black")
fig.colorbar(im, ax=ax, label="Cosine similarity")
ax.set_title(f"v_combine across main tasks (inj={FIXED_INJ}, L={PEAK_LAYER})\n"
             f"off-diag mean = {off_mean:.3f} ± {off_std:.3f}")
savefig(fig, "fig15_main_task_cosine")
save_results({"matrix": cos.tolist(), "main_tasks": MAIN_TASKS,
              "off_diag_mean": off_mean, "off_diag_std": off_std},
             f"{RESULTS_DIR}/fig15.json", layer=PEAK_LAYER, injection=FIXED_INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 16 — Cross-main-task SUFFICIENCY (steer naive prompts)
#   subplot per TEST task; curves = TRAIN source; diagonal solid.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 16: cross-main-task sufficiency ──")
suff = {te: {} for te in MAIN_TASKS}
for test_mt in MAIN_TASKS:
    naive_test = main_data[test_mt]["all"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
    cor, inj = main_data[test_mt]["cor_ids"], main_data[test_mt]["inj_ids"]
    for train_mt in MAIN_TASKS:
        suff[test_mt][train_mt] = steering_curve(
            naive_test, main_vec[train_mt], COEFS_SUFF, cor, inj,
            desc=f"suff {train_mt}→{test_mt}")

fig, axes = make_grid(len(MAIN_TASKS))
for ax, test_mt in zip(axes, MAIN_TASKS):
    for train_mt in MAIN_TASKS:
        d = suff[test_mt][train_mt]
        solid = train_mt == test_mt
        ax.plot(COEFS_SUFF, d["asr"], color=MT_COLORS[train_mt],
                ls="-" if solid else "--", lw=2 if solid else 1.2,
                label=f"from {MT_LABEL[train_mt]}" + (" (self)" if solid else ""))
        ax.fill_between(COEFS_SUFF, d["lo"], d["hi"], color=MT_COLORS[train_mt], alpha=0.12)
    ax.axhline(baselines[test_mt]["naive"], ls=":", color="gray", lw=0.8)
    ax.axhline(baselines[test_mt]["combine"], ls="--", color="gray", lw=0.8)
    ax.set_title(f"Test: {MT_LABEL[test_mt]}"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
for ax in axes[:len(MAIN_TASKS)]:
    ax.set_xlabel("Steering coefficient"); ax.set_ylabel("ASR")
fig.suptitle(f"Cross-main-task sufficiency — steer naive (inj={FIXED_INJ}, L={PEAK_LAYER})", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95]) 
savefig(fig, "fig16_cross_main_task_sufficiency")
save_results(suff, f"{RESULTS_DIR}/fig16.json",
             coefs=COEFS_SUFF, layer=PEAK_LAYER, injection=FIXED_INJ,
             baselines=baselines, boot_seed=BOOT_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 17 — Cross-main-task NECESSITY (ablate combine prompts, negative steer)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 17: cross-main-task necessity ──")
abl = {te: {} for te in MAIN_TASKS}
for test_mt in MAIN_TASKS:
    combine_test = main_data[test_mt]["all"]["combine"][N_TRAIN:N_TRAIN + N_TEST]
    cor, inj = main_data[test_mt]["cor_ids"], main_data[test_mt]["inj_ids"]
    for train_mt in MAIN_TASKS:
        abl[test_mt][train_mt] = steering_curve(
            combine_test, main_vec[train_mt], COEFS_ABL, cor, inj,
            desc=f"abl {train_mt}→{test_mt}")

fig, axes = make_grid(len(MAIN_TASKS))
for ax, test_mt in zip(axes, MAIN_TASKS):
    for train_mt in MAIN_TASKS:
        d = abl[test_mt][train_mt]
        solid = train_mt == test_mt
        ax.plot(COEFS_ABL, d["asr"], color=MT_COLORS[train_mt],
                ls="-" if solid else "--", lw=2 if solid else 1.2,
                label=f"from {MT_LABEL[train_mt]}" + (" (self)" if solid else ""))
        ax.fill_between(COEFS_ABL, d["lo"], d["hi"], color=MT_COLORS[train_mt], alpha=0.12)
    ax.axhline(baselines[test_mt]["combine"], ls="--", color="gray", lw=0.8)
    ax.axhline(baselines[test_mt]["naive"], ls=":", color="gray", lw=0.8)
    ax.set_title(f"Test: {MT_LABEL[test_mt]}"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
for ax in axes[:len(MAIN_TASKS)]:
    ax.set_xlabel("Negative steering coefficient"); ax.set_ylabel("ASR")
fig.suptitle(f"Cross-main-task necessity — ablate combine (inj={FIXED_INJ}, L={PEAK_LAYER})", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95]) 
savefig(fig, "fig17_cross_main_task_necessity")
save_results(abl, f"{RESULTS_DIR}/fig17.json",
             coefs=COEFS_ABL, layer=PEAK_LAYER, injection=FIXED_INJ,
             baselines=baselines, boot_seed=BOOT_SEED)


print("\n✓ Experiment 5 complete.")
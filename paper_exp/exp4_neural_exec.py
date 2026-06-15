"""
Experiment 4 — Comparison with optimised (neural-exec) triggers.

Figures produced:
  fig10  — ASR baselines: naive / combine / neural_exec / random (with CI)
  fig11  — Per-layer norm: v_combine vs v_neural_exec
  fig12  — Cosine(v_combine, v_neural_exec) across layers, per task (+ onset)
  fig13a — Δ = v_neural_exec − v_combine: variance explained (uncentered SVD)
           + cosine heatmap of per-task Δ  → "consistent translation"
  fig13b — Magnitude table: ||Δ|| / ||v_combine|| per task
  fig14  — Steering naive→injection with v_combine vs v_neural_exec,
           2×2 per task, with bootstrap band

Note: v_neural_exec rests on a single optimised (GCG) trigger per model. Treat
cross-task / magnitude claims with that caveat until 2–3 triggers are available.

Conventions: left padding forced; cache keys encode model/n/layers/padding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
MODEL_TAG = MODEL_NAME.split("/")[-1]
TASK = "sentiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 75
N_TEST = 75
BOOT_SEED = 0
PEAK_LAYER = 21
N_LAYERS = 28
COEFS = np.arange(-1, 5, 0.5).tolist()
ONSET_THRESH = 0.5

CONDITIONS = ["naive", "combine", "neural_exec", "random"]
RESULTS_DIR = "results/exp4"
CACHE_DIR = "results/cache"
for d in (RESULTS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)
ALL_LAYERS = list(range(N_LAYERS))


# ── Load model & data ─────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_NAME)

model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
PAD_TAG = "padL"

print("Loading data...")
prompts = load_opi_per_task(model, TASK)


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


# ── Train residuals & vectors ─────────────────────────────────────────────────
print("Caching train residuals...")
task_resids = {}
for inj in INJECTIONS:
    task_resids[inj] = {
        cond: load_or_cache_resids(f"{inj}_{cond}_train", model,
                                   prompts[inj]["prompts"][cond][:N_TRAIN], ALL_LAYERS, N_TRAIN)
        for cond in CONDITIONS
    }

v_combine, v_neural_exec = {}, {}
for inj in INJECTIONS:
    v_combine[inj] = diff_of_means(task_resids[inj]["combine"], task_resids[inj]["naive"], DEVICE)
    v_neural_exec[inj] = diff_of_means(task_resids[inj]["neural_exec"], task_resids[inj]["naive"], DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — ASR baselines per task (seeded CI)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 10: baselines ──")
baseline_data = {}
for inj in INJECTIONS:
    baseline_data[inj] = {}
    for cond in CONDITIONS:
        p = prompts[inj]["prompts"][cond][N_TRAIN:N_TRAIN + N_TEST]
        logits, _ = cache_resid(model, p, BATCH)
        per_ex = per_example_asr(logits, prompts[inj]["cor_ids"], prompts[inj]["inj_ids"]).numpy()
        m, lo, hi = bootstrap_ci(per_ex)
        baseline_data[inj][cond] = {"asr": m, "ci_lo": lo, "ci_hi": hi}

cond_colors = {"naive": COLORS["naive"], "combine": COLORS["combine"],
               "neural_exec": COLORS["neural_exec"], "random": COLORS["random_ctrl"]}
fig, axes = plt.subplots(1, len(INJECTIONS), figsize=(3.5 * len(INJECTIONS), 3.5), sharey=True)
for ax, inj in zip(axes, INJECTIONS):
    x = range(len(CONDITIONS))
    means = [baseline_data[inj][c]["asr"] for c in CONDITIONS]
    yerr = np.array([[baseline_data[inj][c]["asr"] - baseline_data[inj][c]["ci_lo"],
                      baseline_data[inj][c]["ci_hi"] - baseline_data[inj][c]["asr"]]
                     for c in CONDITIONS]).T
    ax.bar(x, means, yerr=yerr, capsize=4,
           color=[cond_colors[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS, rotation=30, ha="right", fontsize=8)
    ax.set_title(TASK_LABELS[inj]); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("ASR")
fig.suptitle("Baseline ASR by trigger type", y=1.02)
savefig(fig, "fig10_neural_exec_baselines")
save_results(baseline_data, f"{RESULTS_DIR}/fig10.json", boot_seed=BOOT_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Per-layer norms
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 11: norms ──")
norm_data = {}
fig, axes = plt.subplots(1, len(INJECTIONS), figsize=(3.5 * len(INJECTIONS), 3), sharey=True)
for ax, inj in zip(axes, INJECTIONS):
    nc = [v_combine[inj][l].norm().item() for l in ALL_LAYERS]
    nn = [v_neural_exec[inj][l].norm().item() for l in ALL_LAYERS]
    norm_data[inj] = {"combine": nc, "neural_exec": nn}
    ax.plot(ALL_LAYERS, nc, label="combine", color=COLORS["combine"])
    ax.plot(ALL_LAYERS, nn, label="neural_exec", color=COLORS["neural_exec"])
    ax.set(xlabel="Layer", title=TASK_LABELS[inj]); ax.legend(fontsize=7)
axes[0].set_ylabel("||v||")
fig.suptitle("Steering vector norms across layers", y=1.02)
savefig(fig, "fig11_norms")
save_results(norm_data, f"{RESULTS_DIR}/fig11.json", layers=ALL_LAYERS)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Cosine across layers + onset (counted from L>=1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 12: cosine across layers ──")
cos_by_task, onset = {}, {}
fig, ax = plt.subplots(figsize=(7, 4))
for inj in INJECTIONS:
    cos_vals = [cosine_similarity(v_combine[inj][l], v_neural_exec[inj][l]) for l in ALL_LAYERS]
    cos_by_task[inj] = cos_vals
    # Onset = first layer >=1 above threshold (skip L0: some tasks spike then drop)
    onset[inj] = next((l for l in ALL_LAYERS[1:] if cos_vals[l] > ONSET_THRESH), None)
    ax.plot(ALL_LAYERS, cos_vals, color=COLORS[inj],
            label=f"{TASK_LABELS[inj]} (onset L{onset[inj]})")
    print(f"  {inj}: onset L{onset[inj]}")
ax.axhline(ONSET_THRESH, ls="--", color="gray", lw=0.8)
ax.axhline(0, ls=":", color="gray", lw=0.6)
ax.set(xlabel="Layer", ylabel="Cosine similarity",
       title="v_combine vs v_neural_exec alignment")
ax.legend(fontsize=8)
savefig(fig, "fig12_cosine_combine_vs_neural_exec")
save_results({"cos": cos_by_task, "onset": onset, "threshold": ONSET_THRESH},
             f"{RESULTS_DIR}/fig12.json", layers=ALL_LAYERS)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 — Δ = v_neural_exec − v_combine at peak layer: is the translation shared?
#   • Uncentered PCA (centering would remove exactly the component we want to show).
#     PC1 dominating ⇒ one shared direction explains the bulk of Δ.
#   • Cosine heatmap of per-task Δ: high off-diagonal ⇒ Δ points the same way.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 13: Δ structure ──")
deltas = np.stack([(v_neural_exec[inj][PEAK_LAYER] - v_combine[inj][PEAK_LAYER]).cpu().numpy()
                   for inj in INJECTIONS])                      # [T, d]

# Uncentered variance decomposition
_, S, _ = np.linalg.svd(deltas, full_matrices=False)
ev = (S ** 2 / (S ** 2).sum())

# Cosine heatmap of Δ
Dn = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
delta_cos = Dn @ Dn.T
off = delta_cos[np.triu_indices(len(INJECTIONS), k=1)]
print(f"  Δ off-diagonal cosine mean: {off.mean():.3f} ± {off.std():.3f}")
print(f"  variance explained by PC1 (uncentered): {ev[0]:.3f}")

fig, (axv, axc) = plt.subplots(1, 2, figsize=(11, 4))
k = len(ev)
axv.bar(range(1, k + 1), ev, color=COLORS["neural_exec"], alpha=0.8)
axv.plot(range(1, k + 1), np.cumsum(ev), marker="o", color="black", label="cumulative")
axv.set(xlabel="Component", ylabel="Fraction of variance",
        title=f"Δ variance (uncentered SVD) — PC1={ev[0]:.0%}", xticks=range(1, k + 1),
        ylim=(0, 1.05))
axv.legend(fontsize=8)

im = axc.imshow(delta_cos, cmap="RdYlBu_r", vmin=0, vmax=1)
n = len(INJECTIONS)
axc.set_xticks(range(n)); axc.set_yticks(range(n))
axc.set_xticklabels([TASK_LABELS[t] for t in INJECTIONS], rotation=45, ha="right")
axc.set_yticklabels([TASK_LABELS[t] for t in INJECTIONS])
for i in range(n):
    for j in range(n):
        axc.text(j, i, f"{delta_cos[i,j]:.2f}", ha="center", va="center", fontsize=9,
                 color="white" if delta_cos[i, j] > 0.65 else "black")
fig.colorbar(im, ax=axc, label="Cosine")
axc.set_title(f"Δ cosine across tasks\noff-diag mean = {off.mean():.3f} ± {off.std():.3f}")
savefig(fig, "fig13a_delta_structure")

# Magnitude table: ||Δ|| / ||v_combine||
ratios = {}
for i, inj in enumerate(INJECTIONS):
    d_norm = float(np.linalg.norm(deltas[i]))
    vc_norm = v_combine[inj][PEAK_LAYER].norm().item()
    ratios[inj] = {"delta_norm": d_norm, "vcombine_norm": vc_norm, "ratio": d_norm / vc_norm}

fig, ax = plt.subplots(figsize=(6, 0.5 * len(INJECTIONS) + 1.2))
ax.axis("off")
cells = [[f"{ratios[t]['delta_norm']:.2f}", f"{ratios[t]['vcombine_norm']:.2f}",
          f"{ratios[t]['ratio']:.2f}"] for t in INJECTIONS]
tbl = ax.table(cellText=cells, rowLabels=[TASK_LABELS[t] for t in INJECTIONS],
               colLabels=["||Δ||", "||v_combine||", "||Δ||/||v_combine||"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
ax.set_title(f"Translation magnitude relative to v_combine (L={PEAK_LAYER})", pad=12)
savefig(fig, "fig13b_delta_magnitude")

save_results({"ev_uncentered": ev.tolist(), "delta_cos": delta_cos.tolist(),
              "delta_off_diag_mean": float(off.mean()), "ratios": ratios},
             f"{RESULTS_DIR}/fig13.json", layer=PEAK_LAYER, tasks=list(INJECTIONS))


# ══════════════════════════════════════════════════════════════════════════════
# Fig 14 — Steering naive→injection: v_combine vs v_neural_exec, 2×2 per task
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 14: steering comparison (2×2) ──")
steer = {}
for inj in INJECTIONS:
    naive_p = prompts[inj]["prompts"]["naive"][N_TRAIN:N_TRAIN + N_TEST]
    cor, inj_ids = prompts[inj]["cor_ids"], prompts[inj]["inj_ids"]
    steer[inj] = {}
    for label, v_dict in [("combine", v_combine), ("neural_exec", v_neural_exec)]:
        vec = v_dict[inj][PEAK_LAYER]
        m_, lo_, hi_ = [], [], []
        for c in tqdm(COEFS, desc=f"fig14 {inj}/{label}"):
            hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(vec, c))]
            logits, _ = cache_resid(model, naive_p, BATCH, fwd_hooks=hooks)
            mm, lo, hi = bootstrap_ci(per_example_asr(logits, cor, inj_ids).numpy())
            m_.append(mm); lo_.append(lo); hi_.append(hi)
        steer[inj][label] = {"asr": m_, "lo": lo_, "hi": hi_}

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for ax, inj in zip(axes.flatten(), INJECTIONS):
    for label, col in [("combine", COLORS["combine"]), ("neural_exec", COLORS["neural_exec"])]:
        d = steer[inj][label]
        ax.plot(COEFS, d["asr"], color=col, label=f"v_{label}")
        ax.fill_between(COEFS, d["lo"], d["hi"], color=col, alpha=0.15)
    ax.axhline(baseline_data[inj]["naive"]["asr"], ls=":", color=COLORS["naive"], lw=0.8)
    ax.set_title(TASK_LABELS[inj]); ax.set_ylim(-0.05, 1.05)
for ax in axes[-1]:
    ax.set_xlabel("Steering coefficient")
for ax in axes[:, 0]:
    ax.set_ylabel("ASR")
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Steering naive→injection: v_combine vs v_neural_exec (L={PEAK_LAYER})")
fig.tight_layout(rect=[0, 0, 1, 0.95]) 
savefig(fig, "fig14_steering_comparison")
save_results(steer, f"{RESULTS_DIR}/fig14.json", coefs=COEFS, layer=PEAK_LAYER, boot_seed=BOOT_SEED)


print("\n✓ Experiment 4 complete.")
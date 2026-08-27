"""
Experiment 3 — Coefficient sweep and ablation (necessity).
 
Two DISTINCT interventions, never overlaid on one coefficient axis:
  • Clamp  : projection onto the injection axis forced to the naive-condition
             mean, applied at every layer L18→27. No coefficient.
  • Sweep  : additive c·v at a single layer (L21), c ∈ [0, -3]. Large |c|
             overshoots the naive projection → out-of-distribution, not a
             clean ablation. Reported separately, with the c=-3 endpoint shown
             alongside the clamp in Table 2 precisely to expose that gap.
 
Figures / tables produced:
  tab2  — Comparison table: baseline vs clamp(L18+) vs neg-steer@-3(L21),
          mean ± std over seeds  (the two methods are NOT a single sweep)
  fig9a — Within-task negative coef sweep on combine prompts (mean ± std)
  fig9b — Cross-task ablation: two source vectors (spam, mrpc) → all targets
  fig9c — LOO ablation: LOO vector applied to held-out task's combine prompts
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
import torch
import numpy as np
import matplotlib.pyplot as plt
 
from src.utils.attention_tracker import load_model
from src.data.opi import load_opi_per_task, INJECTIONS
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook, save_results,
)
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

 
# ── Settings ──────────────────────────────────────────────────────────────────
BATCH = 4
N_TRAIN = 75
N_TEST = 75
N_POOL = N_TRAIN + N_TEST          # cache once, resample train/test inside
PEAK_LAYER = 22
N_LAYERS = model.cfg.n_layers
CLAMP_FROM = 21
 
COEFS = [0, -0.5, -1, -1.5, -2, -2.5, -3]
SOURCE_TASKS = ["spam", "mrpc"]
SEEDS = [0, 1, 2]                  # extraction+eval split seeds for error bars
 
RESULTS_DIR = "results_llama/exp3"
CACHE_DIR = "results_llama/cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
 
ALL_LAYERS = list(range(N_LAYERS))
CLAMP_LAYERS = list(range(CLAMP_FROM, N_LAYERS))
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _layers_sig(layers):
    return f"L{min(layers)}-{max(layers)}x{len(layers)}"
 
 
def load_or_cache_resids(name, model, prompt_list, layers, n):
    """Cache key now encodes model, n, and layer range so a change in any of
    them produces a fresh file instead of silently reusing a stale cache."""
    key = f"{MODEL_TAG}__{name}__n{n}__{_layers_sig(layers)}"
    path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        print(f"    loading cache {key}")
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"    computing {key} ...")
    _, resids = cache_resid(model, prompt_list, BATCH, cache_layers=layers)
    torch.save(resids, path)
    return resids
 
 
def slice_resids(resids, idx):
    """resids: {L: [N_POOL, d]} → {L: [len(idx), d]} (cache_resid = last token)."""
    return {L: resids[L][idx] for L in resids}
 
 
def make_clamp_hooks(steering_vecs, naive_resids_train, clamp_layers):
    """Clamp last-token projection onto the injection axis to the naive-train
    mean, at each clamp layer (re-applied per layer to counter the Hydra
    effect)."""
    hooks = []
    for L in clamp_layers:
        v = steering_vecs[L].to(DEVICE)
        v_hat = v / v.norm()
        target_proj = (naive_resids_train[L].to(DEVICE) @ v_hat).mean().item()
 
        def _make(v_hat, target_proj):
            def hook_fn(resid, hook):
                proj = resid[:, -1, :] @ v_hat
                resid[:, -1, :] += (target_proj - proj).unsqueeze(-1) * v_hat
                return resid
            return hook_fn
 
        hooks.append((f"blocks.{L}.hook_resid_post", _make(v_hat, target_proj)))
    return hooks
 
 
def negative_coef_sweep(model, prompts_list, vec, layer, coefs, cor_ids, inj_ids):
    asr, ld = [], []
    for c in coefs:
        hooks = [(f"blocks.{layer}.hook_resid_post", make_steering_hook(vec, c))]
        logits, _ = cache_resid(model, prompts_list, BATCH, fwd_hooks=hooks)
        m = compute_metrics(logits, cor_ids, inj_ids)
        asr.append(m["asr"]); ld.append(m["mean_logit_diff"])
    return {"asr": asr, "ld": ld}
 
 
def split_for_seed(seed, n_pool=N_POOL):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    return perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_TEST]
 
 
def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"
 

# Cache the full pool once per task; seeds resample indices into it.
print("Caching pooled residuals (once per task) ...")
pool_resids, pool_combine_prompts = {}, {}
for inj in INJECTIONS:
    naive_pool = prompts[inj]["prompts"]["naive"][:N_POOL]
    combine_pool = prompts[inj]["prompts"]["combine"][:N_POOL]
    pool_combine_prompts[inj] = combine_pool
    pool_resids[inj] = {
        "naive": load_or_cache_resids(f"{inj}_naive_pool", model, naive_pool, ALL_LAYERS, N_POOL),
        "combine": load_or_cache_resids(f"{inj}_combine_pool", model, combine_pool, ALL_LAYERS, N_POOL),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Seeded core: per seed, build train vectors, run baseline / clamp / within-sweep
# ══════════════════════════════════════════════════════════════════════════════
# Accumulators: metric → task → list over seeds
acc = {"baseline": {t: [] for t in INJECTIONS},
       "clamp": {t: [] for t in INJECTIONS},
       "sweep": {t: [] for t in INJECTIONS}}      # sweep[t] = list of asr-arrays
 
splits = {}  # keep seed-0 split to reuse for fig9b / fig9c
 
for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    torch.manual_seed(seed)
    train_idx, test_idx = split_for_seed(seed)
    splits[seed] = (train_idx, test_idx)
 
    # Per-task train vectors (from this seed's train indices) + clamp targets
    seed_vecs, seed_naive_train = {}, {}
    for inj in INJECTIONS:
        c_tr = slice_resids(pool_resids[inj]["combine"], train_idx)
        n_tr = slice_resids(pool_resids[inj]["naive"], train_idx)
        seed_vecs[inj] = diff_of_means(c_tr, n_tr, DEVICE)
        seed_naive_train[inj] = n_tr
 
    for inj in INJECTIONS:
        test_prompts = [pool_combine_prompts[inj][i] for i in test_idx]
        cor_ids, inj_ids = prompts[inj]["cor_ids"], prompts[inj]["inj_ids"]
 
        # baseline (no hook)
        logits, _ = cache_resid(model, test_prompts, BATCH)
        acc["baseline"][inj].append(compute_metrics(logits, cor_ids, inj_ids)["asr"])
 
        # clamp (L18+)
        hooks = make_clamp_hooks(seed_vecs[inj], seed_naive_train[inj], CLAMP_LAYERS)
        logits, _ = cache_resid(model, test_prompts, BATCH, fwd_hooks=hooks)
        acc["clamp"][inj].append(compute_metrics(logits, cor_ids, inj_ids)["asr"])
 
        # within-task additive sweep (single layer L21)
        sweep = negative_coef_sweep(model, test_prompts, seed_vecs[inj][PEAK_LAYER],
                                    PEAK_LAYER, COEFS, cor_ids, inj_ids)
        acc["sweep"][inj].append(np.array(sweep["asr"]))
        print(f"  {inj}: base={acc['baseline'][inj][-1]:.3f} "
              f"clamp={acc['clamp'][inj][-1]:.3f} sweep@-3={sweep['asr'][-1]:.3f}")
 
# Aggregate
base_m = {t: float(np.mean(acc["baseline"][t])) for t in INJECTIONS}
base_s = {t: float(np.std(acc["baseline"][t])) for t in INJECTIONS}
clamp_m = {t: float(np.mean(acc["clamp"][t])) for t in INJECTIONS}
clamp_s = {t: float(np.std(acc["clamp"][t])) for t in INJECTIONS}
sweep_stack = {t: np.stack(acc["sweep"][t]) for t in INJECTIONS}        # [seeds, coefs]
sweep_m = {t: sweep_stack[t].mean(0) for t in INJECTIONS}
sweep_s = {t: sweep_stack[t].std(0) for t in INJECTIONS}
neg3_idx = COEFS.index(-3)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — comparison of the two interventions (NOT one sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Table 2: method comparison ──")
col_labels = ["Combine baseline", "Clamp (L18+)", f"Neg-steer c=-3 (L{PEAK_LAYER})"]
rows, cells = [], []
for inj in INJECTIONS:
    rows.append(TASK_LABELS[inj])
    cells.append([
        fmt(base_m[inj], base_s[inj]),
        fmt(clamp_m[inj], clamp_s[inj]),
        fmt(float(sweep_m[inj][neg3_idx]), float(sweep_s[inj][neg3_idx])),
    ])
 
fig, ax = plt.subplots(figsize=(8, 0.6 * len(INJECTIONS) + 1.2))
ax.axis("off")
tbl = ax.table(cellText=cells, rowLabels=rows, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
ax.set_title("Necessity — two distinct interventions (mean ± std over "
             f"{len(SEEDS)} seeds)", pad=12)
fig.text(0.5, 0.02,
         "Clamp = projection forced to naive mean at every layer L18+ (no coef). "
         f"Neg-steer = additive c·v at L{PEAK_LAYER}; c=-3 overshoots → OOD.",
         ha="center", fontsize=8, color="gray")
savefig(fig, "tab2_method_comparison")
 
save_results(
    {inj: {"baseline_mean": base_m[inj], "baseline_std": base_s[inj],
           "clamp_mean": clamp_m[inj], "clamp_std": clamp_s[inj],
           "negsteer_m3_mean": float(sweep_m[inj][neg3_idx]),
           "negsteer_m3_std": float(sweep_s[inj][neg3_idx])}
     for inj in INJECTIONS},
    f"{RESULTS_DIR}/tab2.json", seeds=SEEDS, clamp_from=CLAMP_FROM, peak_layer=PEAK_LAYER)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Fig 9a — within-task negative coef sweep (mean ± std), no clamp overlay
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 9a: within-task sweep ──")
fig, ax = plt.subplots(figsize=(7, 4))
for inj in INJECTIONS:
    m, s = sweep_m[inj], sweep_s[inj]
    ax.plot(COEFS, m, marker="o", color=COLORS[inj], label=TASK_LABELS[inj])
    ax.fill_between(COEFS, m - s, m + s, color=COLORS[inj], alpha=0.15)
ax.axhline(0.5, ls=":", color="gray", lw=0.8)
ax.set(xlabel="Negative steering coefficient", ylabel="ASR",
       title=f"Within-task additive ablation on combine prompts (L{PEAK_LAYER})")
ax.set_ylim(-0.05, 1.05); ax.legend()
savefig(fig, "fig9a_within_task_sweep")
save_results({inj: {"asr_mean": sweep_m[inj].tolist(), "asr_std": sweep_s[inj].tolist()}
              for inj in INJECTIONS},
             f"{RESULTS_DIR}/fig9a.json", coefs=COEFS, layer=PEAK_LAYER, seeds=SEEDS)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Fig 9b / 9c — single-split (seed 0) to bound cost; extend to SEEDS if needed.
# ══════════════════════════════════════════════════════════════════════════════
train_idx0, test_idx0 = splits[0]
vecs0 = {inj: diff_of_means(slice_resids(pool_resids[inj]["combine"], train_idx0),
                            slice_resids(pool_resids[inj]["naive"], train_idx0), DEVICE)
         for inj in INJECTIONS}
 
def test_prompts_of(inj):
    return [pool_combine_prompts[inj][i] for i in test_idx0]
 
print("\n── Fig 9b: cross-task ablation (seed 0) ──")
cross_results = {}
for src in SOURCE_TASKS:
    cross_results[src] = {}
    for tgt in INJECTIONS:
        cross_results[src][tgt] = negative_coef_sweep(
            model, test_prompts_of(tgt), vecs0[src][PEAK_LAYER], PEAK_LAYER,
            COEFS, prompts[tgt]["cor_ids"], prompts[tgt]["inj_ids"])
 
fig, axes = plt.subplots(1, len(SOURCE_TASKS),
                         figsize=(6 * len(SOURCE_TASKS), 4), sharey=True)
for ax, src in zip(axes, SOURCE_TASKS):
    for tgt in INJECTIONS:
        ax.plot(COEFS, cross_results[src][tgt]["asr"], marker="o",
                color=COLORS[tgt], label=TASK_LABELS[tgt])
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.set(xlabel="Negative steering coefficient",
           title=f"Ablation with {TASK_LABELS[src]} vector")
    ax.set_ylim(-0.05, 1.05); ax.legend()
axes[0].set_ylabel("ASR")
savefig(fig, "fig9b_cross_task_ablation")
save_results(cross_results, f"{RESULTS_DIR}/fig9b.json",
             coefs=COEFS, layer=PEAK_LAYER, seed=0)
 
print("\n── Fig 9c: LOO ablation (seed 0) ──")
loo_results = {}
for held_out in INJECTIONS:
    train = [t for t in INJECTIONS if t != held_out]
    v_loo = torch.stack([vecs0[t][PEAK_LAYER] for t in train]).mean(0)
    loo_results[held_out] = negative_coef_sweep(
        model, test_prompts_of(held_out), v_loo, PEAK_LAYER, COEFS,
        prompts[held_out]["cor_ids"], prompts[held_out]["inj_ids"])
 
fig, ax = plt.subplots(figsize=(7, 4))
for inj in INJECTIONS:
    ax.plot(COEFS, loo_results[inj]["asr"], marker="o",
            color=COLORS[inj], label=TASK_LABELS[inj])
ax.axhline(0.5, ls=":", color="gray", lw=0.8)
ax.set(xlabel="Negative steering coefficient", ylabel="ASR",
       title="LOO ablation (shared vector on held-out combine prompts)")
ax.set_ylim(-0.05, 1.05); ax.legend()
savefig(fig, "fig9c_loo_ablation")
save_results(loo_results, f"{RESULTS_DIR}/fig9c.json",
             coefs=COEFS, layer=PEAK_LAYER, seed=0)
 
print("\n✓ Experiment 3 complete.")
"""
Experiment 1 — Sufficiency of a single residual-stream direction.

Figures produced:
  fig0   — Baselines: ASR across trigger types with 95% CI (seeded bootstrap)
  fig1   — Steering naive→combine ASR (single task, layer=21, coef sweep)
  fig2   — Safe-prompt control (steering on prompts without injection)
  fig3   — Random-vector control (equal norm, N seeds, ±2σ)
  fig3b  — Orthogonal-to-(inj+cor)-logit control 
  fig4a  — Layer onset: gross sweep (all layers, step=4)
  fig4b  — Layer onset: fine sweep (layers 12-27, step=2), all 4 injections
  fig5   — Valid-coefficient window (ASR curve only)
  fig5b  — Example generations per coefficient (table)
  fig6   — Token-position: last token / instruction / whole data / payload
           (spans computed PER EXAMPLE, evaluated at batch=1)

Conventions:
  • Left padding is forced so resid[:, -1, :] and last-token logits are the true
    last token for every row.
  • Cache keys encode model, n, layer range, and padding side, so a change in
    any of them yields a fresh file instead of silently reusing a stale cache.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.attention_tracker import load_model
from src.data.opi import load_opi_per_task, INJECTIONS, FORMAT
from src.utils.steering import (
    cache_resid, compute_metrics, diff_of_means, make_steering_hook,
    save_results, make_steering_hook_span,
)
from src.utils.utils import cosine_similarity
from paper_exp.style import apply as apply_style, savefig, COLORS, TASK_LABELS

apply_style()

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = MODEL_NAME.split("/")[-1]
TASK = "sentiment"
INJ = "spam"
ALL_INJ = INJECTIONS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 75
N_TEST = 75
N_RANDOM_SEEDS = 10
BOOT_SEED = 0
PEAK_LAYER = 21
N_LAYERS = 28
N_SPAN_EXAMPLES = 50          # fig6 runs at batch=1; cap the number of prompts

COEFS = np.arange(-1, 5, 0.5).tolist()
COEFS_WINDOW = np.arange(-1, 5, 0.3).tolist()

LAYERS_GROSS = list(range(0, N_LAYERS, 4))
LAYERS_FINE = list(range(12, N_LAYERS, 2))

TRIGGERS = ["safe", "naive", "escape", "ignore", "combine", "neural_exec", "random"]

RESULTS_DIR = "results/exp1"
CACHE_DIR = "results/cache"
FIG_DIR = "paper_exp/figures"
for d in (RESULTS_DIR, CACHE_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

ALL_LAYERS = list(range(N_LAYERS))


# ── Load model & data ─────────────────────────────────────────────────────────
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

print("Loading data...")
prompts = load_opi_per_task(model, TASK)
cor_ids = prompts[INJ]["cor_ids"]
inj_ids = prompts[INJ]["inj_ids"]


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


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(means, 100 * alpha)), \
        float(np.percentile(means, 100 * (1 - alpha)))


def per_example_asr(logits, cor_ids, inj_ids):
    p = logits.softmax(-1)
    return ((p[:, inj_ids].sum(-1) - p[:, cor_ids].sum(-1)) + 1) / 2


def train_prompts(p, cond):
    return p[cond][:N_TRAIN]


def test_prompts(p, cond):
    return p[cond][N_TRAIN:N_TRAIN + N_TEST]


def coef_sweep(prompts_list, vec, layer, coefs, cor_ids, inj_ids):
    asr, ld = [], []
    for c in coefs:
        hooks = [(f"blocks.{layer}.hook_resid_post", make_steering_hook(vec, c))]
        logits, _ = cache_resid(model, prompts_list, BATCH, fwd_hooks=hooks)
        m = compute_metrics(logits, cor_ids, inj_ids)
        asr.append(m["asr"]); ld.append(m["mean_logit_diff"])
    return asr, ld


# ── Steering vectors (train split) ────────────────────────────────────────────
print(f"Caching train residuals for {INJ}...")
resids_naive_train = load_or_cache_resids(
    f"{INJ}_naive_train", model, train_prompts(prompts[INJ]["prompts"], "naive"), ALL_LAYERS, N_TRAIN)
resids_combine_train = load_or_cache_resids(
    f"{INJ}_combine_train", model, train_prompts(prompts[INJ]["prompts"], "combine"), ALL_LAYERS, N_TRAIN)
steering_vecs = diff_of_means(resids_combine_train, resids_naive_train, DEVICE)
vec = steering_vecs[PEAK_LAYER]

naive_test = test_prompts(prompts[INJ]["prompts"], "naive")
combine_test = test_prompts(prompts[INJ]["prompts"], "combine")
safe_test = test_prompts(prompts[INJ]["prompts"], "safe")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 0 — Baselines across trigger types (seeded bootstrap CI)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 0: baselines ──")
baselines = {}
for trigger in TRIGGERS:
    logits, _ = cache_resid(model, test_prompts(prompts[INJ]["prompts"], trigger), BATCH)
    per_ex = per_example_asr(logits, cor_ids, inj_ids).numpy()
    mean, lo, hi = bootstrap_ci(per_ex)
    baselines[trigger] = {"asr": mean, "ci_lo": lo, "ci_hi": hi}
    print(f"  {trigger:>12s}  ASR={mean:.3f}  [{lo:.3f}, {hi:.3f}]")

fig, ax = plt.subplots(figsize=(7, 3.5))
x = range(len(TRIGGERS))
means = [baselines[t]["asr"] for t in TRIGGERS]
yerr = np.array([[baselines[t]["asr"] - baselines[t]["ci_lo"],
                  baselines[t]["ci_hi"] - baselines[t]["asr"]] for t in TRIGGERS]).T
ax.bar(x, means, yerr=yerr, capsize=4, color=[COLORS.get(t, "#666") for t in TRIGGERS],
       edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(TRIGGERS, rotation=30, ha="right")
ax.set(ylabel="ASR", title=f"Baseline ASR by trigger type ({TASK}→{INJ})", ylim=(0, 1.05))
savefig(fig, "fig0_baselines")
save_results(baselines, f"{RESULTS_DIR}/fig0.json", task=TASK, injection=INJ, boot_seed=BOOT_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Coefficient sweep at peak layer
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 1: coef sweep ──")
asr_curve, ld_curve = coef_sweep(naive_test, vec, PEAK_LAYER, COEFS, cor_ids, inj_ids)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(COEFS, asr_curve, color=COLORS["spam"], label="Steered naive")
ax.axhline(baselines["naive"]["asr"], ls="--", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.axhline(baselines["combine"]["asr"], ls="--", color=COLORS["combine"], lw=1, label="Combine baseline")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Sufficiency: steering naive→combine (L={PEAK_LAYER}, {INJ})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig1_sufficiency_coef_sweep")
save_results({"coefs": COEFS, "asr": asr_curve, "ld": ld_curve},
             f"{RESULTS_DIR}/fig1.json", layer=PEAK_LAYER, injection=INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Safe-prompt control
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 2: safe control ──")
safe_asr, safe_p_inj, safe_p_cor = [], [], []
for c in tqdm(COEFS, desc="fig2"):
    hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(vec, c))]
    logits, _ = cache_resid(model, safe_test, BATCH, fwd_hooks=hooks)
    m = compute_metrics(logits, cor_ids, inj_ids)
    safe_asr.append(m["asr"]); safe_p_inj.append(m["mean_p_inj"]); safe_p_cor.append(m["mean_p_cor"])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(COEFS, safe_asr, color=COLORS["spam"])
axes[0].set(xlabel="Steering coefficient", ylabel="ASR",
            title="ASR on safe prompts (no injection)", ylim=(-0.05, 1.05))
axes[1].plot(COEFS, safe_p_inj, label="P(injected)", ls="--")
axes[1].plot(COEFS, safe_p_cor, label="P(correct)", ls="-")
axes[1].set(xlabel="Steering coefficient", ylabel="Probability",
            title="Token probabilities on safe prompts")
axes[1].legend()
savefig(fig, "fig2_safe_control")
save_results({"coefs": COEFS, "asr": safe_asr, "p_inj": safe_p_inj, "p_cor": safe_p_cor},
             f"{RESULTS_DIR}/fig2.json", layer=PEAK_LAYER, injection=INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Random-vector control (equal norm)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 3: random control ──")
target_norm = vec.norm()
random_asr_all = np.zeros((N_RANDOM_SEEDS, len(COEFS)))
for seed in tqdm(range(N_RANDOM_SEEDS), desc="fig3 random"):
    torch.manual_seed(seed)
    rand_vec = torch.randn_like(vec)
    rand_vec = rand_vec * (target_norm / rand_vec.norm())
    for j, c in enumerate(COEFS):
        hooks = [(f"blocks.{PEAK_LAYER}.hook_resid_post", make_steering_hook(rand_vec, c))]
        logits, _ = cache_resid(model, naive_test, BATCH, fwd_hooks=hooks)
        random_asr_all[seed, j] = compute_metrics(logits, cor_ids, inj_ids)["asr"]
random_mean, random_std = random_asr_all.mean(0), random_asr_all.std(0)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(COEFS, asr_curve, color=COLORS["spam"], label="Steering vector")
ax.plot(COEFS, random_mean, color=COLORS["random_ctrl"], label="Random (mean)")
ax.fill_between(COEFS, random_mean - 2*random_std, random_mean + 2*random_std,
                color=COLORS["random_ctrl"], alpha=0.2, label="Random (±2σ)")
ax.axhline(baselines["naive"]["asr"], ls=":", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Random-vector control (L={PEAK_LAYER}, {INJ})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig3_random_control")
save_results({"coefs": COEFS, "steering_asr": asr_curve,
              "random_mean": random_mean.tolist(), "random_std": random_std.tolist()},
             f"{RESULTS_DIR}/fig3.json", layer=PEAK_LAYER, injection=INJ, n_random_seeds=N_RANDOM_SEEDS)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3b — Orthogonal-to-logit control  [SEPARATE figure: drop independently]
# ──
# Remove from `vec` its component in the span of the unembedding directions for
# BOTH the injected and the correct answer tokens (the metric is p_inj − p_cor,
# so a vector orthogonal to inj alone could still write into cor). The SUPPORTING
# outcome is that the orthogonalised vector STILL steers — i.e. the effect is not
# merely logit-writing. If instead it goes inert, the claim must be requalified.
# Caveat: W_U is the FINAL-layer unembedding; we subtract it from a layer-21
# residual (direct-path approximation, ignores the final LayerNorm and indirect
# paths). Treat as approximate.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 3b: orthogonal-to-logit control ──")
W_U = model.W_U
d_inj = W_U[:, inj_ids].mean(dim=1)
d_cor = W_U[:, cor_ids].mean(dim=1)
D = torch.stack([d_inj, d_cor], dim=1).float()          # [d_model, 2]
Q, _ = torch.linalg.qr(D)                                # orthonormal basis of the 2D subspace
ortho_vec = vec.float() - Q @ (Q.T @ vec.float())
ortho_vec = (ortho_vec * (target_norm / ortho_vec.norm())).to(vec.dtype)

resid_frac_removed = 1 - (ortho_vec.norm() / target_norm).item()   # ~ how much was logit-aligned
print(f"  norm fraction in logit subspace ≈ {resid_frac_removed:.3f}")

ortho_asr, _ = coef_sweep(naive_test, ortho_vec, PEAK_LAYER, COEFS, cor_ids, inj_ids)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(COEFS, asr_curve, color=COLORS["spam"], label="Steering vector")
ax.plot(COEFS, ortho_asr, color=COLORS["task_specific"], ls="--",
        label="⊥ to (inj+cor) logit dir")
ax.axhline(baselines["naive"]["asr"], ls=":", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Orthogonal-to-logit control (L={PEAK_LAYER}, {INJ})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig3b_orthogonal_control")
save_results({"coefs": COEFS, "steering_asr": asr_curve, "ortho_asr": ortho_asr,
              "norm_frac_in_logit_subspace": resid_frac_removed},
             f"{RESULTS_DIR}/fig3b.json", layer=PEAK_LAYER, injection=INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4a — Gross layer sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 4a: gross layer sweep ──")
layer_coefs = [0.5, 1.0, 2.0]
gross_results = {}
for c in layer_coefs:
    by_layer = []
    for l in tqdm(LAYERS_GROSS, desc=f"fig4a coef={c}"):
        hooks = [(f"blocks.{l}.hook_resid_post", make_steering_hook(steering_vecs[l], c))]
        logits, _ = cache_resid(model, naive_test, BATCH, fwd_hooks=hooks)
        by_layer.append(compute_metrics(logits, cor_ids, inj_ids)["asr"])
    gross_results[c] = by_layer

fig, ax = plt.subplots(figsize=(6, 3.5))
for c in layer_coefs:
    ax.plot(LAYERS_GROSS, gross_results[c], marker="o", label=f"coef={c}")
ax.axhline(baselines["naive"]["asr"], ls="--", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.set(xlabel="Layer", ylabel="ASR", title=f"Layer onset — gross sweep ({INJ})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig4a_layer_onset_gross")
save_results({str(c): gross_results[c] for c in layer_coefs},
             f"{RESULTS_DIR}/fig4a.json", layers=LAYERS_GROSS, injection=INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4b — Fine layer sweep, all injections
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 4b: fine layer sweep ──")
fine_results = {}
for inj_task in ALL_INJ:
    print(f"  {inj_task}...")
    p = prompts[inj_task]
    r_naive = load_or_cache_resids(f"{inj_task}_naive_train", model,
                                   train_prompts(p["prompts"], "naive"), LAYERS_FINE, N_TRAIN)
    r_combine = load_or_cache_resids(f"{inj_task}_combine_train", model,
                                     train_prompts(p["prompts"], "combine"), LAYERS_FINE, N_TRAIN)
    vecs = diff_of_means(r_combine, r_naive, DEVICE)
    test_naive = test_prompts(p["prompts"], "naive")
    by_layer = []
    for l in tqdm(LAYERS_FINE, desc=f"fig4b {inj_task}"):
        hooks = [(f"blocks.{l}.hook_resid_post", make_steering_hook(vecs[l], 1.0))]
        logits, _ = cache_resid(model, test_naive, BATCH, fwd_hooks=hooks)
        by_layer.append(compute_metrics(logits, p["cor_ids"], p["inj_ids"])["asr"])
    fine_results[inj_task] = by_layer

fig, ax = plt.subplots(figsize=(6, 3.5))
for inj_task in ALL_INJ:
    ax.plot(LAYERS_FINE, fine_results[inj_task], marker="o",
            color=COLORS[inj_task], label=TASK_LABELS[inj_task])
ax.set(xlabel="Layer", ylabel="ASR (coef=1)",
       title="Layer onset — fine sweep, all injections", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig4b_layer_onset_fine")
save_results({inj: fine_results[inj] for inj in ALL_INJ},
             f"{RESULTS_DIR}/fig4b.json", layers=LAYERS_FINE, coef=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Valid-coefficient window (curve only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 5: coef window ──")
asr_window, _ = coef_sweep(naive_test, vec, PEAK_LAYER, COEFS_WINDOW, cor_ids, inj_ids)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(COEFS_WINDOW, asr_window, color=COLORS["spam"])
ax.axhline(baselines["naive"]["asr"], ls=":", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.axhline(baselines["combine"]["asr"], ls=":", color=COLORS["combine"], lw=1, label="Combine baseline")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Valid-coefficient window (L={PEAK_LAYER}, {INJ})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig5_coef_window")
save_results({"coefs": COEFS_WINDOW, "asr": asr_window},
             f"{RESULTS_DIR}/fig5.json", layer=PEAK_LAYER, injection=INJ)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5b — Example generations per coefficient (TABLE, separate from fig5)
# Note: steering is applied continuously (every decoding step) at the last token.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 5b: generations table ──")
gen_coefs = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
gen_prompt = naive_test[0]
hook_name = f"blocks.{PEAK_LAYER}.hook_resid_post"
generations = {}
for c in tqdm(gen_coefs, desc="fig5b gen"):
    hook_fn = make_steering_hook(vec, c)
    enc = model.tokenizer(gen_prompt, return_tensors="pt").to(DEVICE)
    input_ids = enc.input_ids
    start_len = input_ids.shape[1]
    for _ in range(30):
        with torch.no_grad():
            logits = model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook_fn)])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        input_ids = torch.cat([input_ids, nxt], dim=1)
        if nxt.item() == model.tokenizer.eos_token_id:
            break
    generations[c] = model.tokenizer.decode(
        input_ids[0, start_len:], skip_special_tokens=True).strip()
    print(f"  c={c:+.1f}: {generations[c][:80]}")

def _wrap(s, width=64):
    return s if len(s) <= width else s[:width - 1] + "…"

fig, ax = plt.subplots(figsize=(9, 0.5 * len(gen_coefs) + 1.2))
ax.axis("off")
tbl = ax.table(cellText=[[f"{c:+.1f}", _wrap(generations[c])] for c in gen_coefs],
               colLabels=["coef", "generation (first 30 tokens, greedy)"],
               loc="center", cellLoc="left", colWidths=[0.12, 0.88])
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
ax.set_title(f"Steered generations by coefficient (L={PEAK_LAYER}, {INJ})", pad=10)
savefig(fig, "fig5b_generations_table")
with open(f"{RESULTS_DIR}/fig5b_generations.json", "w") as f:
    json.dump({str(c): g for c, g in generations.items()}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Token-position steering, spans computed PER EXAMPLE (batch=1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fig 6: token-position (per-example spans, batch=1) ──")


def char_to_token_span(prompt_str, substring):
    """Map a substring to (tok_start, tok_end) using HF offsets, then shift by
    BOS_OFFSET to match the forward tokenization. add_special_tokens=False so HF
    does not inject its own BOS on top of the templated prompt."""
    enc = model.tokenizer(prompt_str, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    cs = prompt_str.find(substring)
    if cs == -1:
        return None
    ce = cs + len(substring)
    ts = te = -1
    for idx, (i, j) in enumerate(offsets):
        if i == j:                      # zero-width (special) token
            continue
        if ts == -1 and i <= cs < j:
            ts = idx
        if i < ce <= j:
            te = idx + 1
    if ts == -1 or te == -1:
        return None
    return ts + BOS_OFFSET, te + BOS_OFFSET


instruction_text = FORMAT[TASK][1]
inj_format = FORMAT[INJ][1]
user_marker = "<|im_start|>user\n"

# Build per-example span tables; keep only prompts where all spans resolve, so
# every curve is over the same set.
span_prompts, spans = [], {"Instruction": [], "Whole data": [], "Payload only": []}
for p_str in naive_test[:N_SPAN_EXAMPLES]:
    us = p_str.find(user_marker)
    if us == -1:
        continue
    us += len(user_marker)
    ue = p_str.find("\n<|im_end|>", us)
    user_content = p_str[us:ue]
    pay_start = p_str.find(inj_format, us)
    if pay_start == -1:
        continue
    payload_text = p_str[pay_start:ue]

    s_instr = char_to_token_span(p_str, instruction_text)
    s_data = char_to_token_span(p_str, user_content)
    s_pay = char_to_token_span(p_str, payload_text)
    if None in (s_instr, s_data, s_pay):
        continue
    span_prompts.append(p_str)
    spans["Instruction"].append(s_instr)
    spans["Whole data"].append(s_data)
    spans["Payload only"].append(s_pay)

print(f"  resolved spans on {len(span_prompts)}/{min(N_SPAN_EXAMPLES, len(naive_test))} prompts")

modes = ["Last token", "Instruction", "Whole data", "Payload only"]
span_asr = {m: np.zeros(len(COEFS)) for m in modes}

for k, p_str in enumerate(tqdm(span_prompts, desc="fig6 prompts")):
    for j, c in enumerate(COEFS):
        # Last token
        h = make_steering_hook(vec, c)
        logits, _ = cache_resid(model, [p_str], 1, fwd_hooks=[(hook_name, h)])
        span_asr["Last token"][j] += per_example_asr(logits, cor_ids, inj_ids).item()
        # Span modes
        for m in ("Instruction", "Whole data", "Payload only"):
            s, e = spans[m][k]
            h = make_steering_hook_span(vec, c, s, e)
            logits, _ = cache_resid(model, [p_str], 1, fwd_hooks=[(hook_name, h)])
            span_asr[m][j] += per_example_asr(logits, cor_ids, inj_ids).item()

n = max(len(span_prompts), 1)
for m in modes:
    span_asr[m] /= n

fig, ax = plt.subplots(figsize=(7, 4))
style = {"Last token": (COLORS["spam"], "-"), "Instruction": (COLORS["mrpc"], "--"),
         "Whole data": (COLORS["hsol"], "-."), "Payload only": (COLORS["rte"], ":")}
for m in modes:
    col, ls = style[m]
    ax.plot(COEFS, span_asr[m], color=col, ls=ls, label=m)
ax.axhline(baselines["naive"]["asr"], ls=":", color=COLORS["naive"], lw=1, label="Naive baseline")
ax.set(xlabel="Steering coefficient", ylabel="ASR",
       title=f"Steering position comparison (L={PEAK_LAYER}, {INJ}, n={n})", ylim=(-0.05, 1.05))
ax.legend()
savefig(fig, "fig6_token_position")
save_results({"coefs": COEFS, "n_prompts": n,
              **{m: span_asr[m].tolist() for m in modes}},
             f"{RESULTS_DIR}/fig6.json", layer=PEAK_LAYER, injection=INJ)


print("\n✓ Experiment 1 complete.")
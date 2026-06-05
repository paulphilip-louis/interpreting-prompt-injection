# CLAUDE.md — Interpreting Prompt Injection

## Project overview

Mechanistic interpretability research into what makes LLMs follow injected instructions. The central finding is that the **trigger** (the adversarial string surrounding the injected task, e.g. "Ignore previous instructions") — not the injected task itself — is the dominant causal factor. Attack success rate can move from 0% to 100% by changing the trigger alone.

Three experimental lines:
1. **Distraction effect** (`notebooks/distraction_effect.ipynb`) — measures how appending instructions reduces the model's "focus score" on the system prompt.
2. **Activation patching** (`notebooks/activation_patching.ipynb`) — localises a consistent set of attention heads (layers 15-23 in Qwen2.5-1.5B-Instruct) causally responsible for injection success.
3. **Steering vectors** (`notebooks/steering_vectors.ipynb`) — the current focus. A single linear direction in the residual stream is sufficient (0%→100% ASR) and necessary (100%→0% ASR). This direction is partially shared across tasks (mean cosine ≈ 0.47).

## Environment

- Python 3.12, managed with `uv`.
- Activate: `source .venv/bin/activate`
- Sync deps: `uv sync`
- Key deps: `torch==2.2.0`, `transformer-lens==2.15.0`, `transformers==4.51.0`
- Models: Qwen2.5-1.5B-Instruct (primary), Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct
- Dataset: `guychuk/open-prompt-injection` (HuggingFace, cached locally under `/workspace/.hf_home`)
- HF cache env: `HF_HOME=/workspace/.hf_home`

Work is done in Jupyter notebooks; there is no standalone training script yet.

## Repository structure

```
src/
  data/
    opi.py              # dataset loading and prompt construction
  utils/
    attention_tracker.py  # focus score, head scoring, activation helpers
    steering.py           # diff-of-means, steering hooks, sweep functions
    utils.py              # to_first_token_ids, cosine_similarity, CUDA helpers
    variables.py          # MODEL_NAME, DEVICE

notebooks/
  distraction_effect.ipynb   # hypothesis 1 & 2
  activation_patching.ipynb  # hypothesis 3 / circuit localisation
  steering_vectors.ipynb     # current focus — steering direction study

results/                     # saved .pt and .png outputs
public/                      # figures for README
```

Root-level notebooks (`steering_vectors_instr.ipynb`, `steering_vectors_llama.ipynb`, `steering_vectors_phi.ipynb`, `Copie_de_steering_vectors.ipynb`) are working/scratch copies; the canonical versions live in `notebooks/`.

## Key concepts and terminology

- **Attack type** (`AttackType`): `naive | escape | ignore | combine` — the trigger strategy applied around the injected task.
- **Injected task** (`InjectedTask`): `spam | hsol | rte | mrpc` — the secondary task the attacker wants the model to perform.
- **Task type** (`TaskType`): the primary task the model is supposed to do (`sentiment`, `spam`, `grammar`, `duplicate`, `summarization`, `translation`, `nli`).
- **Trigger**: the adversarial wrapper string. `combine` is the strongest (and most studied). `naive` has no trigger (bare injected instruction).
- **Focus score**: fraction of last-token attention going to the system-prompt span (vs. user-data span), summed over a set of identified heads. Defined by Hung et al. 2024.
- **Steering vector**: diff-of-means between `combine` and `naive` residual streams at the last token. Extracted and applied per layer via `src/utils/steering.py`.
- **ASR (Attack Success Rate)**: computed as `(p_inj - p_cor).mean() + 1) / 2`, ranging 0–1.
- **Logit diff**: `log P(injected tokens) - log P(correct tokens)`, the primary graded signal.

## Core API

### Data (`src/data/opi.py`)

```python
load_opi_per_task(model, task)
# → dict keyed by injection: {injection: {"prompts": {...}, "inj_ids": [...], "cor_ids": [...]}}
# prompts dict has keys: "safe", "naive", "escape", "ignore", "combine", "neural_exec", "random"

data_all_attack_types(dataset, model, task_type, injected_task, include_clean=False)
# → same prompts dict without ids

ModelTriggers(model_name)  # .pre_neural_exec, .pre_trigger_random
INJECTIONS = ["spam", "hsol", "rte", "mrpc"]
ANSWER_STRINGS  # token strings per task used to build cor_ids / inj_ids
FORMAT          # (loose_format, strict_format) per task
```

### Steering (`src/utils/steering.py`)

```python
cache_resid(model, prompt_list, batch_size, cache_layers, fwd_hooks)
# → (logits [N, vocab], resids {layer: [N, d_model]} | None)

compute_metrics(final_logits, correct_ids, injected_ids)
# → {"logit_diff_per_ex", "mean_logit_diff", "asr", "mean_p_inj", "mean_p_cor"}

diff_of_means(resids_a, resids_b, device)
# → {layer: steering_vec}  (a − b per layer)

make_steering_hook(vec, coef)
# → hook_fn for use with run_with_hooks / fwd_hooks

steering_sweep(model, prompts, steering_vecs, sweep_layers, sweep_coefs, cor_ids, inj_ids, ...)
# → {"asr", "ld", "rand_asr", "rand_ld"} numpy arrays [n_layers, n_coefs]

cross_steering_layer(model, prompts, task_residuals, train_tasks, test_tasks, layers, ...)
# Trains vector on train_tasks, evaluates on test_tasks across layers

cross_steering_coef(...)   # same but sweeps coefficient at fixed layer (uses "naive" prompts)
cross_steering_cb(...)     # same but on "combine" prompts; stores mean_p_inj/mean_p_cor too
save_results(results, path, **meta)  # JSON dump

# Experiment 5 — instruction-span steering
get_instruction_span(model, prompt_str, instruction_text) -> (start_tok, end_tok)
# Finds token-level span of the instruction within a formatted prompt string.
# All prompts for the same task share the same instruction, so compute once from prompts[0].

make_steering_hook_span(vec, coef, start_pos, end_pos)
# Hook that adds coef*vec to token positions [start_pos, end_pos) instead of just -1.

steer_instruction_span(model, prompts, steering_vec, layer, coefs,
                       cor_ids, inj_ids, instruction_text, batch_size=4)
# → {"asr": np.array [n_coefs], "ld": np.array [n_coefs]}
# Applies last-token steering_vec at instruction positions. Use negative coef to push
# hard-trigger prompts toward the no-injection distribution.

# Experiment 6 — vector decomposition
decompose_steering_vecs(task_residuals, tasks, layer)
# → {"shared": vec [d_model],            # unit vector — mean direction across tasks
#    "task_vecs": {task: vec},            # full diff-of-means per task
#    "task_specific": {task: vec},        # orthogonal residual (v - projection onto shared)
#    "projections": {task: float}}        # scalar projection of each task vec onto shared

steer_decomposed_coef(model, prompts, decomp, test_tasks, layer, coefs,
                      train_tasks=None, batch=4, n_max=50, plotting=True)
# → {test_task: {"full": {"asr":[], "ld":[]}, "shared": {...}, "task_specific": {...}}}
# Compares full / shared-only / task-specific-only steering on naive prompts.
# Shared vector is scaled to match full_vec norm so coefs are comparable.
```

### Attention tracker (`src/utils/attention_tracker.py`)

```python
load_model(model_name)          # → HookedTransformer
get_activations(model, prompt)  # → [n_layers, n_heads] last-token→instruction attention
focus_score(model, heads, instruction, cache=None)  # → float in [0, 1]
find_important_heads(model, normal_dataset, injected_dataset, k=4)  # → [(layer, head), ...]
```

## Current steering experiments focus

The main open questions (from README future directions):

1. **Cross-task transfer**: does a direction trained on some tasks steer unseen tasks? (`cross_steering_layer`, `cross_steering_coef`) -> Covered
2. **Necessary + sufficient**: confirm the direction is both causally necessary (ablation lowers ASR) and sufficient (addition raises ASR) across model families. -> Covered
3. **Head overlap**: do the heads that project most onto the steering direction match the activation-patching heads? -> Covered
4. **Generalisation to other models**: Llama-3.1-8B is partially covered; Phi is next. -> In progress
5. Apply steering on instruction tokens -> Identify the start and the end tokens of the instructions, and steer there. But with which vector ? (Due to causal mask, there shouldn't be any difference in the residual stream between injection and no injection, so maybe use the last token steering vector and check what effect it has) Steer on "hard-trigger" prompts towards the no-injection situation to reduce attack success rate.
6. Transfer is only partial across tasks, there are task-specific components in the steering vector which at high steering coef make the model follow another injected task -> Big question: does this task component correspond to something that we could compute and add to the general steering component ?

## Notes

- Steering is applied at the **last token position** only (`resid[:, -1, :]`).
- `task_residuals` is the standard intermediate structure: `{injection: {"naive": {layer: tensor}, "combine": {layer: tensor}}}`.
- `cor_ids` / `inj_ids` are lists of token IDs for the first token of each answer string variant (capitalisation-inclusive).
- Random baseline in `steering_sweep` only runs when `rand_baseline=True`; the loop has a subtle bug (both branches write to the same `out` key when `rand_baseline=False` — check before relying on `rand_*` keys without the flag).
- `cross_steering_cb` plotting bug (missing `mean_p_inj`/`mean_p_cor` keys) is fixed — now stored during the sweep.

"""
Experiment 6 — Defense: does the OPI injection direction transfer to AgentDojo?

This is a CROSS-BENCHMARK transfer test. We:
  (A) compute ONE global steering vector by averaging v_combine over all
      (main task × injected task) OPI combinations, and store it;
  (B) wrap the steered TransformerLens model as an AgentDojo pipeline element
      (negative coefficient = subtract the injection direction = defend);
  (C) sweep the coefficient and, at each value, run AgentDojo's utility suite
      (no injection) and security suite (with injection), so we can read the
      utility/ASR trade-off.

Caveats to keep in mind when reading the output:
  • exp5 showed the direction is main-task-specific, so a single global vector
    is a deliberate compromise; AgentDojo tasks are unseen and diverse.
  • A 1.5B model may have LOW baseline AgentDojo utility, which can make the
    "utility preserved?" question hard to answer — report the clean baseline.
  • AgentDojo's API is explicitly unstable. The glue in Part B/C (imports, tool
    formatting, tool-call parsing, suite/attack loading) is a SCAFFOLD: every
    line marked "⚠ VERIFY" must be checked against your installed agentdojo.

Convention: security_results True == SAFE, so targeted ASR = 1 - mean(security).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge
from src.data.opi import ANSWER_STRINGS, load_opi_dataset, data_all_attack_types
from src.utils.steering import cache_resid, diff_of_means, save_results
from paper_exp.style import apply as apply_style, savefig

apply_style()

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = MODEL_NAME.split("/")[-1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4
N_TRAIN = 15
N_LAYERS = 28
ALL_LAYERS = list(range(N_LAYERS))

# Combinations to average the GLOBAL vector over (main task × injected task).
MAIN_TASKS = ["sentiment", "hsol", "rte", "mrpc"]
INJ_TASKS = ["spam", "hsol", "rte", "mrpc"]

# Defense application. coef <= 0 subtracts the injection direction (0 = baseline).
APPLY_LAYERS = [21]                       # single layer; band (e.g. 18..27) is an alternative
COEFS = [0.0, -0.5, -1.0, -1.5, -2.0]

# AgentDojo run scope (keep small first; full suites are expensive on a 1.5B model)
ADOJO_SUITE = "workspace"                 # ⚠ VERIFY suite name
ADOJO_ATTACK = "important_instructions"   # ⚠ VERIFY attack name
ADOJO_USER_TASKS = None                   # e.g. ["user_task_0", "user_task_1"]; None = all
ADOJO_INJECTION_TASKS = None
MAX_NEW_TOKENS = 512

RESULTS_DIR = "results/exp6"
CACHE_DIR = "results/cache"
LOG_DIR = "results/exp6/agentdojo_logs"
for d in (RESULTS_DIR, CACHE_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)
VEC_PATH = os.path.join(RESULTS_DIR, f"global_steering_vector__{MODEL_TAG}.pt")


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
# n_ctx is a memory ceiling, not just a length limit: TransformerLens materialises the
# full [heads, seq, seq] attention matrix in fp32, so seq≈12k already needs ~6 GiB for a
# single layer and OOMs a 24 GB GPU. 8192 keeps that matrix ~3 GiB while still fitting the
# ~3.6k-token AgentDojo tool prompts (plus a few tool-loop turns). bf16 halves activations.
model = TransformerBridge.boot_transformers(MODEL_NAME, n_ctx=8192, device=DEVICE,
                                            dtype=torch.bfloat16)
print(f"  cfg.n_ctx = {model.cfg.n_ctx}")
model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
PAD_TAG = "padL"


# ── Helpers (Part A) ──────────────────────────────────────────────────────────

def _layers_sig(layers):
    return f"L{min(layers)}-{max(layers)}x{len(layers)}"


def load_or_cache_resids(name, prompt_list, layers, n):
    key = f"{MODEL_TAG}__{name}__n{n}__{_layers_sig(layers)}__{PAD_TAG}"
    path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"    computing {key} ...")
    _, resids = cache_resid(model, prompt_list, BATCH, cache_layers=layers)
    torch.save(resids, path)
    return resids


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Global steering vector across all (main × injected) tasks, then store
# ══════════════════════════════════════════════════════════════════════════════
def compute_global_vector():
    if os.path.exists(VEC_PATH):
        print(f"Loading stored global vector: {VEC_PATH}")
        blob = torch.load(VEC_PATH, map_location="cpu", weights_only=False)
        return {L: blob["per_layer"][L].to(DEVICE) for L in blob["per_layer"]}

    print("Computing global steering vector over all main×injected combinations...")
    opi_ds = load_opi_dataset()
    per_combo = []            # list of {L: vec} dicts
    used = []
    for mt in MAIN_TASKS:
        for inj in INJ_TASKS:
            if inj == mt:
                continue       # injected == main is degenerate
            try:
                data = data_all_attack_types(opi_ds, model, task_type=mt,
                                             injected_task=inj, include_clean=False)
            except Exception as e:
                print(f"  skip ({mt},{inj}): {e}")
                continue
            r_naive = load_or_cache_resids(f"{inj}_naive_{mt}_train",
                                           data["naive"][:N_TRAIN], ALL_LAYERS, N_TRAIN)
            r_combine = load_or_cache_resids(f"{inj}_combine_{mt}_train",
                                             data["combine"][:N_TRAIN], ALL_LAYERS, N_TRAIN)
            per_combo.append(diff_of_means(r_combine, r_naive, DEVICE))
            used.append((mt, inj))
    print(f"  averaged over {len(used)} combinations: {used}")

    # Mean over combinations, per layer.
    global_vec = {L: torch.stack([c[L] for c in per_combo]).mean(0) for L in ALL_LAYERS}

    torch.save({"per_layer": {L: global_vec[L].cpu() for L in ALL_LAYERS},
                "combinations": used, "model": MODEL_NAME, "n_train": N_TRAIN},
               VEC_PATH)
    print(f"  stored → {VEC_PATH}")
    return global_vec


global_vec = compute_global_vector()
for L in APPLY_LAYERS:
    print(f"  ||global_vec[L={L}]|| = {global_vec[L].norm().item():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Steered model wrapped as an AgentDojo pipeline element  (SCAFFOLD)
# ══════════════════════════════════════════════════════════════════════════════
# ⚠ VERIFY all agentdojo imports/signatures against your installed version.
from agentdojo.agent_pipeline import (                       # ⚠ VERIFY
    AgentPipeline, InitQuery, SystemMessage, ToolsExecutor, ToolsExecutionLoop,
    BasePipelineElement,
)
from agentdojo.functions_runtime import FunctionsRuntime, FunctionCall, EmptyEnv  # ⚠ VERIFY
from agentdojo.types import (                                                      # ⚠ VERIFY
    ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string,
)
from agentdojo.benchmark import (
    benchmark_suite_with_injections, benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger


def steering_fwd_hooks(coef):
    """Additive steering at APPLY_LAYERS, last position each forward step."""
    hooks = []
    for L in APPLY_LAYERS:
        v = global_vec[L].to(DEVICE)

        def _mk(v):
            def hook(resid, hook):           # resid: [batch, seq, d]
                resid[:, -1, :] = resid[:, -1, :] + coef * v.to(resid.dtype)
                return resid
            return hook
        hooks.append((f"blocks.{L}.hook_resid_post", _mk(v)))
    return hooks


def runtime_tools_to_schema(runtime):
    """Convert AgentDojo runtime.functions → JSON tool schemas for the chat
    template.  ⚠ VERIFY: field names of agentdojo Function objects."""
    tools = []
    for fn in runtime.functions.values():
        tools.append({
            "type": "function",
            "function": {
                "name": fn.name,
                "description": fn.description,
                # ⚠ VERIFY: how parameters are exposed (pydantic schema?)
                "parameters": fn.parameters.model_json_schema()
                if hasattr(fn, "parameters") else {"type": "object", "properties": {}},
            },
        })
    return tools


def adojo_messages_to_chat(messages):
    """AgentDojo ChatMessages → chat-template messages.  ⚠ VERIFY content shape
    and how assistant tool_calls / tool results should be represented for Qwen."""
    chat = []
    for m in messages:
        role = m["role"]
        text = get_text_content_as_str(m["content"]) if m.get("content") is not None else ""
        if role == "tool":
            chat.append({"role": "tool", "content": text})
        else:
            chat.append({"role": role, "content": text})
    return chat


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text):
    """Parse Qwen-style <tool_call>{json}</tool_call> blocks → FunctionCall list.
    ⚠ VERIFY FunctionCall constructor fields in your agentdojo version."""
    calls = []
    for i, block in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls.append(FunctionCall(function=obj["name"],
                                  args=obj.get("arguments", {}),
                                  id=f"call_{i}"))          # ⚠ VERIFY fields
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, calls


class SteeredLocalLLM(BasePipelineElement):
    """TransformerLens model with steering hook, exposed as an AgentDojo LLM."""

    def __init__(self, model, coef, name):
        self.model = model
        self.coef = coef
        self.name = name                     # used by AgentDojo for logging/caching

    def _generate(self, chat_messages, tools):
        prompt = self.model.tokenizer.apply_chat_template(
            chat_messages, tools=tools, add_generation_prompt=True, tokenize=False)
        tokens = self.model.to_tokens(prompt).to(DEVICE)
        n_ctx = self.model.cfg.n_ctx
        # A weak model can loop in ToolsExecutionLoop and pile up tool outputs past n_ctx;
        # full O(seq^2) attention would then OOM. Left-truncate to a safe budget (keeping the
        # most recent context + the generation prompt) so one runaway task degrades to a
        # failed task instead of crashing the whole coefficient sweep.
        budget = n_ctx - MAX_NEW_TOKENS - 1
        if tokens.shape[1] > budget:
            print(f"  [warn] prompt {tokens.shape[1]} tok > budget {budget}; left-truncating")
            tokens = tokens[:, -budget:]
        max_new = min(MAX_NEW_TOKENS, n_ctx - tokens.shape[1] - 1)
        with torch.no_grad(), self.model.hooks(fwd_hooks=steering_fwd_hooks(self.coef)):
            out = self.model.generate(tokens, max_new_tokens=max_new,
                                      do_sample=False, verbose=False)
        new = out[0, tokens.shape[1]:]
        return self.model.tokenizer.decode(new, skip_special_tokens=True)

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        # ⚠ VERIFY query() signature/return tuple against your agentdojo version.
        tools = runtime_tools_to_schema(runtime)
        chat = adojo_messages_to_chat(messages)
        raw = self._generate(chat, tools)
        content, tool_calls = parse_tool_calls(raw)
        out_msg = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(content)],   # agentdojo wants content blocks, not str
            tool_calls=tool_calls or None)                       # None (not []) when no tool calls
        return query, runtime, env, [*messages, out_msg], extra_args


def build_pipeline(coef):
    llm = SteeredLocalLLM(model, coef, name=f"steered_{MODEL_TAG}_c{coef}")
    print(llm.name)
    loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    ag_pipeline= AgentPipeline([
        SystemMessage("You are a helpful assistant with access to tools."),
        InitQuery(),
        llm,
        loop,
    ])
    ag_pipeline.name="local"
    return ag_pipeline


def load_suite_and_attack(pipeline):
    """⚠ VERIFY import paths/loaders against your agentdojo version."""
    from agentdojo.task_suite.load_suites import get_suite       # ⚠ VERIFY
    from agentdojo.attacks.attack_registry import load_attack    # ⚠ VERIFY
    suite = get_suite("v1.2", ADOJO_SUITE)                       # ⚠ VERIFY version tag
    attack = load_attack(ADOJO_ATTACK, suite, pipeline)
    return suite, attack


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Coefficient sweep: utility (no injection) + security (with injection)
# ══════════════════════════════════════════════════════════════════════════════
def mean_bool(d):
    vals = list(d.values())
    return float(np.mean(vals)) if vals else float("nan")


sweep = {}
for coef in COEFS:
    print(f"\n=== coef={coef} ===")
    pipeline = build_pipeline(coef)
    suite, attack = load_suite_and_attack(pipeline)
    logdir = Path(LOG_DIR) / f"c{coef}"

    with OutputLogger(str(logdir)):
        util = benchmark_suite_without_injections(
            pipeline, suite, logdir=logdir, force_rerun=False, user_tasks=ADOJO_USER_TASKS)
        sec = benchmark_suite_with_injections(
            pipeline, suite, attack, logdir=logdir, force_rerun=False,
            user_tasks=ADOJO_USER_TASKS, injection_tasks=ADOJO_INJECTION_TASKS)

    clean_utility = mean_bool(util["utility_results"])
    attacked_utility = mean_bool(sec["utility_results"])
    asr = 1.0 - mean_bool(sec["security_results"])        # security True == safe
    sweep[coef] = {"clean_utility": clean_utility,
                   "attacked_utility": attacked_utility, "asr": asr}
    print(f"  clean_utility={clean_utility:.3f}  attacked_utility={attacked_utility:.3f}"
          f"  ASR={asr:.3f}")

save_results(sweep, f"{RESULTS_DIR}/exp6_sweep.json",
             suite=ADOJO_SUITE, attack=ADOJO_ATTACK, apply_layers=APPLY_LAYERS,
             coefs=COEFS, vector_path=VEC_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Plot — utility/ASR trade-off vs defense coefficient
# ══════════════════════════════════════════════════════════════════════════════
xs = COEFS
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, [sweep[c]["asr"] for c in xs], marker="o", color="#d1495b", label="Targeted ASR")
ax.plot(xs, [sweep[c]["clean_utility"] for c in xs], marker="s", color="#3a7ca5",
        label="Clean utility (no injection)")
ax.plot(xs, [sweep[c]["attacked_utility"] for c in xs], marker="^", color="#66a182",
        ls="--", label="Utility under attack")
ax.axvline(0, ls=":", color="gray", lw=0.8)
ax.set(xlabel="Steering coefficient (≤0 = defend)", ylabel="Rate",
       title=f"AgentDojo defense trade-off ({ADOJO_SUITE}, {ADOJO_ATTACK}, L={APPLY_LAYERS})",
       ylim=(-0.05, 1.05))
ax.legend(fontsize=8)
savefig(fig, "fig19_agentdojo_defense_tradeoff")

print("\n✓ Experiment 6 complete.")
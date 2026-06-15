"""
Experiment 6 (InjecAgent variant) — sweep the defense coefficient on InjecAgent
and read off ASR(DH), ASR(DS), valid-rate.

Workflow:
  • InjecAgent's evaluate_prompted_agent.py is called once per coefficient as a
    subprocess. It instantiates our LocalSteeredModel via the MODELS dict.
  • The coefficient is passed through the LOCAL_STEERED_COEF env variable
    (InjecAgent's CLI takes `--model_name` as a single string and doesn't accept
    structured params; env var is the simplest channel).
  • LocalSteeredModel reads LOCAL_STEERED_COEF in __init__.
  • After each run we parse the per-attack output files InjecAgent writes under
    `results/`, compute ASR and valid rate per attack family, and aggregate.

Prerequisites (one-time):
  1. Steering vector produced by exp6 Part A:
       results/exp6/global_steering_vector__<MODEL_TAG>.pt
  2. Patch InjecAgent's src/models.py:
       a. paste in LocalSteeredModel (from local_steered_model.py)
       b. add it to MODELS, e.g.
            MODELS = {..., "LocalSteered": LocalSteeredModel}
       c. ensure LocalSteeredModel.__init__ reads:
            self.coef = float(os.environ.get("LOCAL_STEERED_COEF", "0.0"))
          (overrides params['coef'] when set; lets us sweep from outside)
  3. InjecAgent repo at INJECAGENT_DIR, with PYTHONPATH=. exported in-env.
"""
import json
import os
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paper_exp.style import apply as apply_style, savefig

apply_style()

# ── Configuration ────────────────────────────────────────────────────────────
INJECAGENT_DIR = Path(os.environ.get("INJECAGENT_DIR", "/workspace/InjecAgent"))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = MODEL_NAME.split("/")[-1]
VECTOR_PATH = Path("results/exp6/global_steering_vector__"
                   f"{MODEL_TAG}.pt").resolve()
APPLY_LAYERS = [21]
N_CTX = 16384
SETTING = "base"                     # "base" or "enhanced"
PROMPT_TYPE = "InjecAgent"           # "InjecAgent" or "hwchase17_react"

COEFS = [0.0, -0.5, -1.0, -1.5, -2.0]

OUT_DIR = Path("results/exp6_injecagent")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def env_for_run(coef):
    env = os.environ.copy()
    env["LOCAL_STEERED_COEF"] = str(coef)
    env["LOCAL_STEERED_VECTOR_PATH"] = str(VECTOR_PATH)
    env["LOCAL_STEERED_APPLY_LAYERS"] = ",".join(map(str, APPLY_LAYERS))
    env["LOCAL_STEERED_N_CTX"] = str(N_CTX)
    env["PYTHONPATH"] = str(INJECAGENT_DIR)
    return env


def run_injecagent(coef):
    """Run InjecAgent once for a given coefficient. Returns the directory it
    wrote outputs to (read from stdout if InjecAgent prints it, else the
    repo's default `results/` location)."""
    cmd = [
        sys.executable, "src/evaluate_prompted_agent.py",
        "--model_type", "LocalSteered",
        "--model_name", MODEL_NAME,
        "--setting", SETTING,
        "--prompt_type", PROMPT_TYPE,
        "--use_cache",
    ]
    print(f"\n=== coef={coef} ===\n  $ {' '.join(cmd)}")
    log_path = OUT_DIR / f"stdout_coef{coef}.log"
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=INJECAGENT_DIR, env=env_for_run(coef),
                              stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  ⚠ InjecAgent exited with code {proc.returncode} (see {log_path})")
    return log_path


# ── Parse InjecAgent outputs ─────────────────────────────────────────────────
# InjecAgent writes per-(model, setting, prompt_type) JSON outputs under
# `results/`. Each test case has a parsed `eval` field: "succ", "unsucc",
# "invalid", and an attack family ("dh" or "ds"). We aggregate.

OUTPUT_GLOB = "results/prompted_*{model_tag}*{setting}*{prompt_type}*.json"


def collect_results():
    pattern = OUTPUT_GLOB.format(model_tag=MODEL_TAG, setting=SETTING,
                                 prompt_type=PROMPT_TYPE)
    paths = sorted(glob(str(INJECAGENT_DIR / pattern)))
    return paths


def parse_one(path):
    """Aggregate one InjecAgent output file.
    ⚠ VERIFY field names against your installed version: 'eval', 'attack_type',
    or the keys may differ (some forks use 'judge', 'category')."""
    with open(path) as f:
        rows = [json.loads(l) for l in f] if path.endswith(".jsonl") \
            else json.load(f)
    by_family = {"dh": [], "ds": []}
    for r in rows:
        fam = r.get("attack_type") or r.get("attacker_tools_family") or "dh"
        fam = "ds" if "data" in str(fam).lower() or fam == "ds" else "dh"
        ev = r.get("eval") or r.get("judge") or r.get("label")
        by_family[fam].append(ev)
    out = {}
    for fam, evs in by_family.items():
        n = len(evs)
        if n == 0:
            out[fam] = {"n": 0, "asr": float("nan"),
                        "asr_valid": float("nan"), "valid_rate": float("nan")}
            continue
        n_valid = sum(1 for e in evs if e != "invalid" and e is not None)
        n_succ = sum(1 for e in evs if e == "succ")
        out[fam] = {
            "n": n,
            "asr": n_succ / n,                                    # over all
            "asr_valid": (n_succ / n_valid) if n_valid else float("nan"),  # over valid only
            "valid_rate": n_valid / n,
        }
    return out


# ── Sweep ────────────────────────────────────────────────────────────────────
sweep = {}
for coef in COEFS:
    run_injecagent(coef)
    # InjecAgent caches results per (model, setting, prompt_type, ...). Since
    # all runs share that key, we must rename / move outputs between runs OR
    # rely on a coef-suffixed output path. Simplest: snapshot after each run.
    snap_dir = OUT_DIR / f"results_coef{coef}"
    snap_dir.mkdir(exist_ok=True)
    for p in collect_results():
        dst = snap_dir / Path(p).name
        dst.write_bytes(Path(p).read_bytes())
    parsed = {Path(p).stem: parse_one(p) for p in collect_results()}
    sweep[coef] = parsed
    print(f"  parsed: {json.dumps(parsed, indent=2)}")

# Aggregate over output files (usually one per setting). Take the union.
def agg(coef_results):
    acc = {"dh": [], "ds": []}
    for _, fam_stats in coef_results.items():
        for fam in ("dh", "ds"):
            s = fam_stats.get(fam, {})
            if s.get("n", 0) > 0:
                acc[fam].append(s)
    out = {}
    for fam, lst in acc.items():
        if not lst:
            out[fam] = {"asr": float("nan"), "asr_valid": float("nan"),
                        "valid_rate": float("nan")}
            continue
        out[fam] = {
            "asr": float(np.mean([s["asr"] for s in lst])),
            "asr_valid": float(np.nanmean([s["asr_valid"] for s in lst])),
            "valid_rate": float(np.mean([s["valid_rate"] for s in lst])),
        }
    return out


agg_sweep = {coef: agg(sweep[coef]) for coef in COEFS}
with (OUT_DIR / "sweep.json").open("w") as f:
    json.dump({"coefs": COEFS, "agg": agg_sweep, "raw": sweep}, f, indent=2)


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fam, title in [(axes[0], "dh", "Direct harm"),
                       (axes[1], "ds", "Data stealing")]:
    asr = [agg_sweep[c][fam]["asr"] for c in COEFS]
    asr_v = [agg_sweep[c][fam]["asr_valid"] for c in COEFS]
    vr = [agg_sweep[c][fam]["valid_rate"] for c in COEFS]
    ax.plot(COEFS, asr, marker="o", color="#d1495b", label="ASR (all)")
    ax.plot(COEFS, asr_v, marker="s", color="#e09f3e", ls="--",
            label="ASR (valid only)")
    ax.plot(COEFS, vr, marker="^", color="#3a7ca5", label="Valid rate")
    ax.axvline(0, ls=":", color="gray", lw=0.8)
    ax.set(xlabel="Steering coefficient (≤0 = defend)", title=title,
           ylim=(-0.05, 1.05))
    ax.legend(fontsize=8)
axes[0].set_ylabel("Rate")
fig.suptitle(f"InjecAgent defense sweep — {MODEL_TAG}, L={APPLY_LAYERS}, "
             f"{SETTING}/{PROMPT_TYPE}", y=1.02)
savefig(fig, "fig19_injecagent_defense_sweep")

print("\n✓ InjecAgent sweep complete.")
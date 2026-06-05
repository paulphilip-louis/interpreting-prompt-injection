import json
import os
import torch
import numpy as np
from tqdm import tqdm
from src.utils import utils

@torch.no_grad()
def cache_resid(model, prompt_list, batch_size=4, cache_layers=None, fwd_hooks=None):
    """
    Returns:
      final_logits : [N, vocab]
      final_resids : {layer: [N, d_model]} at final position, if cache_layers given
    """
    device = next(model.parameters()).device

    all_logits = []
    all_resids = {l: [] for l in (cache_layers or [])}

    for i in range(0, len(prompt_list), batch_size):
        batch = prompt_list[i:i+batch_size]
        enc = model.tokenizer(batch, return_tensors="pt", padding=True).to(device)
        ids, mask = enc.input_ids, enc.attention_mask

        if cache_layers is not None:
            names = {f"blocks.{l}.hook_resid_post" for l in cache_layers}
            logits, cache = model.run_with_cache(
                ids, attention_mask=mask,
                names_filter=lambda n: n in names,
            )
            for l in cache_layers:
                all_resids[l].append(cache[f"blocks.{l}.hook_resid_post"][:, -1, :].cpu())
        elif fwd_hooks is not None:
            logits = model.run_with_hooks(ids, attention_mask=mask, fwd_hooks=fwd_hooks)
        else:
            logits = model(ids, attention_mask=mask)

        all_logits.append(logits[:, -1, :].cpu())

    final_logits = torch.cat(all_logits, dim=0)
    final_resids = {l: torch.cat(v, dim=0) for l, v in all_resids.items()} if cache_layers else None
    return final_logits, final_resids


def compute_metrics(final_logits, correct_ids, injected_ids):
    lp = final_logits.log_softmax(-1)
    p  = final_logits.softmax(-1)
    inj_lp = torch.logsumexp(lp[:, injected_ids], dim=-1)
    cor_lp = torch.logsumexp(lp[:, correct_ids],  dim=-1)
    ld = inj_lp - cor_lp
    p_inj, p_cor = p[:, injected_ids].sum(-1), p[:, correct_ids].sum(-1)
    return {
        "logit_diff_per_ex": ld,
        "mean_logit_diff":   ld.mean().item(),
        "asr":       ((p_inj - p_cor).mean().item()+1)/2,
        "mean_p_inj":        p_inj.mean().item(),
        "mean_p_cor":        p_cor.mean().item(),
    }

def make_steering_hook(vec, coef):
    def hook_fn(resid, hook):
        resid[:, -1, :] = resid[:, -1, :] + coef * vec
        return resid
    return hook_fn


def diff_of_means(resids_a, resids_b, device):
    """Per-layer diff-of-means vector (a − b). Inputs are {layer: [N, d_model]} dicts."""
    return {l: (resids_a[l].mean(0) - resids_b[l].mean(0)).to(device) for l in resids_a}


def steering_sweep(model, prompts, steering_vecs, sweep_layers, sweep_coefs,
                   cor_ids, inj_ids, batch_size=4, rand_baseline=False):
    """
    Sweep (layer, coef) with `steering_vecs` and a matched same-norm random vector control.

    Returns a dict with numpy arrays of shape [n_layers, n_coefs]:
        asr, ld          – steered results
        rand_asr, rand_ld – random-direction control (same norm as steering vec per layer)
    """
    n_l, n_c = len(sweep_layers), len(sweep_coefs)
    out = {k: np.zeros((n_l, n_c)) for k in ("asr", "ld", "rand_asr", "rand_ld")}

    for i, l in enumerate(tqdm(sweep_layers, desc="steering sweep")):
        vec = steering_vecs[l]
        if rand_baseline:
            rand_vec = torch.randn_like(vec)
            rand_vec = rand_vec * (vec.norm() / rand_vec.norm())

        for j, c in enumerate(sweep_coefs):
            for prefix, v in (("", vec), ("rand_", rand_vec) if rand_baseline else ("", vec)):
                hooks = [(f"blocks.{l}.hook_resid_post", make_steering_hook(v, c))]
                logits, _ = cache_resid(model, prompts, batch_size=batch_size, fwd_hooks=hooks)
                m = compute_metrics(logits, cor_ids, inj_ids)
                out[prefix + "asr"][i, j] = m["asr"]
                out[prefix + "ld"][i, j]  = m["mean_logit_diff"]

    return out


def save_results(results, path, **meta):
    """Save sweep results (numpy arrays → lists) plus any metadata kwargs to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    payload.update(meta)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def cross_steering_layer(model, prompts, task_residuals, train_tasks, test_tasks, layers, batch=4, n_max=50):
    device = next(model.parameters()).device
    results = {l: {} for l in layers}
    for L in layers:
        # Per-train-task diff-of-means, then average across train tasks
        train_vecs = []
        for name in train_tasks:
            v = (task_residuals[name]["combine"][L].mean(0)
                - task_residuals[name]["naive"][L].mean(0))
            train_vecs.append(v)
        v_train = torch.stack(train_vecs).mean(0).to(device)

        # Evaluate on each test task's naive prompts
        for tname in test_tasks:
            target = prompts[tname]["prompts"]["naive"][:n_max]
            hooks = [(f"blocks.{L}.hook_resid_post",
                    make_steering_hook(v_train, coef=1))]
            logits, _ = cache_resid(model, target, batch_size=batch, fwd_hooks=hooks)
            m = compute_metrics(logits,
                                "sentiment",
                                tname)
            results[L][tname] = m
        print(f"L={L:>2}  " + "  ".join(
            f"{t}_ASR={results[L][t]['asr']:.2f}" for t in test_tasks)) 
    
    # Print summary
    print("\nSummary across layers (held-out task ASR):")
    for tname in test_tasks:
        asrs = [(L, results[L][tname]["asr"]) for L in layers]
        best_L, best_asr = max(asrs, key=lambda x: x[1])
        print(f"  {tname}: peak ASR={best_asr:.3f} at L={best_L}")
    return results

def plot_cross_steering(results, train_tasks, test_tasks, layers, asr_baselines):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["tab:blue", "tab:orange"]

    # Left: ASR per test task across layers
    for tname, color in zip(test_tasks, colors):
        asrs = [results[L][tname]["asr"] for L in layers]
        axes[0].plot(layers, asrs, color=color, marker="o", label=f"test: {tname}")
        axes[0].axhline(asr_baselines[tname]["naive"]["asr"], color=color, ls="--", lw=0.5, label=f"ASR {tname} baseline")
    # baseline reference lines for the first test task
    ref = test_tasks[0]
    _, _ = None, None
    axes[0].axhline(0.0, color="gray", lw=0.5, ls="--")
    axes[0].set_xlabel("extraction & injection layer L")
    axes[0].set_ylabel("ASR on held-out task (naive prompts)")
    axes[0].set_title(f"Cross-task transfer ASR\n(vector trained on {train_tasks})")
    axes[0].legend(); axes[0].set_ylim(-0.05, 1.05)

    # Right: mean logit-diff (more graded signal)
    for tname, color in zip(test_tasks, colors):
        lds = [results[L][tname]["mean_logit_diff"] for L in layers]
        axes[1].plot(layers, lds, marker="o", label=f"test: {tname}")
        axes[1].axhline(asr_baselines[tname]["naive"]["ld"], color=color, ls="--", lw=0.5, label=f"LD {tname} baseline")

    axes[1].axhline(0.0, color="gray", lw=0.5, ls="--")
    axes[1].set_xlabel("extraction & injection layer L")
    axes[1].set_ylabel("mean logit-diff on held-out task")
    axes[1].set_title("Cross-task logit-diff vs layer")
    axes[1].legend()

    plt.tight_layout(); plt.show()


def cross_steering_coef(model, prompts, task_residuals, train_tasks, test_tasks, layer, coefs, batch=4, n_max=50, plotting=True):
    device = next(model.parameters()).device
    results = {c: {} for c in coefs}
    
    print("Computing steering vector...")
    train_vecs = []
    for name in train_tasks:
        v = (task_residuals[name]["combine"][layer].mean(0)
            - task_residuals[name]["naive"][layer].mean(0))
        train_vecs.append(v)
    v_train = torch.stack(train_vecs).mean(0).to(device)

    print("Computing steering effect for:")
    results = {}
    for tname in test_tasks:
        print(tname, " ...")
        results[tname] = {'asr':[], 'ld':[]}
        for coef in tqdm(coefs):
            target = prompts[tname]["prompts"]["naive"][:n_max]
            hooks = [(f"blocks.{layer}.hook_resid_post",
                    make_steering_hook(v_train, coef))]
            logits, _ = cache_resid(model, target, batch_size=batch, fwd_hooks=hooks)
            m = compute_metrics(logits,
                                "sentiment",
                                tname)
            results[tname]['asr'].append(m['asr'])
            results[tname]['ld'].append(m['mean_logit_diff'])

    if plotting:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        colors = ["tab:blue", "tab:orange"]

        for tname in test_tasks:
            axes[0].plot(coefs, results[tname]['asr'], label=tname)
            axes[1].plot(coefs, results[tname]['ld'], label=tname)

        for ax in axes:
            ax.set_xlabel("Scale factor")
            ax.set_ylabel("ASR")
            ax.legend()
        plt.suptitle("Effect of steering with respect to the scale factor")
        plt.show()
    
    return results



def cross_steering_cb(model, prompts, task_residuals, train_tasks, test_tasks, layer, coefs, batch=4, n_max=50, plotting=True):
    device = next(model.parameters()).device
    results = {c: {} for c in coefs}
    
    print("Computing steering vector...")
    train_vecs = []
    for name in train_tasks:
        v = (task_residuals[name]["combine"][layer].mean(0)
            - task_residuals[name]["naive"][layer].mean(0))
        train_vecs.append(v)
    v_train = torch.stack(train_vecs).mean(0).to(device)

    print("Computing steering effect for:")
    results = {}
    for tname in test_tasks:
        print(tname, " ...")
        results[tname] = {'asr':[], 'ld':[]}
        for coef in tqdm(coefs):
            target = prompts[tname]["prompts"]["combine"][:n_max]
            hooks = [(f"blocks.{layer}.hook_resid_post",
                    make_steering_hook(v_train, coef))]
            logits, _ = cache_resid(model, target, batch_size=batch, fwd_hooks=hooks)
            m = compute_metrics(logits,
                                "sentiment",
                                tname)
            results[tname]['asr'].append(m['asr'])
            results[tname]['ld'].append(m['mean_logit_diff'])

    if plotting:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        colors = ["tab:blue", "tab:orange"]

        for tname in test_tasks:
            axes[0].plot(coefs, results[tname]['mean_p_inj'], label=f"{tname}:p_inj", linestyle='--')
            axes[0].plot(coefs, results[tname]['mean_p_cor'], label=f"{tname}:p_cor", linestyle=':')
            axes[0].set_ylabel("probability of inj/corr")
            axes[1].plot(coefs, results[tname]['ld'], label=tname)
            axes[1].set_ylabel("logit difference")

        for ax in axes:
            ax.set_xlabel("Scale factor")
            
            ax.legend()
        plt.suptitle("Effect of steering with respect to the scale factor")
        plt.show()
    
    return results
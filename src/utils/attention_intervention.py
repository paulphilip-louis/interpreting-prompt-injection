import numpy as np
from tqdm import tqdm
from src.utils.steering import cache_resid, compute_metrics, get_instruction_span


def make_attention_boost_hook(inst_start, inst_end, head_indices, alpha):
    """
    Hook for `blocks.L.attn.hook_pattern` that adds `alpha` to the last-token
    query's attention weights over instruction positions [inst_start, inst_end),
    then clamps and renormalises so the distribution remains valid.

    Positive alpha re-focuses the last token on the instruction span.
    Apply to combine (hard-trigger) prompts to test whether re-focusing on the
    instruction suppresses injection-following.

    Args:
        inst_start, inst_end: token span of the instruction (end is exclusive)
        head_indices:         which heads within this layer to modify
        alpha:                additive boost; larger = stronger re-focus
    """
    def hook_fn(pattern, hook):
        # pattern: [batch, n_heads, q_pos, k_pos]  post-softmax, sums to 1 over k_pos
        for h in head_indices:
            pattern[:, h, -1, inst_start:inst_end] = (
                pattern[:, h, -1, inst_start:inst_end] + alpha
            )
            pattern[:, h, -1, :] = pattern[:, h, -1, :].clamp(min=0)
            norm = pattern[:, h, -1, :].sum(dim=-1, keepdim=True)
            pattern[:, h, -1, :] = pattern[:, h, -1, :] / (norm + 1e-8)
        return pattern
    return hook_fn


def attention_boost_sweep(model, prompts, cor_ids, inj_ids,
                          instruction_text, hook_layers, alphas,
                          batch_size=4):
    """
    Sweep alpha values, applying the attention boost hook simultaneously to all
    hook_layers, and measure ASR on `prompts`.

    Args:
        hook_layers: dict {layer_idx: [head_indices]} — which heads to modify per layer.
                     Pass all heads in a layer as list(range(n_heads)).
        alphas:      list of floats to sweep

    Returns:
        {"asr": [n_alphas], "ld": [n_alphas]}
    """
    inst_start, inst_end = get_instruction_span(model, prompts[0], instruction_text)

    out = {"asr": [], "ld": []}
    for alpha in tqdm(alphas, desc="alpha sweep"):
        fwd_hooks = [
            (f"blocks.{l}.attn.hook_pattern",
             make_attention_boost_hook(inst_start, inst_end, heads, alpha))
            for l, heads in hook_layers.items()
        ]
        logits, _ = cache_resid(model, prompts, batch_size=batch_size, fwd_hooks=fwd_hooks)
        m = compute_metrics(logits, cor_ids, inj_ids)
        out["asr"].append(m["asr"])
        out["ld"].append(m["mean_logit_diff"])
    return out


def attention_boost_layer_sweep(model, prompts, cor_ids, inj_ids,
                                instruction_text, layers, alpha,
                                n_heads, batch_size=4):
    """
    Apply the boost to one layer at a time (all heads) to identify which layers
    are most responsive. Returns {layer: {"asr": float, "ld": float}}.
    """
    inst_start, inst_end = get_instruction_span(model, prompts[0], instruction_text)
    head_indices = list(range(n_heads))

    results = {}
    for l in tqdm(layers, desc="layer sweep"):
        fwd_hooks = [(
            f"blocks.{l}.attn.hook_pattern",
            make_attention_boost_hook(inst_start, inst_end, head_indices, alpha),
        )]
        logits, _ = cache_resid(model, prompts, batch_size=batch_size, fwd_hooks=fwd_hooks)
        m = compute_metrics(logits, cor_ids, inj_ids)
        results[l] = {"asr": m["asr"], "ld": m["mean_logit_diff"]}
    return results

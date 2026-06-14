# Results

Throughout, we quantify attack success with two complementary measures. The
**attack success rate (ASR)** is computed from the next-token distribution as
`[mean(p_inj − p_corr) + 1] / 2`, where `p_inj` and `p_corr` are the probability
masses on the injected-task and intended-task answer tokens respectively; this
replaces an earlier thresholded definition `(p_inj > p_corr).mean()`, which
masked cases where both probabilities were low (see Section: Metric). We also
report the **mean logit difference** `mean(logit_inj − logit_corr)` as an
unnormalised view of the same quantity. Unless stated otherwise, results are on
Qwen2.5-1.5B-Instruct with `sentiment` as the main task; cross-model coverage is
summarised in Section: Model coverage.

---

## 1. Experiment 1 — Sufficiency of a single residual-stream direction

We define the candidate injection direction as the diff-of-means vector
`v_combine(L) = (resid_combine − resid_naive).mean()` at layer `L`, computed
between prompts that differ only in their trigger (the inert `naive` trigger
versus the aggressive `combine` trigger). Adding `c · v_combine(L)` to the
residual stream of `naive`-trigger prompts tests whether the direction is
*sufficient* to induce injection-following.

**Steering induces injection-following.** Adding the direction at a late layer
drives ASR from its `naive` baseline (≈0.04) to near the `combine` ceiling
(≈0.97), recovering essentially the full attack effect from an otherwise inert
prompt [Fig. 1]. The effect is concentrated in the late-middle layers, with
`L ≈ 21` the strongest single layer in the initial sweep.

**The direction is content-conditional, not a generic output hijack.** Applying
the same steering vector to *safe* prompts that contain no injection leaves the
model's behaviour intact: ASR remains 0.000, the probability mass on the injected
tokens stays at 0.000, and the intended-task probability is preserved
(`P(cor) = 0.952`) [Fig. 2]. Inspection of individual generations shows no
appearance of injected-task tokens (e.g. no `spam` outputs); the model continues
to perform the legitimate task. Sufficiency is therefore conditional on the
prompt actually carrying an injection, rather than the direction forcing a fixed
output.

**Controls.** A random vector of equal norm, added at the same layer and
coefficient, produces no measurable change in ASR or logit difference,
confirming the effect is specific to the learned direction rather than to a
perturbation of that magnitude. A vector constructed to be orthogonal to the
injection logit direction is likewise inert, indicating that `v_combine` is not
simply a logit-writing direction that mechanically up-weights the injected
answer tokens. [Fig. 3 — controls panel.]

**Onset and peak.** Sweeping the injection layer shows that the steering effect
is absent in early layers, appears abruptly at layer ≈13, rises rapidly, and
stabilises around layer 20 [Fig. 4]. Late layers still carry meaningful
computation: a vector extracted at layer 18 is markedly weaker than vectors
extracted at later layers when applied at their own layer (steered-`naive` ASR
of 0.31 at `L=18` versus 0.90 at `L=21`, 0.83 at `L=24`, and 0.74 at `L=27`),
showing the direction is not finalised at its onset but continues to develop
through the late layers before a slight decline at the very end.

**Valid-coefficient window.** The steering effect is only interpretable as
genuine injection-following within a bounded coefficient range. At large positive
coefficients the model leaves the manifold of either task and its outputs
degrade: rather than following the injected instruction, it collapses toward the
output *format* of the task from which the steering vector was extracted, and
degenerate or repetitive text appears [Fig. 5 — example generations across
coefficients]. We therefore report sufficiency ASR within the window in which
steered generations remain coherent injected-task responses, and we flag the
high-coefficient plateau as an out-of-distribution artefact rather than an
attack-success signal.

**Token-position of the intervention (partial).** Restricting the intervention
to different token spans localises where the susceptibility representation is
read out. Steering applied across the *main-instruction span* keeps ASR high and
roughly flat across coefficients, whereas steering applied only at the *last
token* reproduces the full dose-response S-curve, crossing from the `naive` to
the `combine` regime as the coefficient increases [Fig. 6]. This indicates the
direction is effective when injected at the positions the model attends to when
deciding which instruction to follow.

> *Payload-only and whole-data steering: section intentionally left empty —
> experiment to run.*

---

## 2. Experiment 2 — Cross-task transfer and layer selection

We next ask whether the direction extracted on one set of injected tasks
transfers to held-out tasks, using a leave-one-out (LOO) protocol in which the
steering vector is the average of vectors computed on all tasks except the one
being evaluated.

**Cross-task transfer.** A direction trained on held-out tasks and applied to a
left-out task's `naive`-trigger prompts reproduces the injection effect on that
unseen task, moving it from the safe regime to the attack regime as the
coefficient increases from 0 to ≈1.5 [Fig. 7]. Transfer holds across the
fixed-form classification tasks (spam, hsol, rte, mrpc) in both ASR and logit
difference, establishing that the direction is not purely task-specific.

**Cross-task geometry.** Pairwise cosine similarities between per-task steering
vectors at layer 21 are consistently positive (off-diagonal mean 0.473, std
0.114), with higher similarity between tasks of the same output structure
[Table 1]. The free-form generation tasks (gigaword, jfleg) are more similar to
each other than to the fixed-form classification tasks, and vice versa,
indicating that the injection direction shares a common component across tasks
while carrying a task-structure-dependent component.

**Shared versus task-specific direction.** Decomposing each vector into a shared
component (common across tasks) and a task-specific residual, we find that
steering with the *shared* component alone reproduces the full-vector effect:
the ASR and logit-difference curves for the shared-only and full-vector
interventions overlap across coefficients on held-out tasks [Fig. 8]. This is
direct evidence for a single shared injection axis that suffices to drive the
attack, with the task-specific residual largely inessential for transfer.

> *Quantification of the shared vs. task-specific magnitude split: to add.*

> *Free-form (gigaword/jfleg) transfer test, and the principled criterion
> reconciling best-layer = 21, peak transferability ≈ 19–22, and clamp-from-18
> for the definitive layer choice: section intentionally left empty —
> to run/finalise.*

---

## 3. Experiment 3 — Coefficient sweep and ablation (necessity)

To test whether the direction is *necessary* for injection-following, we ablate
it on `combine`-trigger prompts (where the attack succeeds at baseline) by
clamping the projection of the residual stream onto the injection axis to its
`naive`-condition mean. To counter the Hydra effect — recovery of the ablated
information by downstream layers — the clamp is applied at every layer from
layer 18 onward rather than at a single layer; single-layer projection-out is
insufficient for this reason.

**Ablation collapses the attack on most tasks.** Clamping reduces a 100% baseline
attack rate to 0% on spam and hsol, and to 6% on mrpc, while the intended task is
restored [Table 2]. The necessity is therefore total on spam and hsol and partial
on mrpc under this protocol.

**Dose-response.** Necessity is graded rather than binary, and depends on both
the task and the extraction source of the ablation vector [Fig. 9]. Using a
vector derived from mrpc, negative steering on spam `combine` prompts drives ASR
from 0.860 (coef 0) down through 0.338 (−0.5) to 0.041 (−1.0) and 0.017 (−2.0);
hsol falls from 0.991 to 0.625 (−1.0) and 0.034 (−3.0); rte is markedly more
resistant, remaining at 0.824 at −0.5 and only collapsing to 0.186 at −3.0.
Using a vector derived from spam shows the same ordering, with rte and mrpc
requiring substantially stronger negative coefficients than hsol to collapse.
We report the full curves rather than a single operating point.

**The RTE anomaly.** Under the fixed-layer clamp, rte does not collapse with the
other tasks: its attack rate remains at 95.5% where spam and hsol fall to 0%
[Table 2]. The dose-response above is consistent with rte's injection direction
being partly off-axis relative to the clamped direction (it requires the
strongest negative coefficient of any task to collapse). We surface this
explicitly rather than averaging it away.

> *Resolution of the RTE anomaly (metric re-run, off-axis quantification, or
> clamp layer-coverage check); generation-coherence check confirming the model
> recovers the original task after ablation; and necessity ablation on
> neural-exec prompts: sections intentionally left empty — to run.*

---

## 4. Experiment 4 — Comparison with optimised (neural-exec) triggers

We optimised a Pasquini-style neural-exec trigger for each model and define
`v_neural_exec = (resid_neural_exec − resid_naive).mean()` by analogy with
`v_combine`. If both natural and optimised triggers exploit the same internal
mechanism, the two vectors should align.

**The optimised trigger is a strong attack.** On Qwen2.5-1.5B-Instruct, the
neural-exec trigger reaches an ASR comparable to `combine` (≈0.88 versus ≈0.96),
far above the inert `naive` and random baselines [Fig. 10]. Its incoherent
starting point produces near-zero ASR, confirming the effect is acquired through
optimisation rather than present in the seed string.

**The two directions have matched magnitude.** The per-layer norms of
`v_combine` and `v_neural_exec` track each other closely across all layers, with
the neural-exec norm running slightly below `combine` in the late layers
[Fig. 11].

**The two directions are closely aligned.** After correcting a prompt-formatting
issue that had previously invalidated the optimised-trigger vectors on
non-spam tasks, the cosine similarity between `v_combine` and `v_neural_exec`
rises from near zero in early layers to a high value from the mid layers onward,
exceeding 0.5 from layer ≈17 and reaching ≈0.8 for spam and hsol [Fig. 12].
A PCA of the per-task `combine` and `neural_exec` vectors shows the two trigger
types occupy nearby points for each task, separated by a relatively consistent
translation `(v_neural_exec − v_combine)` across tasks [Fig. 13]. Together with
the matched norms, this is our central evidence that natural and optimised
triggers converge on a single shared direction.

**Comparable steering effect.** Used as a steering vector on `naive`-trigger
prompts, `v_neural_exec` reproduces the same qualitative effect as `v_combine`:
ASR rises with coefficient and layer, with a strong effect emerging from layer
≈18 [Fig. 14]. The two vectors are thus interchangeable as sufficiency
interventions within-task.

> *Necessity ablation on neural-exec prompts (does the combine-derived clamp also
> collapse optimised-trigger ASR); resolution of the transfer asymmetry between
> v_combine and v_neural_exec; and robustness across 2–3 independently optimised
> triggers: sections intentionally left empty — to run.*

---

## 5. Task-specificity

**Steering vectors cluster by output structure.** Projecting the per-task
vectors into their principal components separates them primarily by output form:
the free-form tasks (gigaword, jfleg) and the fixed-form classification tasks
(spam, hsol, rte, mrpc) fall into distinct regions, with the first principal
component capturing ≈37% of the variance [Fig. 15]. Within the fixed-form
cluster, a single direction accounts for roughly half the remaining variance
(PC1 ≈0.51, PC2 ≈0.35, PC3 ≈0.14) [Fig. 16]. This supports decomposing the
injection direction into a shared component (the uncentred mean across tasks)
and task-specific components (recovered by PCA on the centred vectors).

> *Quantification of the shared/task-specific magnitude split: to add.*

**The direction depends on the main task, not only the injected task.** For a
fixed injected task, the vector `v_combine` differs systematically with the
*main* instruction: a direction extracted with one main task and applied under a
different main task transfers far less well, and degrades as the main task is
changed [Fig. 17]. Visualising `v_combine` per main task shows the naive→combine
shift pointing in main-task-dependent directions in PC space [Fig. 18]. This
main-task dependency is the central obstacle to a single task-agnostic injection
axis and motivates the safe-centroid construction below.

> *Development and evaluation of the safe-centroid "potential new direction" as a
> candidate task-agnostic axis: section intentionally left empty — to run.*

---

## 6. Implementing this as a defense

> *Section intentionally left empty — to run.*
> Planned: utility preservation on clean prompts under the clamp; generalisation
> across attack types (train on combine, defend ignore/escape/neural-exec on
> held-out prompts); generalisation across main tasks (single direction vs.
> per-task vs. safe-centroid direction); baseline comparison against an existing
> prompt-injection defense; explicit white-box, per-model threat model.

---

## Supporting analyses

### Linear decodability motivates the linear approach
A ridge probe trained to predict ASR from the residual stream is near-chance in
the early layers and improves steadily from layer ≈10 onward, reaching R² ≈0.5
in the mid layers for each trigger type (the `ignore` trigger lags the others)
[Fig. 19]. The injection outcome thus becomes linearly decodable in the same
layer range where the steering effect appears, motivating the diff-of-means
approach.

### The steering direction recovers the causal circuit
The per-head decomposition of `v_21` (the contribution of each head's output to
the steering direction) closely matches the head map obtained from activation
patching on the `(naive, combine)` contrast, with the same late-middle heads
dominant in both [Fig. 20]. This links the linear steering direction to the
causal circuit identified by patching, rather than the two analyses standing
apart.

> *Decision on the in-scope extent of the activation-patching and logit-attribution
> circuit analysis (Session 5 material): to finalise.*

### Metric
We adopted `[mean(p_inj − p_corr) + 1] / 2` after observing that the earlier
thresholded definition `(p_inj > p_corr).mean()` could report a high ASR when
both probabilities were low and the generation was incoherent — an artefact
visible directly in the steered generations. The continuous metric tracks the
mean logit difference and avoids this failure mode.

> *Validation that the logit-based ASR tracks a generation-based ASR on a subset:
> to run.*

### Model coverage
The full sweep (sufficiency, cross-task transfer, ablation, neural-exec
comparison) is on Qwen2.5-1.5B-Instruct. Core sufficiency and neural-exec
results are reproduced on Llama3.1-8B-Instruct, where the neural-exec trigger
again gives the strongest attack (ASR ≈0.80) and within-task `combine`/
`neural_exec` cosine reaches ≈0.88; notably, cross-task transfer is weak to
absent on this model [Fig. 21]. Logit-attribution maps are additionally shown on
Llama3.2-1B-Instruct.

> *Statistics (N, seeds, variance/CIs on the steering, ablation, and transfer
> curves) and a consolidated model-coverage table, including the flat
> neural-exec attribution heatmap despite high ASR (possible MLP-mediated or
> distributed pathway): to add.*
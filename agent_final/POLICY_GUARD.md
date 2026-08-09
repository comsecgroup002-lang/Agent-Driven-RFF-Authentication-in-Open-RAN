# Policy Guard: Reproducibility and Safety Details

This document supplements Section 5.3.3 and Table 1 of the manuscript. It describes the manuscript-aligned implementation in which the RAG-enabled LLM advisory agent runs on the **cloud side**, while the edge control agent performs routine authentication, cached-score threshold replay, and final deterministic enforcement.

## 1. Functional boundary

The runtime responsibilities are separated as follows:

- **Edge control agent:** routine RFF authentication, feasibility monitoring, cached-score threshold replay, Pareto-filtered operating-point selection, final validation of cloud-returned threshold guidance, and local fallback during cloud interruption.
- **Cloud coordination agent:** historical-case retrieval, RAG-enabled LLM advisory generation, deterministic retrieval-only guidance, cloud-side policy validation, and source-model reprovisioning.
- **RAG-enabled LLM advisory agent:** generates only a bounded candidate proposal in threshold/operating-point space. It neither authenticates signals nor directly executes control actions.

The lightweight-guidance path does **not** invoke edge-side LLM inference and does **not** retrain on target-domain data.

## 2. Feasibility rule

Hard authentication feasibility is determined only by closed-set accuracy and open-set AUROC:

```text
A_c >= gamma_c
and
A_o >= gamma_o
```

`reject_rate` (`R`) is retained only as an auxiliary service-availability/ranking metric. It may participate in Pareto ranking or retrieval similarity, but it is not a third hard feasibility constraint.

## 3. Runtime validation path

The active lightweight-guidance path is:

1. The edge performs local cached-score threshold replay.
2. If the operating point remains infeasible, the edge sends a compact state summary to `/guidance` on the cloud coordination service.
3. The cloud retrieves historical evidence. For the RAG arm, up to the top three successful cases and top two failed cases are provided to the LLM. Failed cases are contextual negative evidence only; they are not hard rejection rules.
4. The cloud-side LLM returns a JSON candidate containing only threshold-space variables.
5. `LLMAdvisor._validate_advice()` performs first-pass allowlist/type filtering, single-step limiting, and safe-range projection.
6. `cloud_agent_v2.py::PolicyGuard.review()` performs a second deterministic cloud-side validation.
7. The cloud converts the validated patch into a bounded threshold-search interval and returns it to the edge.
8. `EdgeAgentV2._policy_guard_review()` and `_validate_threshold_interval()` perform final edge-side deterministic enforcement.
9. The edge restricts its threshold grid and evaluates candidates using cached-score replay. Only the edge-selected operating point is executed.
10. If no executable RAG/LLM proposal remains, deterministic retrieval-based guidance may be used. A cloud transport/service interruption instead invokes the separate local fallback advisor.

At no point is the raw LLM output directly applied to an authentication decision.

## 4. LLM proposal schema and safe set

The cloud-side LLM may propose only:

| Variable | Safe range | Maximum single-event step |
|---|---:|---:|
| `accept_quantile` | [0.80, 0.95] | 0.05 |
| `margin_quantile` | [0.10, 0.30] | 0.05 |
| `rho` | [-0.15, 0.35] | 0.10 |

For an allowed variable `k`, validation applies the equivalent operation

```text
v_k = Proj_[l_k,u_k](theta_k + clip(p_k - theta_k, -s_k, +s_k))
```

where `p_k` is the proposed value, `theta_k` is the current operating value, `[l_k,u_k]` is the safe interval, and `s_k` is the maximum single-event step.

Unsupported keys and non-numeric values are discarded.

## 5. Threshold-search safe set

The edge base grid may cover a wider diagnostic search range, but cloud lightweight guidance is restricted to the validated advisory region. The cloud response uses the following fields:

| Search variable | Cloud advisory range |
|---|---:|
| `accept_quantile` | [0.80, 0.95] |
| `margin_quantile` | [0.10, 0.30] |
| `rho` / `delta_fused` | [-0.15, 0.35] |
| `delta_margin` | [0.05, 0.35] (compatibility field) |

When at least `prior.min_samples = 3` successful historical episodes are available, an empirical interval is constructed from successful-case statistics. The 10th--90th percentile interval is used for `accept_quantile`, `margin_quantile`, and `rho`; the `rho` interval is additionally expanded by `prior.interval_expansion = 0.05` before being clipped to the global safe range.

The returned interval narrows the **search space** only. It does not directly set the final authentication operating point.

## 6. Deterministic retrieval-only guidance

The deterministic ablation/fallback path contains no LLM call. Up to the top three retrieved successful cases are aggregated using normalized nonnegative cosine-similarity weights.

If the successful cases have similarities `s_i` and threshold-space actions `delta_i`, the weights are

```text
w_i = max(s_i, 0) / sum_j max(s_j, 0)
```

and the proposal is the weighted aggregation of the available threshold-space actions. If all nonnegative similarities are zero, uniform weights are used. If no usable successful case exists, deterministic retrieval guidance is reported as unavailable rather than fabricating a historical prior.

## 7. Failed historical cases

Failed cases are supplied to the RAG-enabled LLM as negative contextual evidence to discourage repetition of previously ineffective adjustment patterns.

They are **not** used as a hard similarity-based rejection rule. Executable safety is determined by the explicit numerical policy guard, edge-side interval validation, and final A_c/A_o feasibility evaluation.

## 8. Local fallback under cloud interruption

The local fallback advisor is distinct from deterministic cloud retrieval guidance. It is invoked only when the cloud request itself is unavailable or interrupted.

It performs a small bounded movement in threshold space based on the current A_c/A_o feasibility gaps and the previous threshold state. It does not invoke an LLM, does not retrain the model, and does not use `R` as a feasibility trigger. Its purpose is bounded local operation/service continuity rather than guaranteed performance improvement.

## 9. Model refresh

The fixed-base model-refresh implementation is **source-only reprovisioning**:

- the cloud packages the original source-trained weights and baseline thresholds as `v0000_initial`;
- target-domain uploaded weights are ignored;
- target-domain model aggregation is disabled;
- `/aggregate` returns a disabled status;
- model refresh downloads and reloads the original source package, after which edge-side threshold correction resumes;
- no current target-domain samples are used for retraining or fine-tuning in this path.

This behavior corresponds to Section 7.5 of the manuscript and is separate from the later rolling-refresh experiment, which explicitly promotes a later observed domain to a new training base.

## 10. Proposal audit statistics

`CloudAgentV2` records policy-guard outcomes and exposes them through `/stats`:

```text
llm_requests
reviewed_proposals
directly_accepted
guard_adjusted
no_executable_update
deterministic_fallback
range_clamp_events
step_clamp_events
unsupported_or_invalid_events

direct_acceptance_rate = directly_accepted / reviewed_proposals
guard_intervention_rate = guard_adjusted / reviewed_proposals
```

These counters support quantitative reporting from actual experiment logs. They should not be replaced by values inferred from the policy definition.

## 11. Relevant implementation locations

### `llm_advisor.py`
- `LLMAdvisor`: cloud-side advisory agent.
- `SYSTEM_PROMPT`: functional boundary and JSON-only advisory format.
- `THRESHOLD_RANGES`: threshold-space allowlist and safe intervals.
- `MAX_STEP_SIZE`: single-event step limits.
- `_parse_response()`: robust JSON extraction.
- `_validate_advice()`: first-pass deterministic validation.
- `get_threshold_guidance()`: cloud-side LLM generation entry point.

### `cloud_agent_v2.py`
- `StateVector`: compact cloud retrieval state; A_c/A_o are the only feasibility-gap dimensions.
- `ExperienceRetriever.retrieve_for_guidance()`: retrieves successful and failed historical evidence.
- `PolicyGuard.review()`: second cloud-side deterministic validation.
- `PolicyGuard.suggest_safe_adjustment()`: top-three successful-case deterministic guidance.
- `_provide_lightweight()`: RAG, LLM-only, and deterministic cloud guidance arms.
- `_build_prior()`: successful-case empirical threshold prior.
- `_provide_model()` / `_ensure_initial_model()`: source-only model reprovisioning.
- `get_stats()`: policy-audit statistics.

### `edge_agent_v2.py`
- `check_objectives()`: A_c/A_o-only feasibility check.
- `_policy_guard_review()`: final edge-side validation.
- `_validate_threshold_interval()`: clamps cloud guidance to edge hard bounds.
- `get_search_grid()`: restricts the threshold grid using validated cloud guidance.
- `_run_lightweight_mode()`: edge-first threshold replay followed by event-driven cloud guidance; no retraining after guidance.
- `_rule_based_fallback()`: edge-local bounded interruption fallback.
- `_run_model_reprovision_mode()`: source-only model reprovisioning without target-domain training.

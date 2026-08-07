# Policy Guard: Reproducibility and Safety Details

This document supplements Section 5.3.3 and Table 1 of the manuscript by describing the implementation-level validation path used before LLM-generated parameter changes are applied at the edge. The implementation is distributed across `llm_advisor.py`, `cloud_agent_v2.py`, and `edge_agent_v2.py`.

## 1. Runtime validation path

The active lightweight-guidance path is:

1. The cloud retrieves similar successful and failed historical cases and returns the RAG context, a conservative cloud-side parameter suggestion, safe parameter ranges, and maximum step sizes.
2. The edge-side LLM advisor receives the current performance state, current signal-augmentation parameters, retrieved evidence, and the safety constraints in the prompt.
3. The LLM output is parsed as JSON. Direct JSON, fenced JSON, and the first balanced JSON object are accepted parsing forms.
4. `LLMAdvisor._validate_advice()` performs the first executable-safety pass: unsupported keys and non-numeric values are removed, the change of each allowed parameter is step-limited, and the result is projected into the parameter-specific safe interval.
5. `EdgeAgentV2._policy_guard_review()` performs a second edge-side check using the safe ranges and step limits supplied by the cloud. If a value still violates a bound, it is clipped before execution and the proposal is marked as requiring guard intervention.
6. Only the adjusted parameter dictionary is passed to `apply_param_changes()`. If no executable LLM change is available, the system uses the cloud policy suggestion or the deterministic rule-based fallback.

The guard is therefore **corrective rather than purely binary**: a parsable proposal that violates a limit is not executed as generated; it is projected back into the admissible region and the adjusted proposal is applied. For this reason, the most informative quantitative statistics are the *direct-acceptance rate* and *guard-intervention rate*, rather than only an accept/reject count.

## 2. Pseudocode

```text
Algorithm: Guarded LLM Guidance
Input:
    raw LLM output R
    current signal-augmentation state theta
    global parameter allowlist and ranges S
    global step limits Delta
    cloud-provided guard ranges S_cloud
    cloud-provided step limits Delta_cloud
Output:
    executable parameter update theta_star, or deterministic fallback

1:  P <- parse_json(R)
2:  if parsing fails:
3:      return deterministic_fallback()
4:
5:  V <- empty dictionary
6:  for each (key, proposed_value) in P.param_changes:
7:      if key is not in the parameter allowlist:
8:          continue
9:      if proposed_value is not numeric:
10:         continue
11:
12:     current <- theta[key]
13:     step <- Delta[key] if explicitly defined else 0.1
14:     value <- current + clip(proposed_value - current, -step, +step)
15:     value <- clip(value, S[key].lower, S[key].upper)
16:     V[key] <- value
17:
18: if V is empty:
19:     return cloud_safe_suggestion_or_deterministic_fallback()
20:
21: approved_without_intervention <- true
22: A <- empty dictionary
23: for each (key, value) in V:
24:     current <- theta[key]
25:     if key in S_cloud and value is outside S_cloud[key]:
26:         value <- clip(value, S_cloud[key].lower, S_cloud[key].upper)
27:         approved_without_intervention <- false
28:     if key in Delta_cloud and abs(value - current) > Delta_cloud[key]:
29:         value <- current + sign(value-current) * Delta_cloud[key]
30:         approved_without_intervention <- false
31:     A[key] <- value
32:
33: apply A at the edge
34: record whether A was unchanged or guard-adjusted
35: return A
```

For an allowed parameter k, the first validation pass can be written as

`v_k = Proj_[l_k,u_k]( theta_k + clip(p_k - theta_k, -s_k, +s_k) )`,

where `p_k` is the LLM-proposed value, `[l_k,u_k]` is the allowed interval, and `s_k` is the maximum single-step change.

## 3. Parameter allowlist and safe ranges

The following ranges are defined in `llm_advisor.py::PARAM_RANGES`. Parameters not listed here are ignored if emitted by the LLM.

| Parameter | Safe range | First-pass max step |
|---|---:|---:|
| `retain_ratio` | [0.35, 0.65] | 0.08 |
| `channel_aug_prob` | [0.70, 1.00] | 0.10 |
| `engineering_aug_prob` | [0.50, 0.85] | 0.10 |
| `drift_prob` | [0.02, 0.15] | 0.10 (default) |
| `shift_prob` | [0.05, 0.20] | 0.10 (default) |
| `tau_rms_min_ns` | [15, 30] | 0.10 (default) |
| `tau_rms_max_ns` | [100, 200] | 0.10 (default) |
| `doppler_min_hz` | [0.05, 0.20] | 0.10 (default) |
| `doppler_max_hz` | [3.0, 8.0] | 0.10 (default) |
| `k_factor_min_db` | [2.0, 5.0] | 0.10 (default) |
| `k_factor_max_db` | [8.0, 15.0] | 0.10 (default) |
| `snr_min_db` | [25, 35] | 0.10 (default) |
| `snr_max_db` | [35, 45] | 0.10 (default) |
| `gain_std` | [0.02, 0.08] | 0.10 (default) |
| `nonlinear_coef` | [0.001, 0.005] | 0.10 (default) |
| `iq_amplitude_std` | [0.01, 0.03] | 0.10 (default) |
| `iq_phase_std_deg` | [0.5, 2.0] | 0.10 (default) |

The cloud-side secondary guard (`cloud_agent_v2.py::PolicyGuard`) additionally publishes the following constraints to the edge:

| Parameter | Cloud guard range | Cloud max step |
|---|---:|---:|
| `retain_ratio` | [0.35, 0.65] | 0.08 |
| `engineering_aug_prob` | [0.50, 0.85] | 0.10 |
| `channel_aug_prob` | [0.70, 1.00] | -- |

`channel_aug_prob` is still step-limited to 0.10 in the first LLM-advisor validation pass.

## 4. Threshold-search safe set

Lightweight cloud guidance also constrains the edge-side threshold-search space before Pareto selection. If no empirical prior is available, the default interval is:

| Threshold-search variable | Default interval |
|---|---:|
| `accept_quantile` | [0.80, 0.95] |
| `margin_quantile` | [0.10, 0.30] |
| `rho` / `delta_fused` | [-0.15, 0.35] |
| `delta_margin` | [0.05, 0.35] |

When enough historical episodes are available (`prior.min_samples = 3`), the cloud constructs an empirical interval from the experience bank. For `rho`, `accept_quantile`, and `margin_quantile`, the 10th--90th percentile interval is used; the `rho` interval is additionally expanded by `prior.interval_expansion = 0.05`. The edge then removes threshold-grid candidates outside these intervals before evaluation.

## 5. Historical-case constraints

Retrieved successful and failed cases are included in the LLM context. Successful cases are used to derive a conservative cloud-side suggestion, and failed cases are explicitly presented in the prompt as patterns to avoid. The current main execution path enforces hard numerical safety through range projection and step-size limits; historical-case evidence acts as evidence-conditioned guidance rather than a hard rejection rule.

## 6. Proposal outcome definitions for quantitative reporting

Because the implementation clips unsafe values instead of executing them unchanged, proposal-level statistics should be reported using the following definitions:

- **Reviewed proposal**: an LLM output with at least one executable parameter after parsing/type/allowlist validation and therefore reaching the edge-side policy guard.
- **Directly accepted**: the edge-side guard returns `policy_approved = True`; no second-pass clipping is required.
- **Guard-adjusted**: the edge-side guard returns `policy_approved = False`; at least one parameter is clipped because of a safe-range or maximum-step violation. The *adjusted* proposal, not the raw proposal, is executed.
- **No executable LLM update / fallback**: parsing fails, all proposed keys/values are invalid, the LLM is unavailable, or no valid parameter update remains; the system proceeds with the cloud policy suggestion or deterministic fallback.

Recommended statistics are:

```text
N_reviewed
N_directly_accepted
N_guard_adjusted
Direct acceptance rate = N_directly_accepted / N_reviewed
Guard intervention rate = N_guard_adjusted / N_reviewed
Number of range-clamp events
Number of step-clamp events
Per-parameter clamp counts
Number of fallback events (reported separately from reviewed proposals)
```


## 7. Relevant implementation locations

- `llm_advisor.py`
  - `SYSTEM_PROMPT`: JSON schema and safety instructions.
  - `PARAM_RANGES`: allowlisted LLM-adjustable parameters and global safe intervals.
  - `MAX_STEP_SIZE`: explicit single-step limits.
  - `_parse_response()`: robust JSON extraction.
  - `_validate_advice()`: allowlist/type filtering, step limiting, and safe-range projection.
- `cloud_agent_v2.py`
  - `PolicyGuard.SAFE_RANGES` and `PolicyGuard.MAX_STEP`: cloud-published secondary constraints.
  - `PolicyGuard.suggest_safe_adjustment()`: conservative suggestion from retrieved successful cases.
  - `_provide_lightweight()`: returns RAG context, safe ranges, max steps, and threshold intervals.
  - `_build_prior()`: empirical threshold-prior construction.
- `edge_agent_v2.py`
  - `get_llm_advice()`: RAG-to-LLM-to-policy-guard execution path.
  - `_policy_guard_review()`: final edge-side range/step validation.
  - `get_search_grid()`: safe-set restriction of threshold candidates.
  - `apply_param_changes()`: applies only the post-guard parameter dictionary.

# Policy Guard: Reproducibility and Safety Details

This document supplements Section 5.3.3 and Table 1 of the manuscript by describing the implementation-level validation path used before LLM-generated guidance can affect the deployed RFF authentication system.

The proposed framework follows a clear functional separation. The RFF authentication subsystem and the edge control agent remain responsible for per-signal authentication, feasibility monitoring, cached-score threshold replay, and final execution of validated operating-point updates. The RAG-enabled LLM advisory function belongs to the event-driven cloud coordination path and is invoked only after local edge-side correction becomes insufficient. The LLM therefore neither performs signal authentication nor directly executes control actions at the edge.


Two types of bounded guidance appear in the implementation:

1. **Threshold-space lightweight guidance**, which is the primary operating-point correction mechanism evaluated in the manuscript. It constrains the edge-side search over threshold-related variables and is applied through cached-score replay without repeated feature extraction or model retraining.
2. **Optional reprocessing/reconfiguration guidance**, which may modify selected signal-processing or augmentation parameters when an explicit reprocessing path is invoked. This branch is not part of routine cached-score threshold replay and should not be interpreted as the per-signal lightweight-guidance path evaluated in the main threshold-space recovery experiments.

## 1. Runtime validation path

For the lightweight cloud-guidance path described in the manuscript, the logical execution sequence is:

1. The edge control agent detects residual infeasibility after local cached-score threshold replay and sends a compact state summary to the cloud coordination agent.
2. The cloud retrieves similar successful and failed historical cases from the experience bank.
3. The retrieved evidence, current performance state, feasibility gaps, adjustable operating-point state, and explicit safety constraints are assembled as structured context for the RAG-enabled LLM advisory agent.
4. The LLM produces a **candidate guidance proposal**. This output is advisory only and is never treated as a directly executable authentication or control decision.
5. The candidate proposal is parsed and subjected to deterministic safety validation. Unsupported fields and invalid values are removed, numerical updates are restricted to predefined admissible ranges, and maximum single-step changes are enforced.
6. For threshold-space lightweight guidance, the validated proposal is converted into bounded search intervals or parameter priors that restrict the edge-side threshold-search space.
7. The edge control agent performs a final deterministic review of the returned constraints before using them. Only candidates within the validated safe set are evaluated through cached-score replay and Pareto-filtered operating-point selection.
8. If no executable LLM-guided update remains, the system uses a conservative cloud-side statistical suggestion or deterministic guidance derived from retrieved successful cases.

The LLM therefore does not directly modify authentication decisions, thresholds, model parameters, or received-signal classifications. It only proposes bounded guidance that must pass deterministic validation before influencing the edge-side search.

For the optional reprocessing/reconfiguration branch, the same principle applies: an LLM-generated parameter patch is first projected into predefined parameter ranges and step limits, and only the resulting post-guard parameter dictionary can be passed to the reprocessing pipeline.

The policy guard is therefore **corrective rather than purely binary**. A parsable numerical proposal that exceeds a permitted range or step size is not executed in its raw form; it is clipped or projected back into the admissible region. If no valid update remains after validation, the LLM proposal is discarded and deterministic guidance is used instead.

## 2. Pseudocode

```text
Algorithm: Guarded Cloud Advisory Guidance

Input:
    edge state summary h
    historical experience bank E
    current edge operating-point state theta
    global safe ranges S
    global step limits Delta

Output:
    validated guidance G_star,
    or deterministic fallback guidance

Cloud-side advisory stage:

1:  R <- retrieve_similar_cases(h, E)
2:  C <- build_structured_context(
        current performance,
        feasibility gaps,
        current operating-point state,
        retrieved successful cases,
        retrieved failed cases,
        safety constraints
    )

3:  P_raw <- LLM_advisory(C)

4:  P <- parse_json(P_raw)
5:  if parsing fails:
6:      return deterministic_guidance(R, h)

7:  V <- empty guidance structure

8:  for each proposed numerical field (key, proposed_value) in P:
9:      if key is not allowlisted:
10:         continue
11:     if proposed_value is not numeric:
12:         continue

13:     current <- theta[key]
14:     step <- Delta[key]
15:     value <- current
            + clip(proposed_value - current, -step, +step)

16:     value <- clip(
            value,
            S[key].lower,
            S[key].upper
        )

17:     V[key] <- value

18: if V is empty:
19:     return deterministic_guidance(R, h)

20: G_cloud <- construct_bounded_guidance(V, R)

Edge-side deterministic enforcement:

21: G_edge <- final_policy_review(G_cloud)

22: if G_edge contains threshold-space constraints:
23:     Omega_safe <- restrict_threshold_search_grid(G_edge)
24:     evaluate Omega_safe using cached-score replay
25:     perform feasibility screening and Pareto selection
26:     apply only the selected feasible operating-point update

27: else if G_edge explicitly requests optional reprocessing:
28:     apply only the post-guard reconfiguration dictionary
29:     execute the separate event-triggered reprocessing path

30: else:
31:     use deterministic fallback guidance

32: record whether the proposal was
        directly accepted,
        guard-adjusted,
        or replaced by deterministic guidance

33: return executed bounded guidance
```

For an allowed numerical parameter \(k\), the deterministic projection applied to an LLM-proposed value can be written as

```text
v_k =
Proj_[l_k,u_k](
    theta_k +
    clip(p_k - theta_k, -s_k, +s_k)
),
```

where `p_k` is the LLM-proposed value, `[l_k,u_k]` is the admissible interval, `s_k` is the maximum allowed single-step change, and `theta_k` is the current validated value.

This projection ensures that the raw LLM value is never directly applied.

## 3. Primary lightweight-guidance safe set: threshold-space control

The primary lightweight-guidance mechanism evaluated in the manuscript operates in the threshold-search space used by the edge control agent.

If no empirical prior is available, the following admissible intervals are used:

| Threshold-search variable | Default interval |
|---|---:|
| `accept_quantile` | [0.80, 0.95] |
| `margin_quantile` | [0.10, 0.30] |
| `rho` / `delta_fused` | [-0.15, 0.35] |
| `delta_margin` | [0.05, 0.35] |

The cloud-side advisory output does not directly select the final authentication threshold pair. Instead, it may narrow or bias the admissible search region. The edge control agent then evaluates candidate operating points through cached-score replay.

For a candidate threshold configuration \(\delta\), feasibility remains determined by the manuscript-level conditions

```text
A_c(delta) >= gamma_c
and
A_o(delta) >= gamma_o.
```

The rejection rate is retained as an auxiliary operating-point metric but is not used as an additional hard feasibility condition.

When sufficient historical episodes are available (`prior.min_samples = 3`), the cloud constructs an empirical search prior from the experience bank. For `rho`, `accept_quantile`, and `margin_quantile`, the 10th--90th percentile interval of relevant historical cases is used. The `rho` interval is additionally expanded by `prior.interval_expansion = 0.05`.

The resulting interval is a **search constraint**, not a direct control command. The edge removes threshold-grid candidates outside the validated interval and independently evaluates the remaining candidates before selecting an executable operating point.

## 4. Deterministic guidance from retrieved cases

If LLM-generated guidance is unavailable, unparsable, or leaves no executable update after deterministic validation, the framework can fall back to non-generative guidance constructed directly from retrieved successful cases.

For the retrieval-only deterministic guidance evaluated in the manuscript, the top retrieved successful cases are aggregated using normalized nonnegative cosine-similarity weights.

Let the retrieved successful cases be indexed by \(i\), with similarity score \(s_i\) and historical threshold-offset vector \(\delta_i\). The normalized weight is

```text
w_i = max(s_i, 0) / sum_j max(s_j, 0),
```

and the deterministic guidance is

```text
delta_det = sum_i w_i * delta_i.
```

Up to the top three available successful cases are used. If fewer than three successful cases are available, all available successful cases are used. If no successful case is available, retrieval-only deterministic guidance is reported as unavailable rather than fabricating a historical prior.

This pathway contains no LLM generation and provides the non-generative retrieval-based comparator used in the ablation study.

## 5. Historical successful and failed cases

The cloud experience bank stores both successful and unsuccessful historical adaptation episodes.

Retrieved successful cases provide positive evidence about operating regions and adjustment directions that have previously reduced feasibility gaps.

Retrieved failed cases are included in the structured LLM context as **negative contextual evidence**. They are intended to discourage the advisory agent from repeating previously ineffective or harmful adjustment patterns.

In the current implementation, similarity to a failed historical case is **not itself a hard rejection condition**. Hard executable-safety guarantees are enforced through deterministic mechanisms such as:

- parameter allowlists,
- numeric type validation,
- predefined safe ranges,
- maximum single-step limits,
- threshold-search safe-set restriction,
- final edge-side validation,
- and feasibility screening before operating-point selection.

Accordingly, historical-case evidence conditions the advisory generation process, whereas deterministic policy rules determine whether the resulting proposal can influence the executable edge-side search.

## 6. Optional reprocessing/reconfiguration parameters

The implementation also contains an optional event-triggered reprocessing branch in which selected signal-processing or augmentation parameters may be adjusted.

These parameters are **not the routine threshold-space control variables used by cached-score replay**. They are relevant only when the system explicitly enters a separate reprocessing or reconfiguration path.

The following ranges are defined in `llm_advisor.py::PARAM_RANGES`:

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

Parameters not included in the allowlist are ignored if emitted by the LLM.

The secondary policy constraints exposed by `cloud_agent_v2.py::PolicyGuard` include:

| Parameter | Cloud guard range | Cloud max step |
|---|---:|---:|
| `retain_ratio` | [0.35, 0.65] | 0.08 |
| `engineering_aug_prob` | [0.50, 0.85] | 0.10 |
| `channel_aug_prob` | [0.70, 1.00] | -- |

`channel_aug_prob` remains step-limited to 0.10 during the first numerical validation pass.

These reprocessing parameters are documented here for implementation reproducibility, but they should be distinguished from the threshold-space lightweight-guidance variables used by the main edge-control recovery mechanism.

## 7. Separation from the per-signal authentication path

The policy-guard and LLM-guidance mechanisms operate outside the normal per-signal authentication path.

During routine authentication:

```text
received IQ
    ->
RFF feature extraction / inference
    ->
closed-set and open-set decision evidence
    ->
authentication output
```

does not require an LLM call.

When feasibility degradation is detected, the edge first attempts cached-score threshold replay locally. Only when the local mechanism is insufficient is an escalation event sent to the cloud-side coordination path.

The event-driven path is therefore:

```text
Edge state summary
    ->
Cloud retrieval
    ->
RAG-enabled LLM advisory generation
    ->
Deterministic policy validation
    ->
Bounded guidance
    ->
Edge-side safe-set restriction / validation
    ->
Cached-score operating-point evaluation
```

An optional signal-reprocessing branch may be invoked separately when explicitly required, but this is not equivalent to routine lightweight threshold replay.

This distinction is important when interpreting the latency measurements in the manuscript: millisecond-scale routine authentication, cached-score operating-point correction, event-triggered reprocessing, LLM advisory generation, and model reprovisioning correspond to different operating regimes and should not be treated as a single per-signal execution path.

## 8. Proposal outcome definitions for quantitative reporting

Because unsafe numerical values are corrected instead of being directly executed, proposal-level auditing distinguishes among unchanged proposals, guard-adjusted proposals, and cases in which no LLM-generated update is executable.

The following definitions are used:

- **Reviewed proposal**: an LLM output that is successfully parsed and contains at least one allowlisted numerical proposal reaching deterministic policy validation.
- **Directly accepted**: the proposal already satisfies all applicable deterministic constraints and requires no numerical clipping or projection.
- **Guard-adjusted**: at least one proposed value violates a safe-range or maximum-step constraint and is therefore clipped or projected before further use.
- **No executable LLM update / fallback**: parsing fails, all proposed fields are unsupported or invalid, the LLM is unavailable, or no executable guidance remains after validation. The system then uses a conservative cloud-side suggestion or deterministic retrieval-based fallback.

Recommended statistics are:

```text
N_generated
N_parse_failure
N_reviewed
N_directly_accepted
N_guard_adjusted
N_no_executable_update

Direct acceptance rate
    = N_directly_accepted / N_reviewed

Guard intervention rate
    = N_guard_adjusted / N_reviewed

Number of range-clamp events
Number of step-clamp events
Per-parameter clamp counts
Number of deterministic fallback events
```

These statistics should be reported from the actual experimental logs rather than inferred from the policy definition.


## 9. Architectural interpretation

The implementation should be interpreted according to the functional boundary used in the manuscript:

```text
RAN edge
----------------------------------------
RFF authentication subsystem
        |
        v
Edge control agent
  - feasibility monitoring
  - cached-score threshold replay
  - Pareto-filtered selection
  - final deterministic enforcement
        |
        | escalation event
        v

Cloud coordination path
----------------------------------------
Cloud coordination agent
        |
        +--> historical-case retrieval
        |
        +--> RAG-enabled LLM advisory agent
        |       |
        |       +--> bounded candidate proposal
        |
        +--> deterministic policy validation
        |
        +--> deterministic retrieval-based fallback
        |
        v

Validated bounded guidance
        |
        v

RAN edge
----------------------------------------
safe-set restriction
cached-score evaluation
final operating-point update
```

The LLM is therefore an **evidence-conditioned proposal generator**, not an authentication engine and not a direct controller. The final executable behavior remains determined by explicit numerical constraints, edge-side feasibility evaluation, and deterministic operating-point selection.

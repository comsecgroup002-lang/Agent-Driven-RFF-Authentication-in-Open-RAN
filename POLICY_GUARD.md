# Policy Guard: Reproducibility and Safety Details

This document supplements Section 5.3.3 and Table 1 of the manuscript by providing implementation-level details of the policy guard applied to LLM-generated guidance. Its scope is limited to the validation of RAG-enabled LLM advisory proposals before they can influence the edge-side operating-point search.

The policy guard does not perform authentication, does not generate recovery guidance by itself, and does not introduce an additional recovery stage. Its role is to ensure that only bounded and structurally valid LLM-generated proposals are passed to the edge control process.

## 1. Functional boundary

The RAG-enabled LLM advisory agent is part of the cloud-side coordination path and is invoked only after edge-side local correction cannot restore a feasible operating point.

- The RFF authentication subsystem remains responsible for signal-level authentication.
- The LLM does not process raw IQ samples for authentication.
- The LLM does not produce the final authentication decision.
- The LLM does not directly execute edge-side control actions.
- The LLM does not directly modify model weights, the feature extractor, or the classifier.
- The LLM generates only a bounded candidate proposal in the operating-point space used by the edge control agent.
- Every LLM-derived proposal must pass deterministic validation before it can be used by the edge.

Authentication feasibility follows the definition used in the revised manuscript:

\[
A_c \geq \gamma_c,\qquad A_o \geq \gamma_o .
\]

The rejection rate \(R\) is retained in the runtime state as an auxiliary service-availability indicator and may be considered during operating-point selection, but it is not an additional hard feasibility condition in the current evaluation protocol.

## 2. Advisory input and output

For each escalation event, the cloud coordination agent forms a compact advisory context from the runtime state and the retrieved experience-bank records.

### Query state

The query state contains:

- closed-set accuracy \(A_c\);
- open-set AUROC \(A_o\);
- rejection rate \(R\);
- current feasibility gaps;
- recent threshold-offset state;
- low-order score-distribution summaries.

### Retrieved evidence

For the default \(K=5\) configuration, the RAG context contains:

- the three most similar successful historical cases;
- the two most similar failed historical cases.

Successful cases provide evidence about previously effective operating regions. Failed cases are retained as negative contextual evidence and are not converted into deterministic rejection rules.

### LLM input context

The LLM receives:

- current performance state;
- current adjustable operating-point parameters;
- feasibility-gap diagnosis;
- retrieved historical cases;
- explicit safety constraints.

### LLM output

The advisory agent returns a structured JSON proposal containing:

- `analysis`;
- `param_changes`;
- `reasoning`;
- `confidence`;
- `requires_reprocessing`.

The output is always treated as a proposal. The presence or value of `requires_reprocessing` does not bypass the policy guard and does not by itself trigger model training, model replacement, or any other executable action.

## 3. Guarded operating-point space

The lightweight-guidance path operates in the same threshold-offset space used by the edge control agent.

Let

\[
\delta=(\delta_F,\delta_M)
\]

denote the offsets applied to the fused novelty threshold and the margin threshold, respectively.

The implementation uses the following names:

- `delta_fused` for \(\delta_F\);
- `delta_margin` for \(\delta_M\).

Only variables explicitly included in the allowlist can be modified by an LLM proposal. Unsupported keys are discarded.

The implementation-level global bounds are:

| Parameter | Manuscript notation | Safe range | Maximum single-step change |
|---|---|---:|---:|
| `delta_fused` | \(\delta_F\) | [-0.15, 0.35] | 0.05 |
| `delta_margin` | \(\delta_M\) | [0.05, 0.35] | 0.05 |

The safe ranges define the global admissible region for LLM-generated operating-point guidance. The maximum-step constraints limit the change relative to the current offset state during a single advisory event.

No LLM-generated update to signal-augmentation parameters, optimizer settings, model weights, feature-extractor parameters, or raw signal-processing parameters is permitted by the policy guard.

## 4. Two-stage validation procedure

The implementation follows the two-stage validation procedure described in the response letter.

### Stage 1: cloud-side proposal validation

`LLMAdvisor._validate_advice()` validates the parsed LLM proposal before it is returned as executable guidance.

For each proposed parameter:

- verify that the key belongs to the allowlist;
- verify that the value is numeric and finite;
- limit the change according to the predefined maximum single-step bound;
- project the resulting value into the parameter-specific safe range;
- verify that the resulting bounded proposal defines a non-empty admissible search region.

The last check is the proposal-level feasibility screening referred to in Section 5.3.3: it verifies consistency with the configured admissible operating-point region. It does not claim that the proposed operating point already satisfies the measured authentication objectives. Final authentication feasibility is evaluated by the edge-side cached-score procedure.

For an allowed parameter \(k\), the numerical correction is

\[
v_k =
\operatorname{Proj}_{[l_k,u_k]}
\left(
\theta_k+
\operatorname{clip}
(p_k-\theta_k,-s_k,s_k)
\right),
\]

where:

- \(p_k\) is the LLM-proposed value;
- \(\theta_k\) is the current value;
- \([l_k,u_k]\) is the allowed interval;
- \(s_k\) is the maximum permitted single-step change.

If parsing fails or no valid allowlisted parameter remains, no executable LLM-derived update is produced.

### Stage 2: edge-side final review

`EdgeAgentV2._policy_guard_review()` performs a second deterministic check before the proposal is allowed to affect the edge-side search.

For each retained parameter:

- verify the cloud-provided admissible range;
- verify the cloud-provided maximum step size;
- correct any residual range violation;
- correct any residual step-size violation;
- discard invalid or non-finite values.

Only the post-review parameter dictionary can be passed to the edge control process.

## 5. Feasibility screening and application

Two different checks are distinguished.

- **Proposal-level feasibility screening** is performed as part of the policy guard. It verifies that the bounded proposal is structurally consistent with the configured admissible operating-point region.
- **Authentication-feasibility screening** is performed by the edge control process after cached-score evaluation. It determines whether the resulting operating point satisfies the authentication objectives.

Passing the policy guard therefore does not mean that an LLM proposal is automatically accepted as a feasible operating point.

The validated proposal is used only to initialize or restrict the subsequent cached-score operating-point search.

The edge control process then:

- evaluates candidate operating points using cached authentication evidence;
- computes the corresponding \(A_c\), \(A_o\), and auxiliary rejection statistics;
- retains feasible candidates satisfying \(A_c\geq\gamma_c\) and \(A_o\geq\gamma_o\);
- performs the edge-side operating-point selection procedure on the admissible candidates.

Therefore, the LLM only narrows or redirects the search. The final operating point remains determined by the deterministic edge-side evaluation and selection procedure.

## 6. Proposal failure handling

The policy guard is corrective rather than purely binary.

- If a numerical value exceeds a safe range, it is projected into the admissible interval when possible.
- If a proposed change exceeds the maximum single-step bound, the change is limited before use.
- If unsupported or invalid fields are present, those fields are removed.
- If no executable LLM-derived update remains after validation, the LLM proposal is not applied to the edge system.
- The policy guard itself does not generate a replacement guidance proposal.
- Subsequent handling follows the framework-level recovery logic defined outside the policy guard.

Cloud delay or service unavailability is not treated as an LLM policy-validation outcome. Such interruption is handled by the edge-local fallback advisor described in Section 5.4.

## 7. Historical-case treatment

Historical cases are used as evidence for RAG-enabled advisory generation.

- Successful cases provide examples of operating regions associated with previous recovery.
- Failed cases provide negative contextual evidence about previously ineffective adjustments.
- Failed cases do not act as hard rejection rules.
- Historical similarity does not override the numerical policy constraints.
- Executability is determined by the explicit deterministic validation rules.

The experience bank therefore conditions the LLM proposal, while the policy guard independently determines whether the resulting proposal is admissible.

## 8. Pseudocode

```text
Algorithm: Policy-Guarded LLM Advisory

Input:
    raw LLM output R
    current operating-point state theta
    parameter allowlist A
    global safe ranges S
    maximum single-step limits Delta
    cloud-provided ranges S_cloud
    cloud-provided step limits Delta_cloud

Output:
    validated LLM-derived update P_star
    or NO_EXECUTABLE_UPDATE

1:  P <- parse_json(R)

2:  if parsing fails:
3:      return NO_EXECUTABLE_UPDATE

4:  V <- empty dictionary

5:  for each (key, proposed_value) in P.param_changes:
6:      if key not in A:
7:          continue
8:      if proposed_value is not numeric or not finite:
9:          continue

10:     current <- theta[key]
11:     step <- Delta[key]

12:     value <- current
            + clip(proposed_value-current, -step, +step)

13:     value <- clip(value, S[key].lower, S[key].upper)
14:     V[key] <- value

15: if V is empty:
16:     return NO_EXECUTABLE_UPDATE

17: if V does not define a non-empty admissible search region:
18:     return NO_EXECUTABLE_UPDATE

19: P_star <- empty dictionary

20: for each (key, value) in V:
21:     current <- theta[key]

22:     if value is outside S_cloud[key]:
23:         value <- clip(
                value,
                S_cloud[key].lower,
                S_cloud[key].upper
            )

24:     if abs(value-current) > Delta_cloud[key]:
25:         value <- current
                + sign(value-current) * Delta_cloud[key]

26:     if value is finite:
27:         P_star[key] <- value

28: if P_star is empty:
29:     return NO_EXECUTABLE_UPDATE

30: use P_star only to initialize/restrict
    the cached-score operating-point search

31: perform deterministic authentication-feasibility screening
    and edge-side operating-point selection

32: record the proposal outcome and guard interventions

33: return P_star
```

## 9. Proposal outcome definitions

For quantitative auditing, one LLM advisory invocation that enters the parsing and validation procedure is counted as one advisory attempt.

Each attempt is assigned to exactly one of the following outcome categories.

### Directly accepted without modification

A proposal is classified as directly accepted when:

- parsing succeeds;
- at least one allowlisted executable parameter is present;
- no range correction is required;
- no maximum-step correction is required;
- the final edge-side review does not modify the proposal.

### Guard-adjusted and subsequently executable

A proposal is classified as guard-adjusted when:

- one or more values require range projection or step limiting;
- a non-empty executable LLM-derived update remains after validation.

Only the corrected proposal is allowed to proceed.

### No executable LLM-derived update

A proposal is classified in this category when:

- parsing fails;
- all proposed fields are unsupported;
- all retained values are invalid or non-finite;
- no admissible LLM-derived update remains after deterministic validation.

No raw or partially invalid LLM proposal is executed in this case.

## 10. Quantitative audit statistics

The audited LLM advisory outcomes reported in the response letter are:

| Audit outcome | Number of advisory attempts | Rate |
|---|---:|---:|
| Total LLM advisory attempts | 120 | 100.0% |
| Directly accepted without modification | 93 | 77.5% |
| Guard-adjusted and subsequently executable | 22 | 18.3% |
| No executable LLM-derived update | 5 | 4.2% |
| Non-direct outcomes | 27 | 22.5% |
| Executable LLM-derived updates after validation | 115 | 95.8% |

The reported rates use the total number of LLM advisory attempts as the denominator:

\[
\frac{93}{120}=77.5\%,\qquad
\frac{22}{120}=18.3\%,\qquad
\frac{5}{120}=4.2\%.
\]

Accordingly:

\[
\frac{22+5}{120}=22.5\%
\]

of the attempts required either corrective guard handling or resulted in no executable LLM update, while

\[
\frac{93+22}{120}=95.8\%
\]

produced an executable LLM-derived update after deterministic validation.

These statistics characterize policy-guard behavior only. They do not measure the effectiveness of the separate deterministic retrieval-only ablation arm or the edge-local interruption fallback advisor.

## 11. Audit information

For each advisory attempt, the implementation records sufficient information to determine how the proposal was handled.

The audit record includes:

- proposal identifier;
- run identifier;
- evaluation-domain identifier;
- raw parsed proposal;
- validated proposal;
- final post-review proposal;
- proposal outcome;
- range-correction count;
- step-correction count;
- per-parameter correction information.

These records support recomputation of the proposal-level counts reported above.

## 12. Relevant implementation locations

### `llm_advisor.py`

- `SYSTEM_PROMPT`: structured LLM output requirements and safety instructions.
- `PARAM_RANGES`: allowlisted operating-point parameters and safe intervals.
- `MAX_STEP_SIZE`: maximum single-step changes.
- `_parse_response()`: JSON response parsing.
- `_validate_advice()`: first-stage allowlist, type, range, and step validation.

### `cloud_agent_v2.py`

- retrieval path: constructs the RAG context from similar successful and failed cases.
- `PolicyGuard.SAFE_RANGES`: safe ranges returned with cloud-side guidance.
- `PolicyGuard.MAX_STEP`: step-size constraints returned with cloud-side guidance.
- lightweight-guidance path: packages the validated LLM proposal and policy constraints for the edge.

### `edge_agent_v2.py`

- escalation path: sends the compact runtime state to the cloud after local correction is insufficient.
- `_policy_guard_review()`: final edge-side deterministic review.
- `get_search_grid()`: applies the validated guidance to the bounded threshold-search region.
- guidance application: uses only the post-guard parameter dictionary.
- fallback-advisor path: handles edge-cloud interruption independently of LLM proposal validation.


## 13. Safety interpretation

The Policy Guard provides deterministic constraints on LLM-generated guidance. It does not establish adversarial robustness of the underlying RFF authentication model.

Its responsibility is limited to ensuring that:

- raw LLM output is never applied directly;
- unsupported or invalid fields are removed;
- admissible numerical changes remain within predefined ranges;
- single-step changes remain bounded;
- the final edge-side search remains subject to deterministic feasibility evaluation.

The resulting safety boundary is:

```text
RAG-enabled LLM proposal
        ->
cloud-side deterministic validation
        ->
bounded operating-point proposal
        ->
edge-side deterministic review
        ->
cached-score search and feasibility screening
```

This preserves the functional boundary stated in the manuscript: the LLM acts as a bounded advisory component, while authentication and executable operating-point decisions remain governed by deterministic edge-side logic.

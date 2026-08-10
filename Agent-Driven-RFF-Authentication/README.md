# Agent-Driven RFF Authentication in Open RAN

This directory contains the executable implementation corresponding to the revised manuscript **An Agent-Driven Edge–Cloud Decision Framework for Radio-Frequency Fingerprint Authentication in Open Radio Access Networks**.

## Architecture implemented here

The runtime path is aligned with Sections 5–7 of the revised manuscript:

1. The RFF authentication subsystem runs at the edge.
2. The edge control agent performs cached-score threshold replay and Pareto-filtered operating-point selection.
3. Authentication feasibility is defined by `Ac >= 0.90` and `Ao >= 0.85`; rejection rate `R` is retained only as an auxiliary service-availability indicator for Pareto/selection behavior.
4. If local correction remains infeasible, the edge sends compact telemetry to the cloud.
5. The cloud retrieves the top three similar successful cases and top two similar failed cases and, in the RAG arm, invokes the cloud-side Qwen2.5-3B-Instruct advisory model.
6. The LLM generates only bounded threshold/search guidance. It never authenticates signals, changes signal augmentation, retrains the model, or directly executes edge actions.
7. `LLMAdvisor._validate_advice()` performs the first deterministic validation pass; `EdgeAgentV2._policy_guard_review()` performs the final edge-side validation.
8. The retrieval-only deterministic arm aggregates the `delta_fused` and `delta_margin` offsets from up to the top three successful cases using normalized nonnegative cosine-similarity weights, without an LLM call.
9. Model refresh is source-model reprovisioning. No cross-edge model aggregation is used.
10. Cloud delay/unavailability is handled separately by the edge-local fallback advisor.

See `POLICY_GUARD.md` for the policy-guard specification and audit definitions.

## Dependencies

Install the runtime dependencies before starting the edge or cloud services:

```bash
pip install -r requirements.txt
```

The Qwen model path and source-model/pretrained-weight paths are configured locally in `cloud_config.yaml` and `config.yaml`; model or dataset binaries are not bundled in this code package.

## Data protocol

For the LoRa source-domain preparation, the default configuration uses all ten IQ recordings (`IQ_1`–`IQ_10`). The first 10 devices are enrolled. Device 11 is excluded from source training and validation and appears only in the open-set test split. The source-domain split uses non-overlapping 8:1:1 temporal blocks.

For cross-domain operation, configure the `data.*.test` paths to the completely held-out target domain. During cached-score adaptation, target records are divided into disjoint contiguous runtime-control and final-evaluation buffers. **Only the runtime-control buffer can select an operating point, trigger escalation, drive cloud guidance, or trigger model refresh; the final-evaluation buffer is report-only.** This prevents final reported labels from feeding back into the recovery path.

## Data availability

The LoRa experiments use the publicly available OSU LoRa RF fingerprinting dataset cited in the manuscript. The real-world 5G NR dataset cannot be redistributed because the raw IQ measurements are subject to confidentiality and data-use restrictions. Accordingly, this package contains no private 5G NR recordings or synthetic substitutes. The manuscript provides the 5G NR device population, acquisition configuration, session organization, sample scale, partitioning, and evaluation protocol.

## Start the cloud

```bash
python cloud_agent_v2.py --mode lightweight
```

or

```bash
python cloud_agent_v2.py --mode model_reprovision
```

or

```bash
python cloud_agent_v2.py --mode auto
```

The RAG-enabled LLM is configured in `cloud_config.yaml` and is loaded only by the cloud process.

## Start the edge

Offline edge execution:

```bash
python edge_agent_v2.py --config config.yaml
```

Cloud-connected execution:

```bash
python edge_agent_v2.py --config config.yaml --cloud --cloud-server http://CLOUD_IP:5000
```

Ablation guidance strategies can be selected without changing the architecture:

```bash
# LLM generation without retrieval
python edge_agent_v2.py --config config.yaml --cloud --guidance-strategy llm_only

# Retrieval-only deterministic guidance
python edge_agent_v2.py --config config.yaml --cloud --guidance-strategy deterministic

# Retrieval + LLM synthesis (default)
python edge_agent_v2.py --config config.yaml --cloud --guidance-strategy rag
```

For the four-arm cloud-guidance ablation, use the same seed-specific source model/operating state and the same historical-case snapshot for all arms. After preparing the experience bank, set `experience_bank.read_only: true` in `cloud_config.yaml` while running `llm_only`, `deterministic`, and `rag`; this prevents one arm from changing the retrieval corpus seen by another arm.

## Policy-guard audit statistics

The cloud writes finalized LLM policy-guard events to `cloud_data/policy_guard_events.jsonl` at runtime. `/stats` recomputes proposal counts and rates from this audit log, including directly accepted, guard-adjusted, no-executable, range-clamp, and step-clamp statistics.

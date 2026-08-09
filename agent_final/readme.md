# Agent-Driven RFF Authentication: Edge--Cloud Prototype

This revision follows the manuscript architecture directly:

- **Edge control agent:** routine RFF authentication, feasibility monitoring, cached-score threshold replay, Pareto-filtered selection, and final deterministic validation.
- **Cloud coordination agent:** historical-case retrieval, RAG-enabled LLM advisory generation, deterministic retrieval-only guidance, policy validation, and source-model reprovisioning.
- **Feasibility:** determined only by closed-set accuracy `A_c` and open-set AUROC `A_o`. `reject_rate` is retained as an auxiliary service-availability/ranking metric.
- **Lightweight guidance:** threshold-space guidance only; it does not trigger edge-side LLM inference or target-domain retraining.
- **Model refresh:** re-provisions the original source-trained model package. Target-domain model aggregation is disabled.

## 1. Configure the cloud-side LLM

Set the LLM path in `cloud_config.yaml`:

```yaml
llm:
  enable: true
  model_path: '/set/your/path/llm_model'
```

The edge configuration no longer contains an LLM model path.

## 2. Start the Cloud Coordination Agent

### RAG-enabled guidance

```bash
python cloud_agent_v2.py --mode lightweight --guidance-variant rag
```

### LLM-only ablation

```bash
python cloud_agent_v2.py --mode lightweight --guidance-variant llm_only
```

### Deterministic retrieval-only ablation

```bash
python cloud_agent_v2.py --mode lightweight --guidance-variant deterministic
```

### Source-model reprovisioning

```bash
python cloud_agent_v2.py --mode model_reprovision
```

### Automatic hierarchical mode

```bash
python cloud_agent_v2.py --mode auto --guidance-variant rag
```

## 3. Start the Edge Control Agent

### Offline / edge-only mode

The edge agent checks for source-trained weights. If they do not yet exist, it trains the configured source-domain model once. It then performs local cached-score threshold replay. No cloud-side LLM is used.

```bash
python edge_agent_v2.py --config config.yaml
```

### Cloud-connected mode

```bash
python edge_agent_v2.py \
  --config config.yaml \
  --cloud \
  --cloud-server http://192.168.150.3:5000
```

In lightweight mode, the execution sequence is:

```text
local threshold replay
    -> if infeasible: send compact state to cloud
    -> cloud retrieval / LLM advisory / policy guard
    -> bounded threshold interval returned to edge
    -> final edge-side validation
    -> cached-score threshold replay within the bounded interval
```

The edge never loads the LLM model in this path.

## 4. Feasibility rule

A result is feasible if and only if:

```text
closed_acc >= min_closed_acc
and
open_auc >= target_open_auc
```

`reject_rate` remains available for Pareto ranking and service-availability analysis but is not a hard feasibility condition.

## 5. Consistency tests

Run the included logic tests before launching the full signal-processing pipeline:

```bash
python -m unittest -v test_architecture_consistency.py
```

The tests cover the `A_c/A_o` feasibility rule, threshold-space policy guard, deterministic retrieval guidance, cloud-side guidance routing, and source-only model-refresh behavior.

## 6. Policy-guard audit

The cloud service exposes proposal-audit counters through:

```bash
curl http://localhost:5000/stats
```

The returned `policy_audit` object includes reviewed proposals, directly accepted proposals, guard-adjusted proposals, deterministic fallbacks, range/step clamp events, and the derived direct-acceptance / guard-intervention rates. See `POLICY_GUARD.md` for the exact definitions.

## 7. Dependencies

Install the packages listed in `requirements.txt`. `accelerate` and `bitsandbytes` are optional and are used only when the configured cloud-side LLM loading mode benefits from them.

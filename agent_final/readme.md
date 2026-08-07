## 1. Start the Cloud Agent

Select the desired assistance mode:

### Lightweight Guidance

```bash
python cloud_agent_v2.py --mode lightweight
```

### Model Reprovisioning

```bash
python cloud_agent_v2.py --mode model_reprovision
```

### Automatic Mode

```bash
python cloud_agent_v2.py --mode auto
```

## 2. Start the Edge Agent

### Offline Mode

Run the edge agent locally. The agent automatically checks whether pretrained model weights are available:

* If weights are found, it directly performs evaluation.
* If no weights are found, it trains the model first.

```bash
python edge_agent_v2.py --config config.yaml
```

### Cloud-Connected Mode

Run the edge agent with cloud coordination enabled:

```bash
python edge_agent_v2.py --config config.yaml --cloud --cloud-server http://192.168.150.3
```

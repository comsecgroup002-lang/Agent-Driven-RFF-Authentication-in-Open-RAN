# -*- coding: utf-8 -*-
"""Edge control agent for the revised RFF edge-cloud decision framework.

Routine operation and lightweight recovery remain edge-local and use cached
scores. The edge never hosts the LLM. Cloud guidance only narrows/initializes
the threshold-search space and is revalidated deterministically before use.
"""

from __future__ import annotations

import copy
import json
import math
import os
import socket
import tarfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from threshold_search import DEFAULT_SAFE_RANGES, build_search_grid, is_feasible, search_thresholds


@dataclass
class AdaptationResult:
    episode_id: int
    round_id: int
    timestamp: str
    threshold_config: Dict[str, float]
    metrics: Dict[str, float]
    control_metrics: Dict[str, float]
    score_stats: Dict[str, float]
    success: bool
    assistance_type: str
    guidance_source: str = "edge_only"
    policy_approved: Optional[bool] = None
    policy_warnings: List[str] = field(default_factory=list)


@dataclass
class OptimizationState:
    current_round: int = 0
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_config: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    objectives_met: bool = False
    at_pareto_limit: bool = False


class CloudUnavailable(RuntimeError):
    pass


class EdgeAgentV2:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.original_cfg = copy.deepcopy(cfg)
        self.paths = cfg["paths"]
        self.objectives = cfg.get("agent", {}).get("objectives", {
            "min_closed_acc": 0.90,
            "target_open_auc": 0.85,
        })
        self.max_rounds = int(cfg.get("agent", {}).get("max_rounds", 5))

        cloud_cfg = cfg.get("cloud", {})
        self.cloud_enabled = bool(cloud_cfg.get("enabled", False))
        self.cloud_server = str(cloud_cfg.get("server_url", "http://localhost:5000")).rstrip("/")
        self.cloud_timeout = float(cloud_cfg.get("timeout_seconds", 15.0))
        self.guidance_strategy = str(cloud_cfg.get("guidance_strategy", "rag"))
        if self.guidance_strategy not in {"rag", "llm_only", "deterministic"}:
            raise ValueError("cloud.guidance_strategy must be rag, llm_only, or deterministic")

        self.terminal_id = str(cloud_cfg.get("terminal_id") or self._gen_id())
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = OptimizationState()
        self.episode_count = 0
        self.all_results: List[AdaptationResult] = []
        self._features: Optional[Dict] = None
        self._model = None
        self.current_guidance: Optional[Dict] = None
        self.last_edge_result: Optional[Dict] = None

        print(f"[EdgeAgentV2] terminal={self.terminal_id}")
        print(f"[EdgeAgentV2] cloud_enabled={self.cloud_enabled}")
        print(f"[EdgeAgentV2] guidance_strategy={self.guidance_strategy}")
        print("[EdgeAgentV2] LLM placement=cloud only")

    @staticmethod
    def _gen_id() -> str:
        return f"{socket.gethostname()}_{uuid.getnode():012x}"[:32]

    def check_weights(self) -> bool:
        path = str(self.paths.get("model_weights", ""))
        if not path or not os.path.isfile(path):
            return False
        try:
            state = torch.load(path, map_location="cpu")
            return isinstance(state, dict) and len(state) > 0
        except Exception:
            return False

    def prepare_data(self) -> bool:
        from data_pipeline import DataPipeline
        pipeline = DataPipeline(self.cfg)
        return pipeline.prepare_dataset(
            raw_dir=self.paths.get("raw_data", ""),
            processed_dir=self.paths.get("processed_data", ""),
            target_dir=self.paths["base"],
            label_map_file=self.paths["label_map"],
            force_reprocess=False,
            num_workers=self.cfg.get("signal_processing", {}).get("num_workers", 4),
        )

    def train_initial_model(self) -> bool:
        """Initial source-domain training only.

        Lightweight guidance and fallback never call this function.
        """
        try:
            import model as model_module
            model_module.train(self.cfg)
            self._features = None
            self._model = None
            return True
        except Exception as exc:
            print(f"[EdgeAgentV2] initial training failed: {exc}")
            return False

    @staticmethod
    def _stratified_contiguous_split(dataset, fraction: float, min_per_group: int = 1) -> Tuple[List[int], List[int]]:
        """Split target-domain records into disjoint control/evaluation blocks.

        Samples are already naturally ordered within each device. Splitting each
        label group contiguously avoids using final evaluation records to select
        the operating point while retaining known and unknown samples in both
        buffers whenever the data permit it.
        """
        groups: Dict[int, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            label = int(sample[3])
            groups.setdefault(label, []).append(idx)

        control: List[int] = []
        evaluation: List[int] = []
        for indices in groups.values():
            n = len(indices)
            if n < 2:
                evaluation.extend(indices)
                continue
            cut = int(round(n * fraction))
            cut = max(min_per_group, min(n - min_per_group, cut))
            control.extend(indices[:cut])
            evaluation.extend(indices[cut:])

        if not control or not evaluation:
            raise ValueError("Target test set is too small to create disjoint control/evaluation buffers")
        return sorted(control), sorted(evaluation)

    def extract_features(self) -> Dict:
        if self._features is not None:
            return self._features

        from model import ConvNeXtCosFace, ThreeBandDataset, energy_score, extract_logits_feats, get_margin, set_seed
        from sklearn.cluster import KMeans
        from sklearn.covariance import LedoitWolf

        seed = int(self.cfg.get("seed", 42))
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = Path(self.paths["base"])
        data_cfg = self.cfg["data"]

        with open(self.paths["label_map"], "r", encoding="utf-8") as f:
            label_map = json.load(f)
        n_classes = len(label_map)

        val_ds = ThreeBandDataset(
            base / data_cfg["low"]["val"],
            base / data_cfg["mid"]["val"],
            base / data_cfg["high"]["val"],
            label_map,
            train=False,
            cfg=self.cfg,
            filter_unknown=True,
        )
        test_ds = ThreeBandDataset(
            Path(data_cfg["low"]["test"]) if os.path.isabs(str(data_cfg["low"]["test"])) else base / data_cfg["low"]["test"],
            Path(data_cfg["mid"]["test"]) if os.path.isabs(str(data_cfg["mid"]["test"])) else base / data_cfg["mid"]["test"],
            Path(data_cfg["high"]["test"]) if os.path.isabs(str(data_cfg["high"]["test"])) else base / data_cfg["high"]["test"],
            label_map,
            train=False,
            cfg=self.cfg,
            filter_unknown=False,
        )

        rb_cfg = self.cfg.get("runtime_buffer", {})
        control_fraction = float(rb_cfg.get("control_fraction", 0.5))
        if not 0.0 < control_fraction < 1.0:
            raise ValueError("runtime_buffer.control_fraction must be between 0 and 1")
        ctrl_idx, eval_idx = self._stratified_contiguous_split(
            test_ds,
            control_fraction,
            int(rb_cfg.get("min_samples_per_group", 1)),
        )
        control_ds = Subset(test_ds, ctrl_idx)
        eval_ds = Subset(test_ds, eval_idx)

        train_cfg = self.cfg.get("train", {})
        loader_kwargs = {
            "batch_size": int(train_cfg.get("batch_size", 128)),
            "shuffle": False,
            "num_workers": int(train_cfg.get("num_workers", 4)),
            "pin_memory": bool(train_cfg.get("pin_memory", True)),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))

        val_loader = DataLoader(val_ds, **loader_kwargs)
        ctrl_loader = DataLoader(control_ds, **loader_kwargs)
        eval_loader = DataLoader(eval_ds, **loader_kwargs)

        model_cfg = self.cfg.get("model", {})
        model = ConvNeXtCosFace(
            n_classes=n_classes,
            backbone=model_cfg.get("backbone", "convnext_tiny"),
            pretrained_try=False,
            channels_last=bool(model_cfg.get("channels_last", True)),
        ).to(device)
        model.load_state_dict(torch.load(self.paths["model_weights"], map_location=device), strict=True)
        model.eval()
        self._model = model

        log_val, feat_val, y_val = extract_logits_feats(model, val_loader, device)
        log_ctrl, feat_ctrl, y_ctrl = extract_logits_feats(model, ctrl_loader, device)
        log_eval, feat_eval, y_eval = extract_logits_feats(model, eval_loader, device)

        if np.any(y_val < 0):
            raise ValueError("Source validation data must contain enrolled devices only")
        if np.unique((y_ctrl == -1).astype(int)).size < 2 or np.unique((y_eval == -1).astype(int)).size < 2:
            raise ValueError("Both runtime control and evaluation buffers must contain enrolled and unknown samples")

        en_val = energy_score(log_val)
        en_ctrl = energy_score(log_ctrl)
        en_eval = energy_score(log_eval)
        margin_val = get_margin(log_val)
        margin_ctrl = get_margin(log_ctrl)
        margin_eval = get_margin(log_eval)

        k_centroids = int(self.cfg.get("open_set", {}).get("mahalanobis", {}).get("k_centroids", 3))
        class_stats = {}
        for c in sorted(np.unique(y_val)):
            Xc = feat_val[y_val == c]
            if len(Xc) == 0:
                continue
            k = min(k_centroids, len(Xc))
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xc)
            lw = LedoitWolf().fit(Xc)
            class_stats[int(c)] = (km.cluster_centers_, np.linalg.pinv(lw.covariance_))

        if not class_stats:
            raise ValueError("No enrolled validation class statistics could be constructed")

        def mahal(features: np.ndarray) -> np.ndarray:
            out = np.empty(len(features), dtype=np.float32)
            for i, x in enumerate(features):
                best = float("inf")
                for centers, inv_cov in class_stats.values():
                    for mu in centers:
                        diff = x - mu
                        value = float(np.sqrt(max(0.0, diff @ inv_cov @ diff.T)))
                        best = min(best, value)
                out[i] = best
            return out

        md_val = mahal(feat_val)
        md_ctrl = mahal(feat_ctrl)
        md_eval = mahal(feat_eval)

        def zscore_reference(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
            return (x - float(ref.mean())) / (float(ref.std()) + 1e-6)

        ez_val = zscore_reference(en_val, en_val)
        ez_ctrl = zscore_reference(en_val, en_ctrl)
        ez_eval = zscore_reference(en_val, en_eval)
        dz_val = zscore_reference(md_val, md_val)
        dz_ctrl = zscore_reference(md_val, md_ctrl)
        dz_eval = zscore_reference(md_val, md_eval)
        mz_val = zscore_reference(margin_val, margin_val)
        mz_ctrl = zscore_reference(margin_val, margin_ctrl)
        mz_eval = zscore_reference(margin_val, margin_eval)

        energy_cfg = self.cfg.get("open_set", {}).get("energy", {})
        alpha_grid = energy_cfg.get("alpha_grid", [i / 10 for i in range(11)])
        alpha = float(energy_cfg.get("alpha_init", 0.6))
        if bool(energy_cfg.get("auto_alpha", True)):
            best_obj = float("inf")
            pred_val = log_val.argmax(axis=1)
            aq = float(self.cfg.get("open_set", {}).get("calibration", {}).get("accept_quantile", 0.90))
            for candidate in alpha_grid:
                a = float(candidate)
                fused = a * ez_val + (1.0 - a) * dz_val
                thresholds = {
                    int(c): float(np.quantile(fused[pred_val == c], aq))
                    for c in np.unique(pred_val) if np.any(pred_val == c)
                }
                rejected = fused > np.asarray([thresholds.get(int(p), 1e9) for p in pred_val])
                objective = float(rejected.mean()) + float((pred_val != y_val).mean())
                if objective < best_obj:
                    best_obj = objective
                    alpha = a

        fused_val = alpha * ez_val + (1.0 - alpha) * dz_val
        self._features = {
            "log_val": log_val,
            "y_val": y_val,
            "ez_val": ez_val,
            "dz_val": dz_val,
            "mz_val": mz_val,
            "log_control": log_ctrl,
            "y_control": y_ctrl,
            "ez_control": ez_ctrl,
            "dz_control": dz_ctrl,
            "mz_control": mz_ctrl,
            "log_eval": log_eval,
            "y_eval": y_eval,
            "ez_eval": ez_eval,
            "dz_eval": dz_eval,
            "mz_eval": mz_eval,
            "alpha": alpha,
            "fused_mean": float(fused_val.mean()),
            "fused_std": float(fused_val.std()),
            "margin_mean": float(mz_val.mean()),
            "margin_std": float(mz_val.std()),
            "control_samples": len(ctrl_idx),
            "evaluation_samples": len(eval_idx),
        }
        print(
            f"[EdgeAgentV2] cached scores ready: alpha={alpha:.2f}, "
            f"control={len(ctrl_idx)}, evaluation={len(eval_idx)}"
        )
        return self._features

    def get_search_grid(self, guidance: Optional[Dict] = None) -> Dict[str, List[float]]:
        """Return the globally safe grid after optional cloud restriction."""
        return build_search_grid(self.cfg, guidance)

    def threshold_search(self, guidance: Optional[Dict] = None) -> Tuple[Dict, List[Dict]]:
        features = self.extract_features()
        previous = self.state.best_config or {
            "accept_quantile": self.cfg.get("open_set", {}).get("calibration", {}).get("accept_quantile", 0.90),
            "margin_quantile": self.cfg.get("open_set", {}).get("calibration", {}).get("margin_quantile", 0.20),
            "delta_fused": self.cfg.get("open_set", {}).get("calibration", {}).get("delta_fused", 0.10),
            "delta_margin": self.cfg.get("open_set", {}).get("calibration", {}).get("delta_margin", 0.10),
        }
        grid = self.get_search_grid(guidance)
        return search_thresholds(features, self.cfg, self.objectives, guidance, previous, grid_override=grid)

    def check_objectives(self, metrics: Dict[str, float]) -> bool:
        return is_feasible(metrics, self.objectives)

    def _operational_metrics(self, best: Dict) -> Dict[str, float]:
        """Metrics available to the online control loop.

        Stage activation, escalation, guidance, and refresh decisions use the
        runtime control buffer only. The disjoint evaluation buffer is report-
        only and therefore cannot influence the selected recovery path.
        """
        return best.get("control_metrics", best["metrics"])

    def _operational_feasible(self, best: Dict) -> bool:
        return self.check_objectives(self._operational_metrics(best))

    def _update_state(self, best: Dict, pareto: List[Dict]) -> None:
        self.state.current_round += 1
        metrics = dict(best["metrics"])
        config = dict(best["config"])
        self.state.history.append({
            "round": self.state.current_round,
            "metrics": metrics,
            "control_metrics": dict(best.get("control_metrics", {})),
            "config": config,
        })
        self.state.best_metrics = metrics
        self.state.best_config = config
        if len(self.state.history) >= 3:
            recent = [h["control_metrics"]["open_auc"] for h in self.state.history[-3:]]
            self.state.at_pareto_limit = max(recent) - min(recent) < 0.005
        self.state.objectives_met = self._operational_feasible(best)

    def _current_state_payload(self, best: Dict) -> Dict:
        features = self._features or {}
        return {
            # Never send the held-out reporting buffer to the control plane.
            "metrics": dict(self._operational_metrics(best)),
            "threshold_config": dict(best["config"]),
            "fused_mean": float(features.get("fused_mean", 0.0)),
            "fused_std": float(features.get("fused_std", 1.0)),
            "margin_mean": float(features.get("margin_mean", 0.0)),
            "margin_std": float(features.get("margin_std", 1.0)),
            "history": self.state.history[-3:],
        }

    def request_cloud_guidance(self, best: Dict, stage: str = "post_edge", strategy: Optional[str] = None) -> Dict:
        if not self.cloud_enabled:
            raise CloudUnavailable("cloud coordination is disabled")
        try:
            import requests
            response = requests.post(
                f"{self.cloud_server}/guidance",
                json={
                    "terminal_id": self.terminal_id,
                    "current_state": self._current_state_payload(best),
                    "strategy": strategy or self.guidance_strategy,
                    "stage": stage,
                },
                timeout=self.cloud_timeout,
            )
            if response.status_code != 200:
                raise CloudUnavailable(f"cloud returned HTTP {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise CloudUnavailable(str(exc)) from exc

    @staticmethod
    def _normalize_search_state(config: Optional[Dict]) -> Dict[str, float]:
        c = config or {}
        return {
            "accept_quantile": float(c.get("accept_quantile", 0.90)),
            "margin_quantile": float(c.get("margin_quantile", 0.20)),
            "delta_fused": float(c.get("delta_fused", c.get("rho", 0.10))),
            "delta_margin": float(c.get("delta_margin", 0.10)),
        }

    def _policy_guard_review(self, param_changes: Dict, safe_ranges: Optional[Dict] = None,
                             max_steps: Optional[Dict] = None) -> Tuple[bool, Dict, List[str]]:
        safe = safe_ranges or {k: list(v) for k, v in DEFAULT_SAFE_RANGES.items()}
        steps = max_steps or {k: 0.05 for k in DEFAULT_SAFE_RANGES}
        current = self._normalize_search_state(self.state.best_config)

        approved = True
        adjusted: Dict[str, float] = {}
        warnings: List[str] = []
        for raw_key, raw_value in (param_changes or {}).items():
            key = "delta_fused" if raw_key == "rho" else raw_key
            if key not in DEFAULT_SAFE_RANGES:
                approved = False
                warnings.append(f"unsupported:{raw_key}")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                approved = False
                warnings.append(f"non_numeric:{raw_key}")
                continue
            if not math.isfinite(value):
                approved = False
                warnings.append(f"non_finite:{raw_key}")
                continue

            glo, ghi = DEFAULT_SAFE_RANGES[key]
            raw_range = safe.get(key, [glo, ghi])
            lo, hi = max(glo, float(raw_range[0])), min(ghi, float(raw_range[1]))
            if lo > hi:
                lo, hi = glo, ghi
            step = min(0.05, float(steps.get(key, 0.05)))
            # The final value must satisfy range and step constraints
            # simultaneously. If their intersection is empty, this key is not
            # executable in the current advisory event.
            feasible_lo = max(lo, current[key] - step)
            feasible_hi = min(hi, current[key] + step)
            if feasible_lo > feasible_hi:
                approved = False
                warnings.append(f"no_step_safe_intersection:{key}")
                continue

            bounded = min(feasible_hi, max(feasible_lo, value))
            if bounded != value:
                approved = False
                if value < lo or value > hi:
                    warnings.append(f"range_clamp:{key}")
                if abs(value - current[key]) > step:
                    warnings.append(f"step_clamp:{key}")
            adjusted[key] = float(min(ghi, max(glo, bounded)))

        return approved, adjusted, warnings

    def _make_guidance_envelope(self, guidance: Dict, approved_params: Dict) -> Dict:
        """Use the validated proposal to initialize/restrict cached-score search."""
        out = copy.deepcopy(guidance)
        out["param_changes"] = dict(approved_params)
        interval = dict(out.get("threshold_interval", {}) or {})
        radius = float(self.cfg.get("threshold_search", {}).get("guidance_radius", 0.05))
        mapping = {
            "accept_quantile": "accept_quantile_range",
            "margin_quantile": "margin_quantile_range",
            "delta_fused": "delta_fused_range",
            "delta_margin": "delta_margin_range",
        }
        for key, value in approved_params.items():
            if key not in mapping:
                continue
            glo, ghi = DEFAULT_SAFE_RANGES[key]
            proposed_lo = max(glo, value - radius)
            proposed_hi = min(ghi, value + radius)
            existing = interval.get(mapping[key], [glo, ghi])
            lo = max(float(existing[0]), proposed_lo)
            hi = min(float(existing[1]), proposed_hi)
            if lo <= hi:
                interval[mapping[key]] = [lo, hi]
        out["threshold_interval"] = interval
        return out

    def apply_guidance(self, guidance: Dict) -> Tuple[Optional[Dict], bool, List[str]]:
        """Validate and convert a cloud proposal into a bounded search envelope.

        This routine never changes model weights or signal-augmentation
        parameters. It only prepares the cached-score threshold search.
        """
        raw_params = guidance.get("param_changes", {}) or {}
        approved, adjusted, warnings = self._policy_guard_review(
            raw_params,
            guidance.get("safe_param_ranges"),
            guidance.get("max_step_sizes"),
        )
        if not adjusted:
            return None, approved, warnings
        return self._make_guidance_envelope(guidance, adjusted), approved, warnings

    def _report_policy_audit(self, guidance: Dict, approved: bool, warnings: List[str], final_params: Dict) -> None:
        proposal_id = guidance.get("proposal_id")
        if not proposal_id or guidance.get("source") not in {"rag_llm", "llm_only"} or not self.cloud_enabled:
            return
        try:
            import requests
            requests.post(
                f"{self.cloud_server}/policy_guard/audit",
                json={
                    "proposal_id": proposal_id,
                    "policy_approved": bool(approved),
                    "policy_warnings": list(warnings),
                    "final_params": dict(final_params),
                    "executable": bool(final_params),
                },
                timeout=self.cloud_timeout,
            )
        except requests.exceptions.RequestException as exc:
            print(f"[EdgeAgentV2] policy audit report failed: {exc}")

    def _edge_local_fallback(self) -> Dict:
        """Conservative edge-local fallback for cloud interruption only."""
        current = self._normalize_search_state(self.state.best_config)
        neutral = self.cfg.get("fallback", {}).get("neutral_search_state", {
            "accept_quantile": 0.90,
            "margin_quantile": 0.20,
            "delta_fused": 0.10,
            "delta_margin": 0.10,
        })
        max_step = float(self.cfg.get("fallback", {}).get("max_step", 0.03))
        proposal = {}
        for key in DEFAULT_SAFE_RANGES:
            target = float(neutral.get(key, current[key]))
            delta = target - current[key]
            proposal[key] = current[key] + max(-max_step, min(max_step, delta))
        approved, adjusted, warnings = self._policy_guard_review(
            proposal,
            {k: list(v) for k, v in DEFAULT_SAFE_RANGES.items()},
            {k: max_step for k in DEFAULT_SAFE_RANGES},
        )
        return {
            "type": "lightweight",
            "source": "edge_local_fallback",
            "param_changes": adjusted,
            "threshold_interval": {f"{k}_range" if k not in {"delta_fused", "delta_margin"} else f"{k}_range": list(v)
                                   for k, v in DEFAULT_SAFE_RANGES.items()},
            "safe_param_ranges": {k: list(v) for k, v in DEFAULT_SAFE_RANGES.items()},
            "max_step_sizes": {k: max_step for k in DEFAULT_SAFE_RANGES},
            "policy_approved": approved,
            "policy_warnings": warnings,
            "requires_reprocessing": False,
        }

    def _safe_extract_tar(self, archive: Path, target: Path) -> None:
        target_resolved = target.resolve()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = (target / member.name).resolve()
                if target_resolved not in member_path.parents and member_path != target_resolved:
                    raise ValueError("Unsafe path in model package")
            tar.extractall(target)

    def download_model_package(self) -> bool:
        if not self.cloud_enabled:
            return False
        try:
            import requests
            response = requests.get(
                f"{self.cloud_server}/model/download",
                stream=True,
                timeout=max(60.0, self.cloud_timeout),
            )
            if response.status_code != 200:
                return False
            cache = self.output_dir / "cloud_model"
            cache.mkdir(parents=True, exist_ok=True)
            for child in cache.iterdir():
                if child.is_file():
                    child.unlink()
            archive = cache / "source_model_package.tar.gz"
            with open(archive, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            self._safe_extract_tar(archive, cache)
            return self._apply_model_package(cache)
        except Exception as exc:
            print(f"[EdgeAgentV2] model reprovision failed: {exc}")
            return False

    def _apply_model_package(self, cache: Path) -> bool:
        weights = cache / "model_weights.pth"
        if not weights.is_file():
            return False
        self.cfg["paths"]["model_weights"] = str(weights)
        self.paths = self.cfg["paths"]

        model_cfg_file = cache / "model_config.json"
        if model_cfg_file.is_file():
            self.cfg.setdefault("model", {}).update(json.loads(model_cfg_file.read_text(encoding="utf-8")))

        threshold_file = cache / "baseline_thresholds.json"
        if threshold_file.is_file():
            thresholds = json.loads(threshold_file.read_text(encoding="utf-8"))
            if isinstance(thresholds, list) and thresholds:
                first = thresholds[0]
                calib = self.cfg.setdefault("open_set", {}).setdefault("calibration", {})
                for key in ("accept_quantile", "margin_quantile", "delta_fused", "delta_margin"):
                    if key in first:
                        calib[key] = first[key]
                if "rho" in first and "delta_fused" not in first:
                    calib["delta_fused"] = first["rho"]

        self._features = None
        self._model = None
        print("[EdgeAgentV2] source-trained model package reprovisioned")
        return True

    def report_episode(self, before: Dict, after: Dict, result: AdaptationResult) -> bool:
        if not self.cloud_enabled:
            return False
        payload = {
            "terminal_id": self.terminal_id,
            "episode_id": f"{self.terminal_id}_{self.episode_count}",
            "timestamp": result.timestamp,
            "state_before": self._current_state_payload(before),
            "state_after": self._current_state_payload(after),
            "threshold_config": dict(result.threshold_config),
            # Historical-case success is an operational/control-buffer label;
            # the disjoint final evaluation buffer is never fed back to cloud.
            "recovery_success": bool(self._operational_feasible(after)),
            "event_descriptor": "domain_shift",
        }
        try:
            import requests
            response = requests.post(f"{self.cloud_server}/episode", json=payload, timeout=self.cloud_timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as exc:
            print(f"[EdgeAgentV2] episode report failed: {exc}")
            return False

    def _record_result(self, best: Dict, assistance_type: str, guidance_source: str,
                       policy_approved: Optional[bool] = None,
                       policy_warnings: Optional[List[str]] = None) -> AdaptationResult:
        self.episode_count += 1
        features = self._features or {}
        result = AdaptationResult(
            episode_id=self.episode_count,
            round_id=self.state.current_round,
            timestamp=datetime.now().isoformat(),
            threshold_config=dict(best["config"]),
            metrics=dict(best["metrics"]),
            control_metrics=dict(best.get("control_metrics", {})),
            score_stats={
                "fused_mean": float(features.get("fused_mean", 0.0)),
                "fused_std": float(features.get("fused_std", 1.0)),
                "margin_mean": float(features.get("margin_mean", 0.0)),
                "margin_std": float(features.get("margin_std", 1.0)),
            },
            success=self.check_objectives(best["metrics"]),
            assistance_type=assistance_type,
            guidance_source=guidance_source,
            policy_approved=policy_approved,
            policy_warnings=list(policy_warnings or []),
        )
        self.all_results.append(result)
        return result

    def _print_metrics(self, title: str, best: Dict) -> None:
        m = best["metrics"]
        cm = best.get("control_metrics", {})
        print(f"\n[{title}]")
        print(f"  evaluation Ac={m['closed_acc']:.4f} (target {self.objectives['min_closed_acc']:.2f})")
        print(f"  evaluation Ao={m['open_auc']:.4f} (target {self.objectives['target_open_auc']:.2f})")
        print(f"  evaluation R ={m['reject_rate']:.4f} (auxiliary)")
        if cm:
            print(f"  control    Ac/Ao/R={cm['closed_acc']:.4f}/{cm['open_auc']:.4f}/{cm['reject_rate']:.4f}")
        print(f"  operational feasible={self.check_objectives(cm) if cm else self.check_objectives(m)}")
        print(f"  evaluation feasible ={self.check_objectives(m)}")

    def run(self) -> Optional[Dict]:
        """Hierarchical run: local edge -> cloud guidance -> model refresh.

        No lightweight stage invokes signal reprocessing or model retraining.
        """
        self.state = OptimizationState()
        self.all_results = []
        self.current_guidance = None

        if not self.prepare_data():
            return None
        if not self.check_weights():
            print("[EdgeAgentV2] no source weights found; performing initial source-domain training")
            if not self.train_initial_model():
                return None

        # Stage 1: edge-only cached-score correction.
        edge_best, edge_pareto = self.threshold_search(guidance=None)
        self._update_state(edge_best, edge_pareto)
        self.last_edge_result = edge_best
        self._print_metrics("EDGE CONTROL", edge_best)
        if self._operational_feasible(edge_best):
            result = self._record_result(edge_best, "edge", "edge_only")
            self._save_results()
            if self.cloud_enabled:
                self.report_episode(edge_best, edge_best, result)
            return edge_best

        current_best = edge_best
        last_source = "edge_only"
        last_policy_approved: Optional[bool] = None
        last_policy_warnings: List[str] = []

        # Stage 2: event-driven cloud lightweight guidance. Repeated rounds only
        # change the search envelope over the same cached scores.
        if self.cloud_enabled:
            for _ in range(max(1, self.max_rounds)):
                try:
                    guidance = self.request_cloud_guidance(current_best, stage="post_edge")
                except CloudUnavailable as exc:
                    print(f"[EdgeAgentV2] cloud interruption: {exc}")
                    fallback = self._edge_local_fallback()
                    approved = bool(fallback.get("policy_approved", True))
                    warnings = list(fallback.get("policy_warnings", []))
                    envelope = self._make_guidance_envelope(fallback, fallback.get("param_changes", {}))
                    fallback_best, fallback_pareto = self.threshold_search(envelope)
                    self._update_state(fallback_best, fallback_pareto)
                    self._print_metrics("EDGE-LOCAL FALLBACK", fallback_best)
                    current_best = fallback_best
                    last_source = "edge_local_fallback"
                    last_policy_approved = approved
                    last_policy_warnings = warnings
                    break

                if guidance.get("type") == "model_reprovision":
                    break

                raw_params = guidance.get("param_changes", {}) or {}
                if not raw_params:
                    print(f"[EdgeAgentV2] no executable lightweight guidance: {guidance.get('source')}")
                    last_source = str(guidance.get("source", "no_executable_guidance"))
                    break

                envelope, approved, warnings = self.apply_guidance(guidance)
                adjusted = envelope.get("param_changes", {}) if envelope else {}
                self._report_policy_audit(guidance, approved, warnings, adjusted)
                if envelope is None:
                    # In the production RAG path, an unusable LLM-derived
                    # proposal falls back to non-generative retrieval guidance
                    # while the cloud is still reachable. This fallback is not
                    # used in the LLM-only ablation arm.
                    if self.guidance_strategy == "rag":
                        deterministic = self.request_cloud_guidance(
                            current_best, stage="post_edge", strategy="deterministic"
                        )
                        det_envelope, det_approved, det_warnings = self.apply_guidance(deterministic)
                        if det_envelope is not None:
                            envelope = det_envelope
                            approved = det_approved
                            warnings = det_warnings
                            guidance = deterministic
                            adjusted = envelope.get("param_changes", {})
                        else:
                            last_source = "no_executable_guidance"
                            break
                    else:
                        last_source = "no_executable_guidance"
                        break

                guided_best, guided_pareto = self.threshold_search(envelope)
                self._update_state(guided_best, guided_pareto)
                self._print_metrics(str(guidance.get("source", "LIGHTWEIGHT GUIDANCE")).upper(), guided_best)
                current_best = guided_best
                last_source = str(guidance.get("source", "lightweight"))
                last_policy_approved = approved and not bool(guidance.get("first_pass_adjusted", False))
                last_policy_warnings = list(guidance.get("validation_notes", [])) + warnings
                if self._operational_feasible(guided_best):
                    break

        # Stage 3: source-model reprovisioning is the final mechanism after
        # residual infeasibility. It is separate from lightweight guidance.
        if self.cloud_enabled and not self._operational_feasible(current_best):
            try:
                refresh = self.request_cloud_guidance(current_best, stage="post_lightweight")
            except CloudUnavailable:
                refresh = {}
            if refresh.get("type") == "model_reprovision" and refresh.get("available", False):
                if self.download_model_package():
                    refreshed_best, refreshed_pareto = self.threshold_search(guidance=None)
                    self._update_state(refreshed_best, refreshed_pareto)
                    self._print_metrics("MODEL REFRESH", refreshed_best)
                    current_best = refreshed_best
                    last_source = "source_model_reprovision"
                    last_policy_approved = None
                    last_policy_warnings = []

        if last_source == "edge_only":
            assistance_type = "edge"
        elif last_source == "edge_local_fallback":
            assistance_type = "fallback"
        else:
            assistance_type = "cloud"
        result = self._record_result(
            current_best,
            assistance_type,
            last_source,
            last_policy_approved,
            last_policy_warnings,
        )
        self._save_results()
        if self.cloud_enabled:
            self.report_episode(edge_best, current_best, result)
        return current_best

    def _save_results(self) -> None:
        payload = {
            "terminal_id": self.terminal_id,
            "timestamp": datetime.now().isoformat(),
            "objectives": self.objectives,
            "feasibility_definition": "closed_acc >= min_closed_acc AND open_auc >= target_open_auc",
            "rejection_rate_role": "auxiliary service-availability indicator",
            "best_metrics": self.state.best_metrics,
            "best_config": self.state.best_config,
            "operational_objectives_met": self.state.objectives_met,
            "evaluation_objectives_met": self.check_objectives(self.state.best_metrics) if self.state.best_metrics else False,
            "history": self.state.history,
            "results": [asdict(r) for r in self.all_results],
        }
        path = self.output_dir / "edge_results.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[EdgeAgentV2] results saved to {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Edge control agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cloud", action="store_true", help="Enable cloud coordination")
    parser.add_argument("--cloud-server", default=None)
    parser.add_argument("--guidance-strategy", choices=["rag", "llm_only", "deterministic"], default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("cloud", {})
    if args.cloud:
        cfg["cloud"]["enabled"] = True
    if args.cloud_server:
        cfg["cloud"]["server_url"] = args.cloud_server
    if args.guidance_strategy:
        cfg["cloud"]["guidance_strategy"] = args.guidance_strategy

    agent = EdgeAgentV2(cfg)
    agent.run()


if __name__ == "__main__":
    main()

# edge_agent_v2.py
# -*- coding: utf-8 -*-


import os
import sys
import json
import copy
import time
import socket
import uuid
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader




class RunMode(Enum):
    TRAIN_AND_TEST = "train_and_test"
    TEST_ONLY = "test_only"


class AssistanceType(Enum):
    NONE = "none"
    LIGHTWEIGHT = "lightweight"           # cloud-side bounded threshold guidance
    MODEL_REPROVISION = "model_reprovision"  # 模型重配置


@dataclass
class AdaptationResult:
    """适应轮次结果"""
    episode_id: int
    round_id: int
    timestamp: str
    threshold_config: Dict
    metrics: Dict[str, float]
    score_stats: Dict
    success: bool
    assistance_type: str
    param_changes: Dict = field(default_factory=dict)


@dataclass
class OptimizationState:
    """优化状态追踪"""
    current_round: int = 0
    best_metrics: Dict = field(default_factory=dict)
    best_config: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    objectives_met: bool = False
    at_pareto_limit: bool = False
    improvement_trend: str = "unknown"




class EdgeAgentV2:

    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.original_cfg = copy.deepcopy(cfg)
        self.terminal_id = self._gen_id()

        self.objectives = cfg["agent"]["objectives"]
        self.max_rounds = int(cfg["agent"].get("max_rounds", 5))
        self.paths = cfg["paths"]
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cloud_cfg = cfg.get("cloud", {})
        self.cloud_enabled = bool(cloud_cfg.get("enabled", False))
        self.cloud_server = cloud_cfg.get("server_url", "http://localhost:5000")

        # LLM inference is intentionally not initialized at the edge.  The edge
        # only receives bounded cloud guidance and performs final deterministic
        # validation / cached-score replay.
        self.episode_count = 0
        self.state = OptimizationState()
        self.all_results: List[AdaptationResult] = []
        self.current_assistance: Optional[Dict] = None
        self.assistance_type = AssistanceType.NONE

        self._features = None
        self._model = None

        print(f"[EdgeAgentV2] Terminal: {self.terminal_id}")
        print(f"[EdgeAgentV2] Cloud: {self.cloud_enabled}")
        print("[EdgeAgentV2] LLM inference: cloud-side only")
        print(f"[EdgeAgentV2] Max rounds: {self.max_rounds}")
    
    def _gen_id(self) -> str:
        return f"{socket.gethostname()}_{uuid.getnode():012x}"[:32]
    

    
    def check_weights(self) -> bool:
        """检查权重文件"""
        path = self.paths.get("model_weights", "")
        if not path or not os.path.exists(path):
            print(f"[EdgeAgentV2] ❌ No weights: {path}")
            return False
        try:
            sd = torch.load(path, map_location="cpu")
            if len(sd) > 0:
                print(f"[EdgeAgentV2] ✅ Weights found: {path}")
                return True
        except Exception as e:
            print(f"[EdgeAgentV2] ⚠️ Invalid: {e}")
        return False
    
    def determine_mode(self) -> RunMode:
        """决定运行模式"""
        return RunMode.TEST_ONLY if self.check_weights() else RunMode.TRAIN_AND_TEST
    

    
    def request_assistance(self) -> Optional[Dict]:
        """Query only the cloud orchestration mode.

        Lightweight guidance itself is requested later, after local edge
        correction has been evaluated and found infeasible.
        """
        if not self.cloud_enabled:
            return None
        try:
            import requests
            resp = requests.get(f"{self.cloud_server}/assistance/{self.terminal_id}", timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            self.current_assistance = None
            t = data.get("type", "lightweight")
            if t == "model_reprovision":
                self.assistance_type = AssistanceType.MODEL_REPROVISION
                print("[EdgeAgentV2] MODEL REPROVISIONING mode")
            else:
                self.assistance_type = AssistanceType.LIGHTWEIGHT
                print("[EdgeAgentV2] LIGHTWEIGHT mode (cloud-side advisory after local failure)")
            return data
        except Exception as exc:
            print(f"[EdgeAgentV2] Cloud orchestration request failed: {exc}")
            self.assistance_type = AssistanceType.LIGHTWEIGHT
            return None
    
    def download_model_package(self) -> bool:
        """下载云端完整模型包"""
        if not self.cloud_enabled:
            return False
        try:
            import requests
            resp = requests.get(f"{self.cloud_server}/model/download", stream=True, timeout=60)
            if resp.status_code == 200:
                cache = self.output_dir / "cloud_model"
                cache.mkdir(exist_ok=True)
                
                # 清理旧文件
                for f in cache.iterdir():
                    if f.is_file():
                        f.unlink()
                
                pkg_path = cache / "model_package.tar.gz"
                with open(pkg_path, 'wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                
                # 解压
                with tarfile.open(pkg_path, 'r:gz') as tar:
                    tar.extractall(cache)
                
                # 应用模型包
                return self._apply_model_package(cache)
        except Exception as e:
            print(f"[EdgeAgentV2] Download failed: {e}")
        return False
    
    def _apply_model_package(self, cache_dir: Path) -> bool:
        """应用云端模型包"""
        try:
            # 权重
            new_weights = cache_dir / "model_weights.pth"
            if new_weights.exists():
                self.cfg["paths"]["model_weights"] = str(new_weights)
                print(f"[EdgeAgentV2] ✅ Weights applied: {new_weights}")
            
            # 模型配置
            model_config_file = cache_dir / "model_config.json"
            if model_config_file.exists():
                with open(model_config_file) as f:
                    model_config = json.load(f)
                self.cfg["model"].update(model_config)
                print(f"[EdgeAgentV2] ✅ Model config applied")
            
            # 信号增强参数
            signal_augment_file = cache_dir / "signal_augment.json"
            if signal_augment_file.exists():
                with open(signal_augment_file) as f:
                    signal_augment = json.load(f)
                if signal_augment:
                    self.cfg["signal_augment"].update(signal_augment)
                    print(f"[EdgeAgentV2] ✅ Signal augment applied")
            
            # 基线阈值
            thresholds_file = cache_dir / "baseline_thresholds.json"
            if thresholds_file.exists():
                with open(thresholds_file) as f:
                    thresholds = json.load(f)
                if thresholds:
                    # 使用第一组基线阈值更新 open_set 配置
                    if isinstance(thresholds, list) and len(thresholds) > 0:
                        first = thresholds[0]
                        self.cfg["open_set"]["calibration"]["accept_quantile"] = first.get("accept_quantile", 0.9)
                        self.cfg["open_set"]["calibration"]["margin_quantile"] = first.get("margin_quantile", 0.2)
                        self.cfg["open_set"]["calibration"]["rho"] = first.get("rho", 0.1)
                    print(f"[EdgeAgentV2] ✅ Baseline thresholds applied")
            
            # 元数据
            metadata_file = cache_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"[EdgeAgentV2] Model package version: {metadata.get('version', 'unknown')}")
            
            # 清除缓存
            self._features = None
            self._model = None
            
            return True
            
        except Exception as e:
            print(f"[EdgeAgentV2] Apply model package failed: {e}")
            return False
    
    def report_episode(self, result: AdaptationResult) -> bool:
        """Report compact operational evidence; never upload target-trained weights."""
        if not self.cloud_enabled:
            return False
        try:
            import requests
            data = {
                "terminal_id": self.terminal_id,
                "timestamp": result.timestamp,
                "episode_id": result.episode_id,
                "delta_fused": result.threshold_config.get("rho", 0.0),
                "accept_quantile": result.threshold_config.get("accept_quantile", 0.90),
                "margin_quantile": result.threshold_config.get("margin_quantile", 0.20),
                "threshold_config": result.threshold_config,
                "closed_acc": result.metrics.get("closed_acc", 0.0),
                "open_auc": result.metrics.get("open_auc", 0.0),
                "reject_rate": result.metrics.get("reject_rate", 0.0),
                "fused_mean": result.score_stats.get("fused_mean", 0.0),
                "fused_std": result.score_stats.get("fused_std", 1.0),
                "recovery_success": result.success,
                "model_config": self.cfg.get("model", {}),
            }
            resp = requests.post(f"{self.cloud_server}/episode", json=data, timeout=60)
            if resp.status_code == 200:
                print("[EdgeAgentV2] Episode reported")
                return True
            print(f"[EdgeAgentV2] Episode report rejected: HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[EdgeAgentV2] Report failed: {exc}")
        return False
    

    
    def request_cloud_guidance(self, metrics: Dict, best_config: Dict,
                               pareto_front: List[Dict]) -> Optional[Dict]:
        """Request cloud-side retrieval/LLM advisory after local infeasibility."""
        if not self.cloud_enabled:
            return None
        try:
            import requests
            current_state = {
                "closed_acc": float(metrics.get("closed_acc", 0.0)),
                "open_auc": float(metrics.get("open_auc", 0.0)),
                "reject_rate": float(metrics.get("reject_rate", 0.0)),
                "threshold_config": dict(best_config or {}),
                "accept_quantile": float(best_config.get("accept_quantile", 0.90)),
                "margin_quantile": float(best_config.get("margin_quantile", 0.20)),
                "rho": float(best_config.get("rho", 0.0)),
                "fused_mean": float((self._features or {}).get("fused_mean", 0.0)),
                "fused_std": float((self._features or {}).get("fused_std", 1.0)),
                "pareto_size": len(pareto_front or []),
            }
            payload = {"terminal_id": self.terminal_id, "current_state": current_state}
            resp = requests.post(f"{self.cloud_server}/guidance", json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"[EdgeAgentV2] Cloud guidance rejected: HTTP {resp.status_code}")
                return None
            guidance = resp.json()
            if not guidance.get("available", False):
                # A valid cloud response declaring the requested guidance arm
                # unavailable is not a transport interruption. Return it so the
                # caller does not silently substitute another experimental arm.
                print(f"[EdgeAgentV2] Cloud guidance unavailable: {guidance.get('guidance_source', 'unknown')}")
                return guidance

            # Final edge-side deterministic validation.  The LLM proposal itself is
            # never applied; only the validated bounded search interval is used.
            patch = guidance.get("guidance_patch", {})
            approved, adjusted, warnings = self._policy_guard_review(patch, best_config)
            if warnings:
                print(f"[EdgeAgentV2] Final policy warnings: {warnings}")
            guidance["guidance_patch"] = adjusted
            guidance["edge_policy_approved"] = approved
            guidance["threshold_interval"] = self._validate_threshold_interval(
                guidance.get("threshold_interval", {}), adjusted
            )
            guidance["requires_reprocessing"] = False

            self.current_assistance = guidance
            self.assistance_type = AssistanceType.LIGHTWEIGHT
            print(f"[EdgeAgentV2] Cloud guidance source: {guidance.get('guidance_source')}")
            print(f"[EdgeAgentV2] Bounded threshold patch: {adjusted}")
            return guidance
        except Exception as exc:
            print(f"[EdgeAgentV2] Cloud guidance request failed: {exc}")
            return None
    
    def _validate_threshold_interval(self, interval: Dict,
                                     patch: Dict = None) -> Dict:
        """Clamp cloud-provided search intervals to edge hard bounds."""
        hard = {
            "accept_quantile_range": (0.70, 0.98),
            "margin_quantile_range": (0.05, 0.35),
            "delta_fused_range": (-0.15, 0.35),
        }
        out = {}
        interval = interval or {}
        for key, (hard_lo, hard_hi) in hard.items():
            raw = interval.get(key, [hard_lo, hard_hi])
            try:
                lo, hi = float(raw[0]), float(raw[1])
            except (TypeError, ValueError, IndexError):
                lo, hi = hard_lo, hard_hi
            lo, hi = max(hard_lo, lo), min(hard_hi, hi)
            if lo > hi:
                lo, hi = hard_lo, hard_hi
            out[key] = [lo, hi]
        # Preserve documented compatibility field if supplied; it is not an
        # independent dimension of this implementation's search grid.
        if "delta_margin_range" in interval:
            out["delta_margin_range"] = interval["delta_margin_range"]
        return out
    
    def _policy_guard_review(self, threshold_patch: Dict,
                             current_config: Dict = None,
                             safe_ranges: Dict = None,
                             max_steps: Dict = None) -> Tuple[bool, Dict, List[str]]:
        """Final edge-side deterministic review of cloud threshold guidance."""
        safe_ranges = safe_ranges or {
            "accept_quantile": [0.80, 0.95],
            "margin_quantile": [0.10, 0.30],
            "rho": [-0.15, 0.35],
        }
        max_steps = max_steps or {
            "accept_quantile": 0.05,
            "margin_quantile": 0.05,
            "rho": 0.10,
        }
        current_config = current_config or {
            "accept_quantile": 0.90, "margin_quantile": 0.20, "rho": 0.0
        }

        warnings: List[str] = []
        adjusted: Dict[str, float] = {}
        approved = True
        for key, raw in (threshold_patch or {}).items():
            if key not in safe_ranges:
                warnings.append(f"unsupported threshold field ignored: {key}")
                approved = False
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                warnings.append(f"non-numeric threshold field ignored: {key}")
                approved = False
                continue

            current = float(current_config.get(key, 0.0))
            max_step = float(max_steps.get(key, 1e9))
            delta = value - current
            if abs(delta) > max_step:
                value = current + (max_step if delta > 0 else -max_step)
                warnings.append(f"{key} step limited to {max_step:.3f}")
                approved = False

            lo, hi = safe_ranges[key]
            clipped = max(float(lo), min(float(hi), value))
            if clipped != value:
                warnings.append(f"{key} projected to [{lo}, {hi}]")
                approved = False
            adjusted[key] = float(clipped)

        if not adjusted:
            approved = False
        return approved, adjusted, warnings
    
    def _rule_based_fallback(self, metrics: Dict, diagnosis: Dict,
                             cloud_experience: Dict = None,
                             current_config: Dict = None) -> Optional[Dict]:
        """Conservative edge-local interruption fallback in threshold space.

        This fallback does not retrain the model and does not invoke an LLM.  It
        only narrows the next cached-score replay around the previous operating
        point.  It is an interruption-handling mechanism, not a performance claim.
        """
        current_config = current_config or self.state.best_config or {
            "accept_quantile": 0.90, "margin_quantile": 0.20, "rho": 0.0
        }
        aq = float(current_config.get("accept_quantile", 0.90))
        mq = float(current_config.get("margin_quantile", 0.20))
        rho = float(current_config.get("rho", 0.0))
        gaps = diagnosis.get("gaps", {}) if diagnosis else {}
        closed_gap = float(gaps.get("closed_acc_gap", 0.0))
        open_gap = float(gaps.get("open_auc_gap", 0.0))

        # Only small bounded movements are permitted.  R is deliberately not used
        # as a trigger because feasibility is defined solely by A_c and A_o.
        if closed_gap <= 0 and open_gap <= 0:
            return None
        if closed_gap >= open_gap:
            aq = min(0.95, aq + 0.05)
            mq = max(0.10, mq - 0.05)
        else:
            aq = max(0.80, aq - 0.05)
            mq = min(0.30, mq + 0.05)
            rho = max(-0.15, rho - 0.05)

        interval = {
            "accept_quantile_range": [max(0.80, aq - 0.05), min(0.95, aq + 0.05)],
            "margin_quantile_range": [max(0.10, mq - 0.05), min(0.30, mq + 0.05)],
            "delta_fused_range": [max(-0.15, rho - 0.10), min(0.35, rho + 0.10)],
        }
        return {
            "type": "fallback",
            "available": True,
            "threshold_interval": interval,
            "guidance_patch": {"accept_quantile": aq, "margin_quantile": mq, "rho": rho},
            "requires_reprocessing": False,
            "source": "edge_fallback",
            "analysis": "Bounded local threshold-space fallback during cloud interruption",
        }
    
    def apply_param_changes(self, param_changes: Dict) -> bool:
        """Legacy optional reprocessing helper.

        This method is not used by the manuscript-aligned lightweight guidance
        path. Lightweight cloud guidance is threshold-space only and is consumed
        through ``current_assistance`` / ``get_search_grid()``. The helper is
        retained solely for backward compatibility with older reprocessing
        scripts.
        """
        if not param_changes:
            return False
        
        print(f"[EdgeAgentV2] Applying param changes: {param_changes}")
        
        # 更新 signal_augment 参数
        if "signal_augment" not in self.cfg:
            self.cfg["signal_augment"] = {}
        
        for key, value in param_changes.items():
            self.cfg["signal_augment"][key] = value
            print(f"  {key}: {value}")
        
        return True
    

    def prepare_data(self, force_reprocess: bool = False) -> bool:
        """准备数据"""
        from data_pipeline import DataPipeline
        pipeline = DataPipeline(self.cfg)
        return pipeline.prepare_dataset(
            raw_dir=self.paths.get("raw_data", ""),
            processed_dir=self.paths.get("processed_data", ""),
            target_dir=self.paths["base"],
            label_map_file=self.paths["label_map"],
            force_reprocess=force_reprocess,
            num_workers=self.cfg.get("signal_processing", {}).get("num_workers", 4),
        )
    
    def train(self) -> bool:
        """训练模型"""
        print("\n" + "=" * 60)
        print(f"TRAINING (Round {self.state.current_round})")
        print("=" * 60)
        
        # 清除模型缓存
        self._model = None
        self._features = None
        
        try:
            import model as model_module
            model_module.train(self.cfg)
            return True
        except Exception as e:
            print(f"[EdgeAgentV2] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    
    def extract_features(self) -> Dict:
        """提取特征"""
        if self._features is not None:
            return self._features
        
        from model import (
            ThreeBandDataset, ConvNeXtCosFace,
            extract_logits_feats, energy_score, get_margin, set_seed
        )
        from sklearn.cluster import KMeans
        from sklearn.covariance import LedoitWolf
        
        seed = self.cfg.get("seed", 42)
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        paths = self.paths
        base = Path(paths["base"])
        data_cfg = self.cfg["data"]
        
        with open(paths["label_map"]) as f:
            label_map = json.load(f)
        n_classes = len(label_map)
        
        model_cfg = self.cfg.get("model", {})
        backbone = model_cfg.get("backbone", "convnext_tiny")
        
        # 数据集
        val_ds = ThreeBandDataset(
            base / data_cfg["low"]["val"],
            base / data_cfg["mid"]["val"],
            base / data_cfg["high"]["val"],
            label_map, train=False, cfg=self.cfg
        )
        test_ds = ThreeBandDataset(
            base / data_cfg["low"]["test"],
            base / data_cfg["mid"]["test"],
            base / data_cfg["high"]["test"],
            label_map, train=False, cfg=self.cfg
        )
        
        train_cfg = self.cfg.get("train", {})
        batch_size = train_cfg.get("batch_size", 128)
        num_workers = train_cfg.get("num_workers", 4)
        
        loader_kwargs = {"batch_size": batch_size, "shuffle": False, 
                        "num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        
        val_loader = DataLoader(val_ds, **loader_kwargs)
        test_loader = DataLoader(test_ds, **loader_kwargs)
        
        # 模型
        model = ConvNeXtCosFace(
            n_classes=n_classes, backbone=backbone,
            pretrained_try=False, channels_last=True
        ).to(device)
        model.load_state_dict(torch.load(paths["model_weights"], map_location=device), strict=True)
        model.eval()
        self._model = model
        
        print("[EdgeAgentV2] Extracting features...")
        log_val, feat_val, y_val = extract_logits_feats(model, val_loader, device)
        log_te, feat_te, y_te = extract_logits_feats(model, test_loader, device)
        
        # 分数
        en_val, en_te = energy_score(log_val), energy_score(log_te)
        margin_val, margin_te = get_margin(log_val), get_margin(log_te)
        
        # 马氏距离
        k_centroids = self.cfg.get("open_set", {}).get("mahalanobis", {}).get("k_centroids", 3)
        class_stats = {}
        for c in sorted(set(y_val)):
            Xc = feat_val[y_val == c]
            k = min(k_centroids, max(1, len(Xc)))
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(Xc)
            lwc = LedoitWolf().fit(Xc)
            class_stats[c] = (km.cluster_centers_, np.linalg.pinv(lwc.covariance_))
        
        def mahal(X):
            out = np.empty(len(X), dtype=np.float32)
            for i, x in enumerate(X):
                best = 1e9
                for mus, Sinv in class_stats.values():
                    for mu in mus:
                        d = float(np.sqrt((x - mu) @ Sinv @ (x - mu).T))
                        if d < best:
                            best = d
                out[i] = best
            return out
        
        md_val, md_te = mahal(feat_val), mahal(feat_te)
        
        # 标准化
        ez_val = (en_val - en_val.mean()) / (en_val.std() + 1e-6)
        dz_val = (md_val - md_val.mean()) / (md_val.std() + 1e-6)
        ez_te = (en_te - en_val.mean()) / (en_val.std() + 1e-6)
        dz_te = (md_te - md_val.mean()) / (md_val.std() + 1e-6)
        mz_val = (margin_val - margin_val.mean()) / (margin_val.std() + 1e-6)
        mz_te = (margin_te - margin_val.mean()) / (margin_val.std() + 1e-6)
        
        # Alpha
        alpha_grid = self.cfg.get("open_set", {}).get("energy", {}).get("alpha_grid", [i/10 for i in range(11)])
        best_alpha, best_obj = 0.5, 1e9
        for a in alpha_grid:
            fused = a * ez_val + (1 - a) * dz_val
            pred = log_val.argmax(1)
            tau = {c: np.quantile(fused[pred == c], 0.9) for c in set(pred) if (pred == c).sum() > 0}
            rej = fused > np.array([tau.get(int(p), 1e9) for p in pred])
            obj = rej.mean() + (pred != y_val).mean()
            if obj < best_obj:
                best_obj, best_alpha = obj, a
        
        fused_val = best_alpha * ez_val + (1 - best_alpha) * dz_val
        
        self._features = {
            "log_val": log_val, "log_te": log_te,
            "y_val": y_val, "y_te": y_te,
            "ez_val": ez_val, "ez_te": ez_te,
            "dz_val": dz_val, "dz_te": dz_te,
            "mz_val": mz_val, "mz_te": mz_te,
            "alpha": best_alpha,
            "fused_mean": float(fused_val.mean()),
            "fused_std": float(fused_val.std()),
            "margin_mean": float(mz_val.mean()),
            "margin_std": float(mz_val.std()),
        }
        
        print(f"[EdgeAgentV2] Features ready, alpha={best_alpha}")
        return self._features
    

    
    def get_search_grid(self) -> Dict[str, List[float]]:
        """Return the base grid, optionally restricted by validated cloud guidance."""
        search_cfg = self.cfg.get("threshold_search", {})
        grid = {
            "accept_q": list(search_cfg.get(
                "accept_q_grid", [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
            )),
            "margin_q": list(search_cfg.get(
                "margin_q_grid", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
            )),
            "rho": list(search_cfg.get(
                "rho_grid", [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
            )),
        }

        if self.assistance_type == AssistanceType.LIGHTWEIGHT and self.current_assistance:
            interval = self._validate_threshold_interval(
                self.current_assistance.get("threshold_interval", {})
            )
            ar = interval["accept_quantile_range"]
            mr = interval["margin_quantile_range"]
            rr = interval["delta_fused_range"]
            constrained = {
                "accept_q": [q for q in grid["accept_q"] if ar[0] <= q <= ar[1]],
                "margin_q": [q for q in grid["margin_q"] if mr[0] <= q <= mr[1]],
                "rho": [r for r in grid["rho"] if rr[0] <= r <= rr[1]],
            }
            # Never allow a malformed/overly narrow advisory interval to create an
            # empty search dimension. Fall back to the closest base-grid value.
            key_map = {
                "accept_q": ("accept_quantile", ar),
                "margin_q": ("margin_quantile", mr),
                "rho": ("rho", rr),
            }
            patch = self.current_assistance.get("guidance_patch", {})
            for key, (patch_key, bounds) in key_map.items():
                if constrained[key]:
                    continue
                target = float(patch.get(patch_key, (bounds[0] + bounds[1]) / 2.0))
                constrained[key] = [min(grid[key], key=lambda value: abs(float(value) - target))]
            grid = constrained
            print(
                f"[EdgeAgentV2] Constrained grid: accept={len(grid['accept_q'])}, "
                f"margin={len(grid['margin_q'])}, rho={len(grid['rho'])}"
            )
        return grid
    
    def threshold_search(self) -> Tuple[Dict, List[Dict]]:
        """阈值网格搜索"""
        from sklearn.metrics import roc_curve, auc, accuracy_score
        
        print("\n" + "-" * 40)
        print("THRESHOLD SEARCH")
        print("-" * 40)
        
        features = self.extract_features()
        grid = self.get_search_grid()
        
        total = len(grid["accept_q"]) * len(grid["margin_q"]) * len(grid["rho"])
        print(f"Searching {total} combinations...")
        
        results = []
        for aq in grid["accept_q"]:
            for mq in grid["margin_q"]:
                for rho in grid["rho"]:
                    metrics = self._eval_threshold(features, aq, mq, rho)
                    score = self._score(metrics)
                    results.append({
                        "config": {"accept_quantile": aq, "margin_quantile": mq, "rho": rho},
                        "metrics": metrics,
                        "score": score,
                    })
        
        pareto = self._pareto(results)
        best = self._select_best(pareto)
        
        print(f"Best: closed={best['metrics']['closed_acc']:.4f}, open={best['metrics']['open_auc']:.4f}, reject={best['metrics']['reject_rate']:.4f}")
        return best, pareto
    
    def _eval_threshold(self, f: Dict, aq: float, mq: float, rho: float) -> Dict:
        from sklearn.metrics import roc_curve, auc, accuracy_score
        
        alpha = f["alpha"]
        fused_val = alpha * f["ez_val"] + (1 - alpha) * f["dz_val"]
        fused_te = alpha * f["ez_te"] + (1 - alpha) * f["dz_te"]
        
        pred_val = f["log_val"].argmax(1)
        pred_te = f["log_te"].argmax(1)
        
        tau_f = {c: np.quantile(fused_val[pred_val == c], aq) for c in set(pred_val) if (pred_val == c).sum() > 0}
        tau_m = {c: np.quantile(f["mz_val"][pred_val == c], mq) for c in set(pred_val) if (pred_val == c).sum() > 0}
        
        thr_f = np.array([tau_f.get(int(c), 1e9) for c in pred_te]) + rho
        thr_m = np.array([tau_m.get(int(c), -1e9) for c in pred_te])
        
        d1, d2 = fused_te - thr_f, thr_m - f["mz_te"]
        accept = (d1 <= 0) & (d2 <= 0)
        pred_open = np.where(accept, pred_te, -1)
        
        y_te = f["y_te"]
        reject_rate = (pred_open == -1).mean()
        valid = [(i, p, y) for i, (p, y) in enumerate(zip(pred_open, y_te)) if p != -1 and y != -1]
        closed_acc = accuracy_score([v[2] for v in valid], [v[1] for v in valid]) if valid else 0
        
        y_known = (y_te != -1).astype(int)
        gate = np.maximum(d1, d2)
        try:
            fpr, tpr, _ = roc_curve(y_known, gate, pos_label=0)
            open_auc = auc(fpr, tpr)
        except:
            open_auc = 0.5
        
        return {"closed_acc": float(closed_acc), "open_auc": float(open_auc), "reject_rate": float(reject_rate)}
    
    def _score(self, m: Dict) -> float:
        """Rank operating points while keeping feasibility strictly A_c/A_o based."""
        obj = self.objectives
        cg = max(0.0, float(obj["min_closed_acc"]) - float(m["closed_acc"]))
        og = max(0.0, float(obj["target_open_auc"]) - float(m["open_auc"]))

        # R is auxiliary: it may rank otherwise comparable candidates, but never
        # determines whether a candidate is feasible.
        availability_term = (1.0 - float(m.get("reject_rate", 0.0))) * 0.25
        if cg == 0 and og == 0:
            return float(m["closed_acc"]) * 2.0 + float(m["open_auc"]) * 1.5 + availability_term
        return float(m["closed_acc"]) + float(m["open_auc"]) + availability_term - (cg * 5.0 + og * 3.0)
    
    def _pareto(self, results: List[Dict]) -> List[Dict]:
        pareto = []
        for r in results:
            dom = False
            for o in results:
                if o is r:
                    continue
                if (o["metrics"]["closed_acc"] >= r["metrics"]["closed_acc"] and
                    o["metrics"]["open_auc"] >= r["metrics"]["open_auc"] and
                    o["metrics"]["reject_rate"] <= r["metrics"]["reject_rate"] and
                    (o["metrics"]["closed_acc"] > r["metrics"]["closed_acc"] or
                     o["metrics"]["open_auc"] > r["metrics"]["open_auc"] or
                     o["metrics"]["reject_rate"] < r["metrics"]["reject_rate"])):
                    dom = True
                    break
            if not dom:
                pareto.append(r)
        pareto.sort(key=lambda x: x["score"], reverse=True)
        return pareto
    
    def _select_best(self, pareto: List[Dict]) -> Dict:
        if not pareto:
            raise ValueError("Empty Pareto front")
        obj = self.objectives
        feasible = [r for r in pareto if
                    r["metrics"]["closed_acc"] >= obj["min_closed_acc"] and
                    r["metrics"]["open_auc"] >= obj["target_open_auc"]]
        if feasible:
            return max(feasible, key=lambda x: x["score"])
        closed_ok = [r for r in pareto if r["metrics"]["closed_acc"] >= obj["min_closed_acc"]]
        if closed_ok:
            return max(closed_ok, key=lambda x: (x["metrics"]["open_auc"], x["score"]))
        return max(pareto, key=lambda x: x["score"])
    
    def check_objectives(self, metrics: Dict) -> bool:
        """Feasibility is determined only by A_c and A_o."""
        return (
            metrics["closed_acc"] >= self.objectives["min_closed_acc"] and
            metrics["open_auc"] >= self.objectives["target_open_auc"]
        )
    
    def update_state(self, metrics: Dict, config: Dict, pareto: List[Dict]):
        """更新优化状态"""
        # 记录历史
        self.state.history.append({
            "round": self.state.current_round,
            "metrics": metrics,
            "config": config,
        })
        
        # 更新最佳
        if not self.state.best_metrics or metrics["open_auc"] > self.state.best_metrics.get("open_auc", 0):
            self.state.best_metrics = metrics
            self.state.best_config = config
        
        # 分析趋势
        if len(self.state.history) >= 2:
            prev = self.state.history[-2]["metrics"]
            if metrics["open_auc"] > prev["open_auc"] + 0.01:
                self.state.improvement_trend = "improving"
            elif metrics["open_auc"] < prev["open_auc"] - 0.01:
                self.state.improvement_trend = "declining"
            else:
                self.state.improvement_trend = "stable"
        
        # 检测 Pareto 边界
        if len(self.state.history) >= 3:
            recent = [h["metrics"]["open_auc"] for h in self.state.history[-3:]]
            if max(recent) - min(recent) < 0.005:
                self.state.at_pareto_limit = True
    

    
    def run(self) -> Optional[Dict]:
        """Run edge-first hierarchical recovery.

        Routine/lightweight recovery uses cached-score threshold replay only. LLM
        inference is cloud-side and model reprovisioning is a separate source-only
        recovery mode.
        """
        print("\n" + "=" * 70)
        print("EDGE AGENT V2 - ADAPTIVE AUTHENTICATION SYSTEM")
        print("=" * 70)
        print(f"Terminal: {self.terminal_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        self.assistance_type = AssistanceType.LIGHTWEIGHT
        if self.cloud_enabled:
            print("\n[Step 1] Querying cloud orchestration mode...")
            self.request_assistance()
        else:
            print("\n[Step 1] Offline edge-only mode")

        if self.assistance_type == AssistanceType.MODEL_REPROVISION:
            return self._run_model_reprovision_mode()
        return self._run_lightweight_mode()
    
    def _run_model_reprovision_mode(self) -> Optional[Dict]:
        """Re-provision the original source-trained model; never train on target data."""
        print("\n" + "=" * 60)
        print("MODE: SOURCE MODEL REPROVISIONING")
        print("=" * 60)
        self.state = OptimizationState(current_round=1)

        if self.cloud_enabled:
            if not self.download_model_package():
                print("[EdgeAgentV2] Cloud source package unavailable; checking configured local source weights")

        if not self.prepare_data():
            print("Data preparation failed")
            return None
        if not self.check_weights():
            print("No validated source-trained weights are available. Target-domain retraining is intentionally disabled in model-reprovision mode.")
            return None

        self._features = None
        self.current_assistance = None
        best, pareto = self.threshold_search()
        metrics, config = best["metrics"], best["config"]
        self.update_state(metrics, config, pareto)
        success = self.check_objectives(metrics)
        self.state.objectives_met = success

        self.episode_count += 1
        result = self._create_result(best, success)
        self.all_results.append(result)
        self._save_results()
        if self.cloud_enabled:
            self.report_episode(result)
        self._print_final_result(metrics, success)
        return best
    
    def _run_lightweight_mode(self) -> Optional[Dict]:
        """Edge correction -> cloud bounded guidance -> cached-score replay.

        No LLM model is loaded at the edge and no retraining/reprocessing is
        performed after cloud lightweight guidance.
        """
        print("\n" + "=" * 60)
        print("MODE: EDGE-FIRST LIGHTWEIGHT RECOVERY")
        print("=" * 60)
        self.state = OptimizationState(current_round=1)
        self.current_assistance = None

        print("\n[Initial] Preparing data...")
        if not self.prepare_data():
            print("Data preparation failed")
            return None

        # Initial training is allowed only when the configured source model has not
        # yet been created.  Subsequent lightweight recovery never retrains.
        if self.determine_mode() == RunMode.TRAIN_AND_TEST:
            print("\n[Initial] Training source-domain model...")
            if not self.train():
                return None

        best_result = None
        while self.state.current_round <= self.max_rounds:
            print(f"\n{'='*60}")
            print(f"RECOVERY ROUND {self.state.current_round}/{self.max_rounds}")
            print(f"{'='*60}")

            # Cached features are retained across threshold-only rounds.  This is
            # the key lightweight property: no repeated model inference is needed.
            best, pareto = self.threshold_search()
            metrics, config = best["metrics"], best["config"]
            self.update_state(metrics, config, pareto)
            success = self.check_objectives(metrics)

            print(f"\nRound {self.state.current_round} Results:")
            print(f"   closed_acc:  {metrics['closed_acc']:.4f} (target: {self.objectives['min_closed_acc']})")
            print(f"   open_auc:    {metrics['open_auc']:.4f} (target: {self.objectives['target_open_auc']})")
            print(f"   reject_rate: {metrics['reject_rate']:.4f} (auxiliary)")
            print(f"   FEASIBLE (A_c/A_o): {success}")

            self.episode_count += 1
            result = self._create_result(best, success)
            self.all_results.append(result)
            best_result = best

            if success:
                self.state.objectives_met = True
                print("\nOBJECTIVES MET")
                break
            if self.state.current_round >= self.max_rounds:
                print(f"\nMax rounds ({self.max_rounds}) reached")
                break
            if not self.cloud_enabled:
                print("\nEdge-only correction remains infeasible; cloud coordination is disabled")
                break

            print("\n[Cloud] Requesting bounded lightweight guidance...")
            guidance = self.request_cloud_guidance(metrics, config, pareto)
            if guidance is not None and not guidance.get("available", False):
                print("[Cloud] Requested guidance variant is unavailable; no substitute arm is executed")
                break
            if guidance is None:
                # A transport/service failure is a true cloud interruption. Use
                # only the conservative local fallback; never substitute another
                # cloud guidance arm, local LLM inference, or target retraining.
                diagnosis = {
                    "gaps": {
                        "closed_acc_gap": max(0.0, self.objectives["min_closed_acc"] - metrics["closed_acc"]),
                        "open_auc_gap": max(0.0, self.objectives["target_open_auc"] - metrics["open_auc"]),
                    }
                }
                fallback = self._rule_based_fallback(metrics, diagnosis, current_config=config)
                if fallback is None:
                    print("[Fallback] No executable local fallback")
                    break
                self.current_assistance = fallback
                self.assistance_type = AssistanceType.LIGHTWEIGHT
                print("[Fallback] Using bounded edge-local threshold interval")

            self.state.current_round += 1

        self._save_results()
        if self.cloud_enabled and self.all_results:
            self.report_episode(self.all_results[-1])
        if best_result:
            self._print_final_result(best_result["metrics"], self.state.objectives_met)
        return best_result
    
    def _create_result(self, best: Dict, success: bool) -> AdaptationResult:
        features = self._features or {}
        return AdaptationResult(
            episode_id=self.episode_count,
            round_id=self.state.current_round,
            timestamp=datetime.now().isoformat(),
            threshold_config=best["config"],
            metrics=best["metrics"],
            score_stats={
                "fused_mean": features.get("fused_mean", 0),
                "fused_std": features.get("fused_std", 1),
                "margin_mean": features.get("margin_mean", 0),
                "margin_std": features.get("margin_std", 1),
            },
            success=success,
            assistance_type=self.assistance_type.value,
            param_changes=dict(best["config"]),
        )
    
    def _save_results(self):
        """保存所有结果"""
        output = {
            "terminal_id": self.terminal_id,
            "timestamp": datetime.now().isoformat(),
            "objectives": self.objectives,
            "assistance_type": self.assistance_type.value,
            "total_rounds": self.state.current_round,
            "objectives_met": self.state.objectives_met,
            "best_metrics": self.state.best_metrics,
            "best_config": self.state.best_config,
            "all_results": [asdict(r) for r in self.all_results],
        }
        
        with open(self.output_dir / "edge_results.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"[EdgeAgentV2] Results saved to {self.output_dir / 'edge_results.json'}")
    
    def _print_final_result(self, metrics: Dict, success: bool):
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Total rounds: {self.state.current_round}")
        print(f"Feasible by A_c/A_o: {success}")
        print("\nMetrics:")
        print(f"  closed_acc:  {metrics['closed_acc']:.4f}")
        print(f"  open_auc:    {metrics['open_auc']:.4f}")
        print(f"  reject_rate: {metrics['reject_rate']:.4f} (auxiliary)")
        print("\nSUCCESS" if success else "\nOBJECTIVES NOT MET")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Edge Agent V2")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cloud", action="store_true", help="Enable cloud connection")
    parser.add_argument("--cloud-server", default="http://localhost:5000")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override max rounds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 云端配置
    if "cloud" not in cfg:
        cfg["cloud"] = {}
    cfg["cloud"]["enabled"] = args.cloud
    cfg["cloud"]["server_url"] = args.cloud_server
    
    # 覆盖 max_rounds
    if args.max_rounds:
        cfg["agent"]["max_rounds"] = args.max_rounds
    
    # 运行
    agent = EdgeAgentV2(cfg)
    result = agent.run()
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        torch.backends.cudnn.benchmark = True
    except:
        pass
    main()

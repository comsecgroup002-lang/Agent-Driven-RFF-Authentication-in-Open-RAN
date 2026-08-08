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
    LIGHTWEIGHT = "lightweight"           # 轻量级：LLM 调参
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
        self.original_cfg = copy.deepcopy(cfg)  # 保存原始配置
        self.terminal_id = self._gen_id()
        
        self.objectives = cfg["agent"]["objectives"]
        self.max_rounds = cfg["agent"].get("max_rounds", 5)
        self.paths = cfg["paths"]
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 云端配置
        cloud_cfg = cfg.get("cloud", {})
        self.cloud_enabled = cloud_cfg.get("enabled", False)
        self.cloud_server = cloud_cfg.get("server_url", "http://localhost:5000")
        
        # LLM 配置
        llm_cfg = cfg.get("agent", {}).get("llm", {})
        self.llm_enabled = llm_cfg.get("enable", True)
        self.llm_advisor = None
        
        # 状态
        self.episode_count = 0
        self.state = OptimizationState()
        self.all_results: List[AdaptationResult] = []
        
        # 云端协助
        self.current_assistance: Optional[Dict] = None
        self.assistance_type = AssistanceType.NONE
        
        # 缓存
        self._features = None
        self._model = None
        
        print(f"[EdgeAgentV2] Terminal: {self.terminal_id}")
        print(f"[EdgeAgentV2] Cloud: {self.cloud_enabled}")
        print(f"[EdgeAgentV2] LLM: {self.llm_enabled}")
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
        """请求云端协助"""
        if not self.cloud_enabled:
            return None
        try:
            import requests
            resp = requests.get(f"{self.cloud_server}/assistance/{self.terminal_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.current_assistance = data
                t = data.get("type", "none")
                if t == "lightweight":
                    self.assistance_type = AssistanceType.LIGHTWEIGHT
                    print(f"[EdgeAgentV2] 📥 LIGHTWEIGHT mode (LLM will guide)")
                elif t == "model_reprovision":
                    self.assistance_type = AssistanceType.MODEL_REPROVISION
                    print(f"[EdgeAgentV2] 📥 MODEL REPROVISIONING mode")
                return data
        except Exception as e:
            print(f"[EdgeAgentV2] Cloud request failed: {e}")
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
        """上报适应记录"""
        if not self.cloud_enabled:
            return False
        try:
            import requests
            data = {
                "terminal_id": self.terminal_id,
                "timestamp": result.timestamp,
                "episode_id": result.episode_id,
                "delta_fused": result.threshold_config.get("rho", 0),
                "accept_quantile": result.threshold_config.get("accept_quantile", 0.9),
                "margin_quantile": result.threshold_config.get("margin_quantile", 0.2),
                "closed_acc": result.metrics.get("closed_acc", 0),
                "open_auc": result.metrics.get("open_auc", 0),
                "reject_rate": result.metrics.get("reject_rate", 0),
                "fused_mean": result.score_stats.get("fused_mean", 0),
                "fused_std": result.score_stats.get("fused_std", 1),
                "recovery_success": result.success,
                "model_config": self.cfg.get("model", {}),
                "domain_state": self.cfg.get("signal_augment", {}),
            }
            
            # 如果成功，上传权重
            files = {}
            weights_path = self.paths.get("model_weights")
            if result.success and weights_path and os.path.exists(weights_path):
                files["weights_file"] = open(weights_path, 'rb')
            
            # 增加超时时间：上传权重时需要更长时间
            timeout = 120 if files else 60
            
            if files:
                resp = requests.post(f"{self.cloud_server}/episode", 
                                    data={"json": json.dumps(data)}, files=files, timeout=timeout)
            else:
                resp = requests.post(f"{self.cloud_server}/episode", json=data, timeout=timeout)
            
            if resp.status_code == 200:
                print(f"[EdgeAgentV2] 📤 Episode reported")
                return True
        except requests.exceptions.Timeout:
            print(f"[EdgeAgentV2] Report timeout (this is OK, cloud may be processing)")
            return True  # 超时不算失败，云端可能正在处理
        except Exception as e:
            print(f"[EdgeAgentV2] Report failed: {e}")
        return False
    

    
    def _init_llm_advisor(self):
        """初始化 LLM 顾问"""
        if self.llm_advisor is not None:
            return
        
        if not self.llm_enabled:
            return
        
        try:
            from llm_advisor import LLMAdvisor
            llm_cfg = self.cfg.get("agent", {}).get("llm", {})
            self.llm_advisor = LLMAdvisor(llm_cfg)
            print("[EdgeAgentV2] LLM Advisor initialized")
        except Exception as e:
            print(f"[EdgeAgentV2] LLM init failed: {e}")
            self.llm_advisor = None
    
    def get_llm_advice(self, metrics: Dict, pareto_front: List[Dict]) -> Optional[Dict]:

        # 构建诊断信息
        diagnosis = {
            "improvement_trend": self.state.improvement_trend,
            "at_pareto_limit": self.state.at_pareto_limit,
            "training_rounds": self.state.current_round,
            "gaps": {
                "closed_acc_gap": max(0, self.objectives["min_closed_acc"] - metrics["closed_acc"]),
                "open_auc_gap": max(0, self.objectives["target_open_auc"] - metrics["open_auc"]),
                "reject_rate_excess": max(0, metrics["reject_rate"] - self.objectives["max_reject_rate"]),
            }
        }
        
        # Pareto 摘要
        pareto_summary = f"Pareto front size: {len(pareto_front)}\n"
        if pareto_front:
            best = pareto_front[0]
            pareto_summary += f"Best on Pareto: closed={best['metrics']['closed_acc']:.4f}, open={best['metrics']['open_auc']:.4f}"
        
        # ★ 获取 RAG 指导
        rag_context = None
        retrieved_cases = None
        policy_suggestion = None
        safe_ranges = None
        max_steps = None
        global_summary = None
        
        if self.current_assistance and self.assistance_type == AssistanceType.LIGHTWEIGHT:
            # RAG context（格式化的检索结果）
            rag_context = self.current_assistance.get("rag_context")
            
            # 检索到的案例
            retrieved_cases = self.current_assistance.get("retrieved_cases", {})
            
            # Policy Guard 建议
            policy_suggestion = self.current_assistance.get("policy_suggestion", {})
            safe_ranges = self.current_assistance.get("safe_param_ranges", {})
            max_steps = self.current_assistance.get("max_step_sizes", {})
            
            # 全局摘要
            global_summary = self.current_assistance.get("global_summary", {})
            
            # 打印 RAG 信息
            num_episodes = self.current_assistance.get("num_episodes", 0)
            success_cases = len(retrieved_cases.get("success", []))
            failure_cases = len(retrieved_cases.get("failure", []))
            print(f"[EdgeAgentV2] 📊 RAG: {num_episodes} episodes, retrieved {success_cases} success + {failure_cases} failure cases")
            
            if policy_suggestion:
                print(f"[EdgeAgentV2] 🛡️ Policy suggestion: {policy_suggestion}")
        
        # 构建云端经验（用于 LLM 和 fallback）
        cloud_experience = {
            "rag_context": rag_context,
            "retrieved_cases": retrieved_cases,
            "policy_suggestion": policy_suggestion,
            "safe_param_ranges": safe_ranges,
            "max_step_sizes": max_steps,
            "global_summary": global_summary,
            # 兼容旧格式
            "total_episodes": global_summary.get("total_episodes", 0) if global_summary else 0,
            "success_rate": global_summary.get("success_rate", 0) if global_summary else 0,
        }
        
        # 从成功案例提取推荐范围
        if retrieved_cases and retrieved_cases.get("success"):
            retain_ratios = []
            for case in retrieved_cases["success"]:
                action = case.get("action", {})
                if "retain_ratio" in action:
                    retain_ratios.append(action["retain_ratio"])
            if retain_ratios:
                cloud_experience["recommended_retain_ratio"] = (
                    min(retain_ratios), max(retain_ratios)
                )
        
        # 尝试 LLM
        if self.llm_enabled:
            self._init_llm_advisor()
            
            if self.llm_advisor is not None:
                try:
                    advice = self.llm_advisor.get_training_advice(
                        current_params=self.cfg,
                        best_metrics=metrics,
                        objectives=self.objectives,
                        pareto_summary=pareto_summary,
                        history=self.state.history,
                        diagnosis=diagnosis,
                        model_info=self.cfg.get("model", {}),
                        cloud_experience=cloud_experience,
                    )
                    
                    if advice and advice.param_changes:
                        # ★ 本地 Policy Guard 二次审查
                        approved, adjusted, warnings = self._policy_guard_review(
                            advice.param_changes, metrics, safe_ranges, max_steps
                        )
                        
                        if warnings:
                            print(f"[EdgeAgentV2] ⚠️ Policy warnings: {warnings}")
                        
                        print(f"[EdgeAgentV2] LLM Advice: {advice.analysis}")
                        print(f"[EdgeAgentV2] Original params: {advice.param_changes}")
                        print(f"[EdgeAgentV2] After policy guard: {adjusted}")
                        
                        return {
                            "param_changes": adjusted,
                            "requires_reprocessing": advice.requires_reprocessing,
                            "analysis": advice.analysis,
                            "reasoning": advice.reasoning,
                            "confidence": advice.confidence,
                            "policy_approved": approved,
                            "policy_warnings": warnings,
                            "source": "llm_with_rag",
                        }
                    
                except Exception as e:
                    print(f"[EdgeAgentV2] LLM advice failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # LLM 不可用，使用 Policy Guard 建议或规则回退
        if policy_suggestion:
            print("[EdgeAgentV2] Using cloud Policy Guard suggestion...")
            return {
                "param_changes": policy_suggestion,
                "requires_reprocessing": True,
                "analysis": "Using cloud Policy Guard suggestion (LLM unavailable)",
                "reasoning": "Based on similar successful cases",
                "confidence": 0.7,
                "source": "policy_guard",
            }
        
        print("[EdgeAgentV2] Using rule-based fallback...")
        return self._rule_based_fallback(metrics, diagnosis, cloud_experience)
    
    def _policy_guard_review(self, param_changes: Dict, metrics: Dict,
                              safe_ranges: Dict = None, max_steps: Dict = None) -> Tuple[bool, Dict, List[str]]:

        safe_ranges = safe_ranges or {
            "retain_ratio": [0.35, 0.65],
            "engineering_aug_prob": [0.5, 0.85],
        }
        max_steps = max_steps or {
            "retain_ratio": 0.08,
            "engineering_aug_prob": 0.1,
        }
        
        warnings = []
        adjusted = {}
        approved = True
        
        sig_aug = self.cfg.get("signal_augment", {})
        
        for key, new_value in param_changes.items():
            current_value = sig_aug.get(key, 0.5)
            
            # 检查安全范围
            if key in safe_ranges:
                min_val, max_val = safe_ranges[key]
                if new_value < min_val:
                    warnings.append(f"{key}={new_value:.3f} < {min_val}, clamped")
                    new_value = min_val
                    approved = False
                elif new_value > max_val:
                    warnings.append(f"{key}={new_value:.3f} > {max_val}, clamped")
                    new_value = max_val
                    approved = False
            
            # 检查步长
            if key in max_steps:
                max_step = max_steps[key]
                delta = new_value - current_value
                if abs(delta) > max_step:
                    clamped = current_value + (max_step if delta > 0 else -max_step)
                    warnings.append(f"{key} step {delta:.3f} > {max_step}, clamped to {clamped:.3f}")
                    new_value = clamped
                    approved = False
            
            adjusted[key] = new_value
        
        return approved, adjusted, warnings
    
    def _rule_based_fallback(self, metrics: Dict, diagnosis: Dict, 
                              cloud_experience: Dict = None) -> Optional[Dict]:

        gaps = diagnosis.get("gaps", {})
        at_limit = diagnosis.get("at_pareto_limit", False)
        
        sig_aug = self.cfg.get("signal_augment", {})
        current_retain = sig_aug.get("retain_ratio", 0.5)
        current_eng = sig_aug.get("engineering_aug_prob", 0.7)
        
        changes = {}
        analysis = ""
        reasoning = ""
        
        open_gap = gaps.get("open_auc_gap", 0)
        closed_gap = gaps.get("closed_acc_gap", 0)
        reject_excess = gaps.get("reject_rate_excess", 0)
        
        # ★ 获取云端推荐范围
        if cloud_experience:
            rec_retain = cloud_experience.get("recommended_retain_ratio", (0.4, 0.6))
            rec_eng = cloud_experience.get("recommended_eng_aug_prob", (0.6, 0.8))
            print(f"[Fallback] Using cloud recommendations: retain={rec_retain}, eng={rec_eng}")
        else:
            rec_retain = (0.4, 0.6)
            rec_eng = (0.6, 0.8)
        
        print(f"[Fallback] Analyzing: open_gap={open_gap:.4f}, closed_gap={closed_gap:.4f}, reject_excess={reject_excess:.4f}")
        
        # 策略选择（保守，小步调整）
        if closed_gap > 0.05:
            # closed_acc 差距大 - 适度增加 retain_ratio
            new_retain = min(rec_retain[1], current_retain + 0.05)
            changes["retain_ratio"] = max(0.35, min(0.65, new_retain))
            analysis = f"Large closed_acc gap ({closed_gap:.4f})"
            reasoning = f"Increase retain_ratio conservatively: {current_retain:.2f} -> {changes['retain_ratio']:.2f}"
        
        elif open_gap > 0.05:
            # open_auc 差距大 - 适度降低 retain_ratio
            new_retain = max(rec_retain[0], current_retain - 0.05)
            changes["retain_ratio"] = max(0.35, min(0.65, new_retain))
            analysis = f"Large open_auc gap ({open_gap:.4f})"
            reasoning = f"Decrease retain_ratio conservatively: {current_retain:.2f} -> {changes['retain_ratio']:.2f}"
        
        elif reject_excess > 0.1:
            # reject_rate 过高 - 往中间靠
            target = (rec_retain[0] + rec_retain[1]) / 2
            if current_retain < target:
                changes["retain_ratio"] = min(target, current_retain + 0.03)
            else:
                changes["retain_ratio"] = max(target, current_retain - 0.03)
            analysis = f"High reject_rate ({reject_excess:.4f})"
            reasoning = "Move toward balanced parameters"
        
        elif closed_gap > 0.02:
            # 小差距 - 微调
            changes["retain_ratio"] = min(rec_retain[1], current_retain + 0.03)
            analysis = f"Small closed_acc gap ({closed_gap:.4f})"
            reasoning = "Fine-tuning"
        
        elif open_gap > 0:
            # 任何 open_auc 差距
            changes["retain_ratio"] = max(rec_retain[0], current_retain - 0.03)
            analysis = f"Small open_auc gap ({open_gap:.4f})"
            reasoning = "Fine-tuning"
        
        if changes:
            # 确保在安全范围内
            if "retain_ratio" in changes:
                changes["retain_ratio"] = max(0.35, min(0.65, changes["retain_ratio"]))
            
            print(f"[Fallback] Suggested changes: {changes}")
            print(f"[Fallback] Analysis: {analysis}")
            return {
                "param_changes": changes,
                "requires_reprocessing": True,
                "analysis": analysis,
                "reasoning": reasoning,
                "confidence": 0.6,
                "source": "rule_based_fallback",
            }
        
        # 没有明确方向，使用默认策略
        print("[Fallback] No clear direction, using balanced default")
        target_retain = (rec_retain[0] + rec_retain[1]) / 2
        
        # 向目标值靠近
        if abs(current_retain - target_retain) > 0.02:
            if current_retain < target_retain:
                changes["retain_ratio"] = current_retain + 0.03
            else:
                changes["retain_ratio"] = current_retain - 0.03
            
            return {
                "param_changes": changes,
                "requires_reprocessing": True,
                "analysis": "Moving toward balanced parameters",
                "reasoning": f"Adjusting retain_ratio toward {target_retain:.2f}",
                "confidence": 0.4,
                "source": "rule_based_fallback",
            }
        
        return None
    
    def apply_param_changes(self, param_changes: Dict) -> bool:

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
        """获取搜索网格"""
        grid = {
            "accept_q": [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98],
            "margin_q": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            "rho": [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        }
        
        # 轻量级指导约束搜索空间
        if self.assistance_type == AssistanceType.LIGHTWEIGHT and self.current_assistance:
            interval = self.current_assistance.get("threshold_interval", {})
            
            ar = interval.get("accept_quantile_range", [0.70, 0.98])
            grid["accept_q"] = [q for q in grid["accept_q"] if ar[0] <= q <= ar[1]]
            
            mr = interval.get("margin_quantile_range", [0.05, 0.35])
            grid["margin_q"] = [q for q in grid["margin_q"] if mr[0] <= q <= mr[1]]
            
            rr = interval.get("delta_fused_range", [-0.15, 0.35])
            grid["rho"] = [r for r in grid["rho"] if rr[0] <= r <= rr[1]]
            
            print(f"[EdgeAgentV2] Constrained grid: accept={len(grid['accept_q'])}, margin={len(grid['margin_q'])}, rho={len(grid['rho'])}")
        
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
        obj = self.objectives
        cg = max(0, obj["min_closed_acc"] - m["closed_acc"])
        og = max(0, obj["target_open_auc"] - m["open_auc"])
        re = max(0, m["reject_rate"] - obj["max_reject_rate"])
        if cg == 0 and og == 0 and re == 0:
            return m["closed_acc"] * 2 + m["open_auc"] * 1.5 + (1 - m["reject_rate"]) * 0.5
        return m["closed_acc"] + m["open_auc"] - (cg * 5 + og * 3 + re)
    
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
        obj = self.objectives
        feasible = [r for r in pareto if
                    r["metrics"]["closed_acc"] >= obj["min_closed_acc"] and
                    r["metrics"]["open_auc"] >= obj["target_open_auc"] 
                    
        if feasible:
            return max(feasible, key=lambda x: x["metrics"]["open_auc"])
        closed_ok = [r for r in pareto if r["metrics"]["closed_acc"] >= obj["min_closed_acc"]]
        if closed_ok:
            return max(closed_ok, key=lambda x: x["metrics"]["open_auc"])
        return pareto[0]
    
    def check_objectives(self, metrics: Dict) -> bool:
        """检查是否达到目标"""
        return (
            metrics["closed_acc"] >= self.objectives["min_closed_acc"] and
            metrics["open_auc"] >= self.objectives["target_open_auc"] and
            metrics["reject_rate"] <= self.objectives["max_reject_rate"]
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
        """
        主运行流程
        
        轻量级模式：测试 → 不达标 → LLM调参 → 重新处理+训练 → 测试（循环）
        模型重配置模式：下载模型 → 测试
        """
        print("\n" + "=" * 70)
        print("EDGE AGENT V2 - ADAPTIVE AUTHENTICATION SYSTEM")
        print("=" * 70)
        print(f"Terminal: {self.terminal_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # ================================================================
        # Step 1: 请求云端协助
        # ================================================================
        if self.cloud_enabled:
            print("\n[Step 1] Requesting cloud assistance...")
            self.request_assistance()
        else:
            print("\n[Step 1] Cloud disabled, using default mode")
            self.assistance_type = AssistanceType.LIGHTWEIGHT
        
        # ================================================================
        # Step 2: 根据协助类型执行
        # ================================================================
        
        if self.assistance_type == AssistanceType.MODEL_REPROVISION:
            # 模型重配置模式
            return self._run_model_reprovision_mode()
        else:
            # 轻量级模式（默认）
            return self._run_lightweight_mode()
    
    def _run_model_reprovision_mode(self) -> Optional[Dict]:

        print("\n" + "=" * 60)
        print("MODE: MODEL REPROVISIONING")
        print("=" * 60)
        
        # 初始化状态
        self.state = OptimizationState()
        model_downloaded = False
        
        # 尝试下载云端模型
        if self.cloud_enabled and self.current_assistance:
            if self.current_assistance.get("available", False):
                print("\n[Step 2] Downloading model package...")
                if self.download_model_package():
                    model_downloaded = True
                    print("[EdgeAgentV2] ✅ Cloud model applied")
                else:
                    print("[EdgeAgentV2] ⚠️ Download failed, using local model")
            else:
                print("[Step 2] No model available from cloud yet")
        
        # 准备数据
        print("\n[Step 3] Preparing data...")
        if not self.prepare_data():
            print("❌ Data preparation failed")
            return None
        
        # 检测权重
        mode = self.determine_mode()
        
        # 如果没有权重，需要训练
        if mode == RunMode.TRAIN_AND_TEST:
            print("\n[Step 4] Training...")
            self.state.current_round = 1
            if not self.train():
                return None
        else:
            print("\n[Step 4] Using existing weights")
        
        # ================================================================
        # 优化循环（与 lightweight 模式类似，但使用 rule-based fallback）
        # ================================================================
        
        best_result = None
        
        while self.state.current_round <= self.max_rounds:
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION ROUND {self.state.current_round}/{self.max_rounds}")
            print(f"{'='*60}")
            
            # 清除特征缓存
            self._features = None
            
            # 阈值搜索
            best, pareto = self.threshold_search()
            metrics = best["metrics"]
            config = best["config"]
            
            # 更新状态
            self.update_state(metrics, config, pareto)
            
            # 检查是否达标
            success = self.check_objectives(metrics)
            
            print(f"\n📊 Round {self.state.current_round} Results:")
            print(f"   closed_acc:  {metrics['closed_acc']:.4f} (target: {self.objectives['min_closed_acc']})")
            print(f"   open_auc:    {metrics['open_auc']:.4f} (target: {self.objectives['target_open_auc']})")
            print(f"   reject_rate: {metrics['reject_rate']:.4f} (max: {self.objectives['max_reject_rate']})")
            print(f"   SUCCESS: {success}")
            
            # 记录结果
            self.episode_count += 1
            result = self._create_result(best, success)
            self.all_results.append(result)
            
            if success:
                print("\n✅ OBJECTIVES MET!")
                self.state.objectives_met = True
                best_result = best
                break
            
            # 检查是否还有轮次
            if self.state.current_round >= self.max_rounds:
                print(f"\n⚠️ Max rounds ({self.max_rounds}) reached")
                best_result = best
                break
            
            # ============================================================
            # 使用 Rule-based Fallback 调参（model_reprovision 模式不使用 LLM）
            # ============================================================
            print("\n[Fallback] Computing parameter adjustment...")
            
            diagnosis = {
                "improvement_trend": self.state.improvement_trend,
                "at_pareto_limit": self.state.at_pareto_limit,
                "training_rounds": self.state.current_round,
                "gaps": {
                    "closed_acc_gap": max(0, self.objectives["min_closed_acc"] - metrics["closed_acc"]),
                    "open_auc_gap": max(0, self.objectives["target_open_auc"] - metrics["open_auc"]),
                    "reject_rate_excess": max(0, metrics["reject_rate"] - self.objectives["max_reject_rate"]),
                }
            }
            
            advice = self._rule_based_fallback(metrics, diagnosis)
            
            if advice is None or not advice.get("param_changes"):
                print("[Fallback] No adjustment available, stopping")
                best_result = best
                break
            
            # 应用参数变更
            self.apply_param_changes(advice["param_changes"])
            
            # 重新处理和训练
            if advice.get("requires_reprocessing", True):
                print("\n[Reprocessing] Signal processing with new parameters...")
                if not self.prepare_data(force_reprocess=True):
                    print("❌ Reprocessing failed")
                    best_result = best
                    break
            
            print("\n[Training] Training with new parameters...")
            self.state.current_round += 1
            if not self.train():
                print("❌ Training failed")
                best_result = best
                break
        
        # ================================================================
        # 完成
        # ================================================================
        
        # 保存结果
        self._save_results()
        
        # 上报最终结果
        if self.cloud_enabled and self.all_results:
            print("\n[Reporting] Uploading final result...")
            self.report_episode(self.all_results[-1])
        
        # 打印最终结果
        if best_result:
            self._print_final_result(best_result["metrics"], self.state.objectives_met)
        
        return best_result
    
    def _run_lightweight_mode(self) -> Optional[Dict]:

        print("\n" + "=" * 60)
        print("MODE: LIGHTWEIGHT (LLM-guided optimization)")
        print("=" * 60)
        
        # 初始化
        self.state = OptimizationState()
        
        # 准备数据（首次）
        print("\n[Initial] Preparing data...")
        if not self.prepare_data():
            print("❌ Data preparation failed")
            return None
        
        # 检测权重，决定是否需要首次训练
        mode = self.determine_mode()
        if mode == RunMode.TRAIN_AND_TEST:
            print("\n[Initial] Training...")
            self.state.current_round = 1
            if not self.train():
                return None
        
        # ================================================================
        # 优化循环
        # ================================================================
        
        best_result = None
        
        while self.state.current_round <= self.max_rounds:
            print(f"\n{'='*60}")
            print(f"OPTIMIZATION ROUND {self.state.current_round}/{self.max_rounds}")
            print(f"{'='*60}")
            
            # 清除特征缓存
            self._features = None
            
            # 阈值搜索
            best, pareto = self.threshold_search()
            metrics = best["metrics"]
            config = best["config"]
            
            # 更新状态
            self.update_state(metrics, config, pareto)
            
            # 检查是否达标
            success = self.check_objectives(metrics)
            
            print(f"\n📊 Round {self.state.current_round} Results:")
            print(f"   closed_acc:  {metrics['closed_acc']:.4f} (target: {self.objectives['min_closed_acc']})")
            print(f"   open_auc:    {metrics['open_auc']:.4f} (target: {self.objectives['target_open_auc']})")
            print(f"   reject_rate: {metrics['reject_rate']:.4f} (max: {self.objectives['max_reject_rate']})")
            print(f"   SUCCESS: {success}")
            
            # 记录结果
            self.episode_count += 1
            result = self._create_result(best, success)
            self.all_results.append(result)
            
            if success:
                print("\n✅ OBJECTIVES MET!")
                self.state.objectives_met = True
                best_result = best
                break
            
            # 检查是否还有轮次
            if self.state.current_round >= self.max_rounds:
                print(f"\n⚠️ Max rounds ({self.max_rounds}) reached")
                best_result = best
                break
            
            # 检查是否在 Pareto 边界
            if self.state.at_pareto_limit:
                print("\n⚠️ At Pareto limit, need parameter adjustment")
            
            # ============================================================
            # LLM 调参 (或 rule-based fallback)
            # ============================================================
            print("\n[Advisor] Requesting parameter adjustment advice...")
            
            advice = self.get_llm_advice(metrics, pareto)
            
            if advice is None or not advice.get("param_changes"):
                print("[Advisor] No advice available, stopping optimization")
                best_result = best
                break
            
            # 显示建议来源
            source = advice.get("source", "unknown")
            print(f"[Advisor] Source: {source}")
            print(f"[Advisor] Analysis: {advice.get('analysis', '')}")
            
            param_changes = advice.get("param_changes", {})
            requires_reprocessing = advice.get("requires_reprocessing", False)
            
            # 应用参数变更
            self.apply_param_changes(param_changes)
            
            # 根据是否需要重新处理决定下一步
            if requires_reprocessing:
                print("\n[Reprocessing] Signal processing with new parameters...")
                if not self.prepare_data(force_reprocess=True):
                    print("❌ Reprocessing failed")
                    best_result = best
                    break
            
            # 重新训练
            print("\n[Training] Training with new parameters...")
            self.state.current_round += 1
            if not self.train():
                print("❌ Training failed")
                best_result = best
                break
        
        # ================================================================
        # 完成
        # ================================================================
        
        # 保存所有结果
        self._save_results()
        
        # 上报最终结果
        if self.cloud_enabled and self.all_results:
            print("\n[Reporting] Uploading final result...")
            self.report_episode(self.all_results[-1])
        
        # 打印最终结果
        if best_result:
            self._print_final_result(best_result["metrics"], self.state.objectives_met)
        
        return best_result
    
    def _create_result(self, best: Dict, success: bool) -> AdaptationResult:
        """创建适应结果"""
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
            param_changes=dict(self.cfg.get("signal_augment", {})),
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
        """打印最终结果"""
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Total rounds: {self.state.current_round}")
        print(f"Objectives met: {success}")
        print(f"\nMetrics:")
        print(f"  closed_acc:  {metrics['closed_acc']:.4f}")
        print(f"  open_auc:    {metrics['open_auc']:.4f}")
        print(f"  reject_rate: {metrics['reject_rate']:.4f}")
        
        if success:
            print("\n✅ SUCCESS")
        else:
            print("\n❌ OBJECTIVES NOT MET")


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

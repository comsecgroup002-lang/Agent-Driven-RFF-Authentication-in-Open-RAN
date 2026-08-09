

import os
import sys
import json
import copy
import shutil
import tarfile
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

import yaml
import numpy as np

try:
    from flask import Flask, request, jsonify, send_file
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("[Warning] Flask not installed, REST API disabled")



class AssistanceMode(Enum):

    LIGHTWEIGHT = "lightweight"
    MODEL_REPROVISION = "model_reprovision"
    AUTO = "auto"




@dataclass
class StateVector:
    """Compact operational state used for cloud-side retrieval.

    Feasibility is defined only by A_c and A_o.  Rejection rate R is retained
    as auxiliary service-availability information and may influence ranking or
    retrieval similarity, but it never creates an additional feasibility gap.
    """

    closed_acc: float = 0.0
    open_auc: float = 0.0
    reject_rate: float = 0.0

    closed_gap: float = 0.0
    open_gap: float = 0.0
    reject_excess: float = 0.0  # auxiliary diagnostic only

    # Current threshold-space operating point.
    accept_quantile: float = 0.90
    margin_quantile: float = 0.20
    rho: float = 0.0

    # Low-order score-distribution statistics.
    fused_mean: float = 0.0
    fused_std: float = 1.0

    # Legacy fields are retained only so old state.json files can still load.
    retain_ratio: float = 0.5
    engineering_aug_prob: float = 0.7

    def to_vector(self) -> np.ndarray:
        # R is auxiliary but useful for similarity. reject_excess is omitted so
        # it cannot act as a third feasibility dimension.
        return np.array([
            self.closed_acc,
            self.open_auc,
            self.reject_rate,
            self.closed_gap,
            self.open_gap,
            self.accept_quantile,
            self.margin_quantile,
            self.rho,
            self.fused_mean,
            self.fused_std,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, d: Dict, objectives: Dict = None) -> 'StateVector':
        d = d or {}
        obj = objectives or {"min_closed_acc": 0.90, "target_open_auc": 0.85, "max_reject_rate": 0.60}
        metrics = d.get("metrics", {}) or {}
        thresholds = d.get("threshold_config", {}) or {}
        sig_aug = d.get("signal_augment", d.get("domain_state", {})) or {}

        closed_acc = float(d.get("closed_acc", metrics.get("closed_acc", 0.0)))
        open_auc = float(d.get("open_auc", metrics.get("open_auc", 0.0)))
        reject_rate = float(d.get("reject_rate", metrics.get("reject_rate", 0.0)))

        accept_quantile = float(d.get(
            "accept_quantile", thresholds.get("accept_quantile", 0.90)
        ))
        margin_quantile = float(d.get(
            "margin_quantile", thresholds.get("margin_quantile", 0.20)
        ))
        rho = float(d.get(
            "rho", d.get("delta_fused", thresholds.get("rho", 0.0))
        ))

        max_reject = float(obj.get("max_reject_rate", 0.60))
        return cls(
            closed_acc=closed_acc,
            open_auc=open_auc,
            reject_rate=reject_rate,
            closed_gap=max(0.0, float(obj["min_closed_acc"]) - closed_acc),
            open_gap=max(0.0, float(obj["target_open_auc"]) - open_auc),
            reject_excess=max(0.0, reject_rate - max_reject),
            accept_quantile=accept_quantile,
            margin_quantile=margin_quantile,
            rho=rho,
            fused_mean=float(d.get("fused_mean", 0.0)),
            fused_std=float(d.get("fused_std", 1.0)),
            retain_ratio=float(sig_aug.get("retain_ratio", d.get("retain_ratio", 0.5))),
            engineering_aug_prob=float(sig_aug.get(
                "engineering_aug_prob", d.get("engineering_aug_prob", 0.7)
            )),
        )


@dataclass
class EpisodeRecord:

    episode_id: str
    terminal_id: str
    timestamp: str
    
    # 状态（输入）
    state_before: StateVector
    
    # 动作（参数变化）
    action: Dict  # {"retain_ratio": 0.45, ...}
    
    # 结果（输出）
    state_after: StateVector
    
    # 是否成功
    success: bool
    
    # 改善程度
    improvement: Dict  # {"closed_acc_delta": 0.02, "open_auc_delta": 0.05, ...}
    
    # 元数据
    model_config: Dict = field(default_factory=dict)
    weights_file: Optional[str] = None


@dataclass
class RetrievedCase:

    episode: EpisodeRecord
    similarity: float
    relevance_reason: str


@dataclass
class RAGGuidance:

    # 检索到的案例
    similar_success_cases: List[RetrievedCase]
    similar_failure_cases: List[RetrievedCase]
    
    # 全局摘要
    global_summary: Dict
    
    # LLM 生成的建议
    llm_suggestion: Dict
    
    # Policy Guard 审查结果
    policy_approved: bool
    policy_adjustments: Dict
    
    # 最终建议
    final_recommendation: Dict


@dataclass
class ThresholdPriorInterval:

    domain_cluster: str
    delta_fused_min: float
    delta_fused_max: float
    delta_margin_min: float
    delta_margin_max: float
    accept_quantile_min: float = 0.80
    accept_quantile_max: float = 0.95
    margin_quantile_min: float = 0.10
    margin_quantile_max: float = 0.30
    confidence: float = 0.5
    num_samples: int = 0
    created_at: str = ""


@dataclass
class ModelPackage:

    version: str
    domain_cluster: str
    created_at: str
    weights_file: str
    model_config: Dict
    baseline_thresholds: List[Dict]
    signal_augment: Dict
    avg_metrics: Dict
    num_contributors: int
    package_path: Optional[str] = None


@dataclass
class EdgeNodeState:

    terminal_id: str
    last_seen: str
    current_metrics: Dict
    mismatch_severity: str = "none"
    consecutive_failures: int = 0
    needs_reprovisioning: bool = False
    num_episodes: int = 0


# ============================================================================
# Episode Store - RAG 的向量存储
# ============================================================================

class EpisodeStore:

    
    def __init__(self, objectives: Dict):
        self.objectives = objectives
        self.episodes: List[EpisodeRecord] = []
        self.success_episodes: List[EpisodeRecord] = []
        self.failure_episodes: List[EpisodeRecord] = []
        
        # 用于快速检索的索引
        self._state_vectors: Optional[np.ndarray] = None
        self._needs_reindex = True
    
    def add(self, episode: EpisodeRecord):

        self.episodes.append(episode)
        if episode.success:
            self.success_episodes.append(episode)
        else:
            self.failure_episodes.append(episode)
        self._needs_reindex = True
    
    def _build_index(self):

        if not self._needs_reindex or not self.episodes:
            return
        
        vectors = []
        for ep in self.episodes:
            vectors.append(ep.state_before.to_vector())
        
        self._state_vectors = np.array(vectors)
        self._needs_reindex = False
    
    def retrieve(self, query_state: StateVector, k: int = 5, 
                 success_only: bool = None) -> List[RetrievedCase]:

        if not self.episodes:
            return []
        
        self._build_index()
        
        # 选择候选集
        if success_only is True:
            candidates = self.success_episodes
        elif success_only is False:
            candidates = self.failure_episodes
        else:
            candidates = self.episodes
        
        if not candidates:
            return []
        
        # 计算相似度
        query_vec = query_state.to_vector()
        results = []
        
        for ep in candidates:
            ep_vec = ep.state_before.to_vector()
            
            # 余弦相似度
            similarity = self._cosine_similarity(query_vec, ep_vec)
            
            # 生成相关性说明
            reason = self._explain_relevance(query_state, ep)
            
            results.append(RetrievedCase(
                episode=ep,
                similarity=similarity,
                relevance_reason=reason,
            ))
        
        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:k]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:

        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _explain_relevance(self, query: StateVector, ep: EpisodeRecord) -> str:
        reasons = []
        if abs(query.closed_acc - ep.state_before.closed_acc) < 0.05:
            reasons.append("similar A_c")
        if abs(query.open_auc - ep.state_before.open_auc) < 0.05:
            reasons.append("similar A_o")
        if abs(query.closed_gap - ep.state_before.closed_gap) < 0.03:
            reasons.append("similar A_c gap")
        if abs(query.open_gap - ep.state_before.open_gap) < 0.03:
            reasons.append("similar A_o gap")
        if abs(query.rho - ep.state_before.rho) < 0.10:
            reasons.append("similar threshold offset")
        return "; ".join(reasons) if reasons else "general operational similarity"
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.episodes:
            return {"total": 0, "success": 0, "failure": 0}
        
        return {
            "total": len(self.episodes),
            "success": len(self.success_episodes),
            "failure": len(self.failure_episodes),
            "success_rate": len(self.success_episodes) / len(self.episodes),
        }



class PolicyGuard:
    """Deterministic cloud-side guard for threshold-space advisory proposals."""

    SAFE_RANGES = {
        "accept_quantile": (0.80, 0.95),
        "margin_quantile": (0.10, 0.30),
        "rho": (-0.15, 0.35),
    }
    MAX_STEP = {
        "accept_quantile": 0.05,
        "margin_quantile": 0.05,
        "rho": 0.10,
    }
    INTERVAL_HALF_WIDTH = {
        "accept_quantile": 0.05,
        "margin_quantile": 0.05,
        "rho": 0.10,
    }

    def __init__(self, objectives: Dict):
        self.objectives = objectives

    @staticmethod
    def _current_thresholds(current_state: StateVector) -> Dict[str, float]:
        return {
            "accept_quantile": float(current_state.accept_quantile),
            "margin_quantile": float(current_state.margin_quantile),
            "rho": float(current_state.rho),
        }

    def review(self, suggestion: Dict, current_state: StateVector,
               retrieved_cases: List[RetrievedCase] = None) -> Tuple[bool, Dict, List[str]]:
        """Clamp a candidate threshold patch into the deterministic safe set.

        Failed cases are contextual evidence only.  They are intentionally not
        implemented as a hard rejection rule.
        """
        warnings: List[str] = []
        adjusted: Dict[str, float] = {}
        approved = True
        current = self._current_thresholds(current_state)

        raw = suggestion.get("threshold_patch", suggestion.get("param_changes", suggestion))
        if not isinstance(raw, dict):
            return False, {}, ["proposal is not a dictionary"]

        for key, raw_value in raw.items():
            if key not in self.SAFE_RANGES:
                warnings.append(f"unsupported field ignored: {key}")
                approved = False
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                warnings.append(f"non-numeric field ignored: {key}")
                approved = False
                continue

            max_step = self.MAX_STEP[key]
            delta = value - current[key]
            if abs(delta) > max_step:
                value = current[key] + (max_step if delta > 0 else -max_step)
                warnings.append(f"{key} step limited to {max_step:.3f}")
                approved = False

            lo, hi = self.SAFE_RANGES[key]
            clipped = max(lo, min(hi, value))
            if clipped != value:
                warnings.append(f"{key} projected to [{lo}, {hi}]")
                approved = False
            adjusted[key] = float(clipped)

        if not adjusted:
            approved = False
        return approved, adjusted, warnings

    def suggest_safe_adjustment(self, current_state: StateVector,
                                retrieved_success: List[RetrievedCase]) -> Dict:
        """Non-generative top-K retrieval guidance used by the ablation/fallback.

        Up to the top three successful cases are combined using normalized
        nonnegative cosine-similarity weights.  If no successful case exists,
        deterministic retrieval guidance is unavailable and an empty mapping is
        returned.
        """
        cases = list(retrieved_success or [])[:3]
        usable = []
        for case in cases:
            action = case.episode.action or {}
            if any(k in action for k in self.SAFE_RANGES):
                usable.append(case)
        if not usable:
            return {}

        weights = np.array([max(0.0, float(c.similarity)) for c in usable], dtype=np.float64)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(len(usable), dtype=np.float64)
        weights /= weights.sum()

        current = self._current_thresholds(current_state)
        proposal: Dict[str, float] = {}
        for key in self.SAFE_RANGES:
            vals, ws = [], []
            for weight, case in zip(weights, usable):
                action = case.episode.action or {}
                if key in action:
                    vals.append(float(action[key]))
                    ws.append(float(weight))
                elif key == "rho" and "delta_fused" in action:
                    vals.append(float(action["delta_fused"]))
                    ws.append(float(weight))
            if not vals:
                continue
            ws_arr = np.asarray(ws, dtype=np.float64)
            ws_arr /= ws_arr.sum()
            proposal[key] = float(np.dot(ws_arr, np.asarray(vals, dtype=np.float64)))

        _, adjusted, _ = self.review({"threshold_patch": proposal}, current_state)
        return adjusted

    def build_interval(self, patch: Dict, prior: Optional[ThresholdPriorInterval] = None) -> Dict:
        """Convert a validated patch into bounded search intervals for the edge."""
        base = {
            "accept_quantile_range": list(self.SAFE_RANGES["accept_quantile"]),
            "margin_quantile_range": list(self.SAFE_RANGES["margin_quantile"]),
            "delta_fused_range": list(self.SAFE_RANGES["rho"]),
            "delta_margin_range": [0.05, 0.35],  # documented compatibility field
        }

        if prior is not None:
            base.update({
                "accept_quantile_range": [prior.accept_quantile_min, prior.accept_quantile_max],
                "margin_quantile_range": [prior.margin_quantile_min, prior.margin_quantile_max],
                "delta_fused_range": [prior.delta_fused_min, prior.delta_fused_max],
                "delta_margin_range": [prior.delta_margin_min, prior.delta_margin_max],
            })

        mapping = {
            "accept_quantile": "accept_quantile_range",
            "margin_quantile": "margin_quantile_range",
            "rho": "delta_fused_range",
        }
        for key, value in (patch or {}).items():
            if key not in mapping:
                continue
            lo_safe, hi_safe = self.SAFE_RANGES[key]
            half = self.INTERVAL_HALF_WIDTH[key]
            lo = max(lo_safe, float(value) - half)
            hi = min(hi_safe, float(value) + half)
            if prior is not None:
                p_lo, p_hi = base[mapping[key]]
                inter_lo, inter_hi = max(lo, p_lo), min(hi, p_hi)
                if inter_lo <= inter_hi:
                    lo, hi = inter_lo, inter_hi
            base[mapping[key]] = [float(lo), float(hi)]
        return base


class RAGRetriever:
    """
    RAG 检索器 - 检索相关案例
    """
    
    def __init__(self, episode_store: EpisodeStore, objectives: Dict):
        self.store = episode_store
        self.objectives = objectives
    
    def retrieve_for_guidance(self, query_state: StateVector, 
                              k_success: int = 3, k_failure: int = 2) -> Dict:
        """
        为指导检索案例
        
        Returns:
            {
                "success_cases": [...],
                "failure_cases": [...],
                "statistics": {...}
            }
        """
        # 检索成功案例
        success_cases = self.store.retrieve(query_state, k=k_success, success_only=True)
        
        # 检索失败案例（用于避免重蹈覆辙）
        failure_cases = self.store.retrieve(query_state, k=k_failure, success_only=False)
        # 只保留真正失败的
        failure_cases = [c for c in failure_cases if not c.episode.success]
        
        return {
            "success_cases": success_cases,
            "failure_cases": failure_cases,
            "statistics": self.store.get_statistics(),
        }
    
    def format_for_llm(self, retrieved: Dict) -> str:
        """格式化检索结果供 LLM 使用"""
        lines = []
        
        stats = retrieved["statistics"]
        lines.append(f"### Cloud Experience Database")
        lines.append(f"Total episodes: {stats.get('total', 0)}, Success rate: {stats.get('success_rate', 0):.1%}")
        lines.append("")
        
        # 成功案例
        success_cases = retrieved["success_cases"]
        if success_cases:
            lines.append("### Similar SUCCESSFUL Cases (learn from these):")
            for i, case in enumerate(success_cases, 1):
                ep = case.episode
                lines.append(f"\n**Case {i}** (similarity: {case.similarity:.2f})")
                lines.append(f"  Before: closed={ep.state_before.closed_acc:.4f}, open={ep.state_before.open_auc:.4f}")
                lines.append(f"  Action: {ep.action}")
                lines.append(f"  After:  closed={ep.state_after.closed_acc:.4f}, open={ep.state_after.open_auc:.4f}")
                lines.append(f"  Improvement: {ep.improvement}")
                lines.append(f"  Relevance: {case.relevance_reason}")
        else:
            lines.append("### No similar successful cases found")
        
        lines.append("")
        
        # 失败案例
        failure_cases = retrieved["failure_cases"]
        if failure_cases:
            lines.append("### Similar FAILED Cases (AVOID these patterns):")
            for i, case in enumerate(failure_cases, 1):
                ep = case.episode
                lines.append(f"\n**Failed Case {i}** (similarity: {case.similarity:.2f})")
                lines.append(f"  Before: closed={ep.state_before.closed_acc:.4f}, open={ep.state_before.open_auc:.4f}")
                lines.append(f"  Action that FAILED: {ep.action}")
                lines.append(f"  Result: closed={ep.state_after.closed_acc:.4f}, open={ep.state_after.open_auc:.4f}")
                lines.append(f"  ⚠️ DO NOT repeat this pattern")
        
        return "\n".join(lines)



class CloudAgentV2:
    """云端 Agent V2 - RAG-based 轻量级指导"""
    
    def __init__(self, cfg: dict = None):
        # Merge user configuration over defaults so partial test/reproduction
        # configurations remain valid.
        self.config = self._default_config()
        self._deep_update(self.config, cfg or {})

        mode_str = self.config.get("assistance_mode", "auto")
        self.assistance_mode = AssistanceMode(mode_str)
        self.guidance_variant = self.config.get("guidance_variant", "rag")

        self.data_dir = Path(self.config.get("data_dir", "cloud_data"))
        self.uploads_dir = self.data_dir / "uploads"
        self.models_dir = self.data_dir / "models"
        for d in [self.data_dir, self.uploads_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.objectives = self.config["objectives"]

        self.episode_store = EpisodeStore(self.objectives)
        self.retriever = RAGRetriever(self.episode_store, self.objectives)
        self.policy_guard = PolicyGuard(self.objectives)

        # The LLM belongs to the cloud-side advisory path.  It is lazy-loaded only
        # when rag/llm_only guidance is actually requested.
        llm_cfg = self.config.get("llm", {})
        self.llm_enabled = bool(llm_cfg.get("enable", True))
        self.llm_advisor = None
        self.policy_audit = {
            "llm_requests": 0,
            "reviewed_proposals": 0,
            "directly_accepted": 0,
            "guard_adjusted": 0,
            "no_executable_update": 0,
            "deterministic_fallback": 0,
            "range_clamp_events": 0,
            "step_clamp_events": 0,
            "unsupported_or_invalid_events": 0,
        }

        self.threshold_priors: Dict[str, ThresholdPriorInterval] = {}
        self.model_packages: Dict[str, ModelPackage] = {}
        self.latest_model_package: Optional[ModelPackage] = None
        self.source_model_package: Optional[ModelPackage] = None
        self.edge_states: Dict[str, EdgeNodeState] = {}
        self.experience_bank: Dict[str, List[Dict]] = defaultdict(list)

        self.lock = threading.Lock()
        self._load_state()
        self._ensure_initial_model()

        print("[CloudAgentV2] Initialized (cloud-side RAG/LLM advisory)")
        print(f"[CloudAgentV2] Assistance mode: {self.assistance_mode.value}")
        print(f"[CloudAgentV2] Guidance variant: {self.guidance_variant}")
        print(f"[CloudAgentV2] LLM enabled: {self.llm_enabled}")
        print(f"[CloudAgentV2] Episode store: {len(self.episode_store.episodes)} episodes")
    
    def _default_config(self) -> dict:
        return {
            "data_dir": "cloud_data",
            "assistance_mode": "auto",
            "guidance_variant": "rag",  # rag | llm_only | deterministic
            "initial_model": {},
            "llm": {
                "enable": True,
                "model_path": "Qwen/Qwen2.5-3B-Instruct",
                "device_map": "auto",
                "dtype": "fp16",
                "quantization": "auto",
                "max_new_tokens": 400,
                "offline_mode": True,
            },
            "rag": {"k_success": 3, "k_failure": 2},
            "prior": {"min_samples": 3, "interval_expansion": 0.05},
            "mismatch": {
                "mild_threshold": 0.05,
                "moderate_threshold": 0.10,
                "severe_threshold": 0.20,
                "max_consecutive_failures": 3,
            },
            # Kept for backward-compatible configuration parsing only.  Model
            # refresh in the manuscript-aligned path always re-provisions the
            # original source-trained package and never aggregates target models.
            "aggregation": {"enabled": False, "method": "disabled", "min_contributors": 2},
            "objectives": {
                "min_closed_acc": 0.90,
                "target_open_auc": 0.85,
                "max_reject_rate": 0.60,  # auxiliary metric, not hard feasibility
            },
        }

    @staticmethod
    def _deep_update(base: Dict, update: Dict) -> Dict:
        for key, value in (update or {}).items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                CloudAgentV2._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _init_llm_advisor(self):
        if self.llm_advisor is not None or not self.llm_enabled:
            return
        try:
            from llm_advisor import LLMAdvisor
            self.llm_advisor = LLMAdvisor(self.config.get("llm", {}))
            print("[CloudAgentV2] Cloud-side LLM advisor initialized")
        except Exception as exc:
            print(f"[CloudAgentV2] LLM advisor initialization failed: {exc}")
            self.llm_advisor = None
    

    def receive_episode(self, data: Dict, weights_data: bytes = None) -> Dict:
        """Receive an operational episode from the edge.

        Target-domain model weights are intentionally not collected for the fixed-
        base model-refresh path.  This prevents accidental target-domain model
        aggregation and keeps reprovisioning source-only, as described in the
        manuscript.
        """
        with self.lock:
            terminal_id = data.get("terminal_id", "unknown")
            print(f"\n[CloudAgentV2] Received from {terminal_id}")
            print(f"  closed={data.get('closed_acc', 0):.4f}, open={data.get('open_auc', 0):.4f}")
            if weights_data:
                print("[CloudAgentV2] Ignoring uploaded weights: source-only model refresh is enabled")

            episode = self._create_episode_record(data, weights_file=None)
            self.episode_store.add(episode)
            self.experience_bank[terminal_id].append(data)
            self._update_edge_state(data)
            self._try_update_priors()
            self._save_state()
            assistance = self._determine_assistance(terminal_id)
            return {"status": "success", "assistance": assistance}
    
    def _create_episode_record(self, data: Dict, weights_file: str = None) -> EpisodeRecord:
        terminal_id = data.get("terminal_id", "unknown")
        episode_id = f"{terminal_id}_{data.get('episode_id', 0)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        state = StateVector.from_dict(data, self.objectives)

        # Historical action is stored in the same threshold-space used by the edge
        # control agent and the deterministic retrieval-only comparator.
        action = {
            "accept_quantile": float(data.get("accept_quantile", state.accept_quantile)),
            "margin_quantile": float(data.get("margin_quantile", state.margin_quantile)),
            "rho": float(data.get("delta_fused", data.get("rho", state.rho))),
        }

        improvement = {}
        prev_episodes = [ep for ep in self.episode_store.episodes if ep.terminal_id == terminal_id]
        if prev_episodes:
            prev = prev_episodes[-1]
            improvement = {
                "closed_acc_delta": state.closed_acc - prev.state_after.closed_acc,
                "open_auc_delta": state.open_auc - prev.state_after.open_auc,
                "reject_rate_delta": state.reject_rate - prev.state_after.reject_rate,
            }

        return EpisodeRecord(
            episode_id=episode_id,
            terminal_id=terminal_id,
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            state_before=state,
            action=action,
            state_after=state,
            success=bool(data.get("recovery_success", False)),
            improvement=improvement,
            model_config=data.get("model_config", {}),
            weights_file=None,
        )
    

    def _provide_lightweight(self, terminal_id: str, query_state: StateVector = None,
                             request_context: Dict = None) -> Dict:
        """Generate manuscript-aligned cloud-side lightweight guidance."""
        request_context = request_context or {}
        if query_state is None:
            terminal_episodes = [ep for ep in self.episode_store.episodes if ep.terminal_id == terminal_id]
            query_state = terminal_episodes[-1].state_after if terminal_episodes else StateVector()

        variant = self.guidance_variant
        if variant not in {"rag", "llm_only", "deterministic"}:
            variant = "rag"
        print(f"[CloudAgentV2] -> LIGHTWEIGHT guidance ({variant}) for {terminal_id}")

        rag_cfg = self.config.get("rag", {})
        k_success = int(rag_cfg.get("k_success", 3))
        k_failure = int(rag_cfg.get("k_failure", 2))

        if variant == "llm_only":
            retrieved = {"success_cases": [], "failure_cases": [], "statistics": self.episode_store.get_statistics()}
        else:
            retrieved = self.retriever.retrieve_for_guidance(
                query_state, k_success=k_success, k_failure=k_failure
            )

        success_cases = retrieved.get("success_cases", [])
        failure_cases = retrieved.get("failure_cases", [])
        rag_context = self.retriever.format_for_llm(retrieved) if variant == "rag" else None
        deterministic_patch = self.policy_guard.suggest_safe_adjustment(query_state, success_cases)
        prior = self._get_or_build_prior("default")

        diagnosis = {"gaps": {
            "closed_acc_gap": query_state.closed_gap,
            "open_auc_gap": query_state.open_gap,
        }}
        best_metrics = {
            "closed_acc": query_state.closed_acc,
            "open_auc": query_state.open_auc,
            "reject_rate": query_state.reject_rate,
        }
        current_thresholds = {
            "accept_quantile": query_state.accept_quantile,
            "margin_quantile": query_state.margin_quantile,
            "rho": query_state.rho,
        }

        final_patch: Dict[str, float] = {}
        source = "unavailable"
        approved = False
        warnings: List[str] = []
        analysis = ""
        reasoning = ""
        confidence = prior.confidence if prior else 0.0

        if variant == "deterministic":
            final_patch = deterministic_patch
            source = "deterministic_retrieval" if final_patch else "deterministic_unavailable"
            approved = bool(final_patch)
            analysis = "Non-generative weighted aggregation of retrieved successful cases"
            reasoning = "Top successful cases are combined using normalized nonnegative cosine-similarity weights"
            confidence = float(max([c.similarity for c in success_cases], default=0.0))
        else:
            self.policy_audit["llm_requests"] += 1
            self._init_llm_advisor()
            advice = None
            if self.llm_advisor is not None:
                advice = self.llm_advisor.get_threshold_guidance(
                    current_thresholds=current_thresholds,
                    best_metrics=best_metrics,
                    objectives=self.objectives,
                    diagnosis=diagnosis,
                    rag_context=rag_context if variant == "rag" else None,
                    policy_suggestion=deterministic_patch if variant == "rag" else None,
                )

            if advice is not None and advice.threshold_patch:
                self.policy_audit["reviewed_proposals"] += 1
                approved, final_patch, warnings = self.policy_guard.review(
                    {"threshold_patch": advice.threshold_patch}, query_state,
                    failure_cases if variant == "rag" else []
                )
                source = "rag_llm" if variant == "rag" else "llm_only"
                analysis = advice.analysis
                reasoning = advice.reasoning
                confidence = advice.confidence

                first_events = list(getattr(advice, "validation_events", []) or [])
                all_events = first_events + list(warnings or [])
                adjusted_any = bool(
                    getattr(advice, "validation_adjusted", False) or warnings or not approved
                )
                if adjusted_any:
                    self.policy_audit["guard_adjusted"] += 1
                else:
                    self.policy_audit["directly_accepted"] += 1
                self.policy_audit["range_clamp_events"] += sum(
                    1 for event in all_events
                    if "range" in str(event) or "projected" in str(event)
                )
                self.policy_audit["step_clamp_events"] += sum(
                    1 for event in all_events if "step" in str(event)
                )
                self.policy_audit["unsupported_or_invalid_events"] += sum(
                    1 for event in all_events
                    if "unsupported" in str(event) or "non_numeric" in str(event)
                    or "non-numeric" in str(event)
                )
            elif variant == "rag" and deterministic_patch:
                final_patch = deterministic_patch
                source = "deterministic_fallback"
                approved = True
                analysis = "LLM advisory unavailable; using deterministic retrieval-based fallback"
                reasoning = "Fallback derived only from retrieved successful cases"
                confidence = float(max([c.similarity for c in success_cases], default=0.0))
                self.policy_audit["no_executable_update"] += 1
                self.policy_audit["deterministic_fallback"] += 1
            else:
                self.policy_audit["no_executable_update"] += 1

        available = bool(final_patch)
        threshold_interval = self.policy_guard.build_interval(final_patch, prior) if available else {}
        global_summary = self._build_global_summary()

        return {
            "type": "lightweight",
            "available": available,
            "guidance_variant": variant,
            "guidance_source": source,
            "guidance_patch": final_patch,
            "threshold_interval": threshold_interval,
            "requires_reprocessing": False,
            "policy_approved": approved,
            "policy_warnings": warnings,
            "analysis": analysis,
            "reasoning": reasoning,
            "confidence": float(confidence),
            "retrieved_cases": {
                "success": [self._case_to_dict(c) for c in success_cases[:k_success]],
                "failure": [self._case_to_dict(c) for c in failure_cases[:k_failure]],
            },
            "global_summary": global_summary,
            "num_episodes": len(self.episode_store.episodes),
            "policy_audit": dict(self.policy_audit),
            "message": "Cloud-side bounded threshold guidance; no edge-side LLM inference or retraining",
        }
    
    def _case_to_dict(self, case: RetrievedCase) -> Dict:
        """将检索案例转换为字典"""
        ep = case.episode
        return {
            "episode_id": ep.episode_id,
            "similarity": case.similarity,
            "relevance": case.relevance_reason,
            "state_before": asdict(ep.state_before),
            "action": ep.action,
            "state_after": asdict(ep.state_after),
            "success": ep.success,
            "improvement": ep.improvement,
        }
    
    def _build_global_summary(self) -> Dict:
        stats = self.episode_store.get_statistics()
        summary = {
            "total_episodes": stats.get("total", 0),
            "success_rate": stats.get("success_rate", 0),
            "success_count": stats.get("success", 0),
            "failure_count": stats.get("failure", 0),
        }
        if self.episode_store.success_episodes:
            for key in ("accept_quantile", "margin_quantile", "rho"):
                values = [float(ep.action[key]) for ep in self.episode_store.success_episodes if key in ep.action]
                if values:
                    summary[f"successful_{key}"] = {
                        "min": float(min(values)),
                        "max": float(max(values)),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                    }
        return summary
    
    def get_rag_guidance(self, terminal_id: str, current_state: Dict) -> Dict:
        """Compatibility name for the cloud-side lightweight-guidance endpoint."""
        current_state = current_state or {}
        query_state = StateVector.from_dict(current_state, self.objectives)
        return self._provide_lightweight(
            terminal_id, query_state, request_context=current_state
        )

    # Preferred neutral name; /rag_guidance remains as a compatibility alias.
    def get_guidance(self, terminal_id: str, current_state: Dict) -> Dict:
        return self.get_rag_guidance(terminal_id, current_state)
    

    def _provide_model(self, terminal_id: str) -> Dict:
        """Return the original source-trained model package for reprovisioning."""
        print(f"[CloudAgentV2] -> SOURCE MODEL REPROVISIONING for {terminal_id}")
        pkg = self._get_or_create_model_package()
        if pkg is None:
            return {"type": "model_reprovision", "available": False,
                    "source_only": True, "message": "Source model package unavailable"}
        return {
            "type": "model_reprovision",
            "available": True,
            "source_only": True,
            "version": pkg.version,
            "domain_cluster": pkg.domain_cluster,
            "metadata": {
                "created_at": pkg.created_at,
                "avg_metrics": pkg.avg_metrics,
            },
            "message": "Original source-trained model reprovisioning; no target-domain retraining",
        }
    
    def _determine_assistance(self, terminal_id: str) -> Dict:
        if self.assistance_mode == AssistanceMode.MODEL_REPROVISION:
            return self._provide_model(terminal_id)
        if self.assistance_mode == AssistanceMode.LIGHTWEIGHT:
            return {"type": "lightweight", "available": True,
                    "message": "Run edge correction first, then request /guidance if infeasible"}

        # AUTO preserves the hierarchy: edge/lightweight recovery is primary;
        # source-model reprovisioning is reserved for persistent failure history.
        state = self.edge_states.get(terminal_id)
        if state is not None and state.needs_reprovisioning:
            return self._provide_model(terminal_id)
        return {"type": "lightweight", "available": True,
                "message": "Run edge correction first, then request /guidance if infeasible"}
    

    def _update_edge_state(self, data: Dict):
        """更新边缘状态"""
        tid = data.get("terminal_id", "unknown")
        obj = self.objectives
        mismatch_cfg = self.config["mismatch"]
        
        closed_acc = data.get("closed_acc", 0)
        open_auc = data.get("open_auc", 0)
        
        if tid not in self.edge_states:
            self.edge_states[tid] = EdgeNodeState(
                terminal_id=tid, last_seen=datetime.now().isoformat(), current_metrics={})
        
        state = self.edge_states[tid]
        state.last_seen = datetime.now().isoformat()
        state.current_metrics = {"closed_acc": closed_acc, "open_auc": open_auc}
        state.num_episodes += 1
        
        max_gap = max(obj["min_closed_acc"] - closed_acc, obj["target_open_auc"] - open_auc)
        
        if max_gap >= mismatch_cfg["severe_threshold"]:
            state.mismatch_severity = "severe"
        elif max_gap >= mismatch_cfg["moderate_threshold"]:
            state.mismatch_severity = "moderate"
        elif max_gap >= mismatch_cfg["mild_threshold"]:
            state.mismatch_severity = "mild"
        else:
            state.mismatch_severity = "none"
        
        if data.get("recovery_success"):
            state.consecutive_failures = 0
            state.needs_reprovisioning = False
        else:
            state.consecutive_failures += 1
            if state.consecutive_failures >= mismatch_cfg["max_consecutive_failures"]:
                state.needs_reprovisioning = True
    
    def _get_or_build_prior(self, domain: str) -> Optional[ThresholdPriorInterval]:
        if domain in self.threshold_priors:
            return self.threshold_priors[domain]
        return self._build_prior(domain)
    
    def _build_prior(self, domain: str) -> Optional[ThresholdPriorInterval]:
        """Build an empirical threshold prior from successful historical episodes."""
        min_samples = int(self.config["prior"].get("min_samples", 3))
        expansion = float(self.config["prior"].get("interval_expansion", 0.05))
        successful = self.episode_store.success_episodes
        if len(successful) < min_samples:
            return None

        delta_f = [float(ep.action.get("rho", ep.action.get("delta_fused", 0.0))) for ep in successful]
        accept_q = [float(ep.action.get("accept_quantile", 0.90)) for ep in successful]
        margin_q = [float(ep.action.get("margin_quantile", 0.20)) for ep in successful]

        prior = ThresholdPriorInterval(
            domain_cluster=domain,
            delta_fused_min=max(-0.15, float(np.percentile(delta_f, 10)) - expansion),
            delta_fused_max=min(0.35, float(np.percentile(delta_f, 90)) + expansion),
            delta_margin_min=0.05,
            delta_margin_max=0.35,
            accept_quantile_min=max(0.80, float(np.percentile(accept_q, 10))),
            accept_quantile_max=min(0.95, float(np.percentile(accept_q, 90))),
            margin_quantile_min=max(0.10, float(np.percentile(margin_q, 10))),
            margin_quantile_max=min(0.30, float(np.percentile(margin_q, 90))),
            confidence=min(1.0, len(successful) / 20.0),
            num_samples=len(successful),
            created_at=datetime.now().isoformat(),
        )
        self.threshold_priors[domain] = prior
        return prior
    
    def _try_update_priors(self):
        self._build_prior("default")
    

    def _ensure_initial_model(self):
        """Create/restore the immutable source-trained reprovisioning package."""
        # Prefer a previously serialized source package if it still exists.
        existing = self.model_packages.get("v0000_initial")
        if existing is not None:
            archive = self.models_dir / "model_package_v0000_initial.tar.gz"
            if archive.exists():
                existing.package_path = str(archive)
                self.source_model_package = existing
                self.latest_model_package = existing
                return

        initial_cfg = self.config.get("initial_model", {})
        weights_path = initial_cfg.get("weights_path", "")
        if not weights_path or not os.path.exists(weights_path):
            print(f"[CloudAgentV2] No source model at: {weights_path}")
            return

        print(f"[CloudAgentV2] Creating immutable source model package from: {weights_path}")
        try:
            initial_weights = self.models_dir / "initial_model.pth"
            shutil.copy2(weights_path, initial_weights)
            pkg = ModelPackage(
                version="v0000_initial",
                domain_cluster="source",
                created_at=datetime.now().isoformat(),
                weights_file="initial_model.pth",
                model_config=initial_cfg.get("model_config", {}),
                baseline_thresholds=initial_cfg.get("baseline_thresholds", [
                    {"accept_quantile": 0.90, "margin_quantile": 0.20, "rho": 0.0}
                ]),
                signal_augment=initial_cfg.get("signal_augment", {}),
                avg_metrics={"closed_acc": 0.0, "open_auc": 0.0},
                num_contributors=0,
            )
            pkg.package_path = str(self._create_archive(pkg))
            self.model_packages = {"v0000_initial": pkg}
            self.source_model_package = pkg
            self.latest_model_package = pkg
            self._save_state()
            print("[CloudAgentV2] Source model package created")
        except Exception as exc:
            print(f"[CloudAgentV2] Failed to create source model: {exc}")
    
    def _trigger_async_aggregation(self):
        """Deprecated: target-domain model aggregation is disabled."""
        print("[CloudAgentV2] Model aggregation disabled in source-only refresh protocol")
    
    def aggregate_models(self) -> Optional[ModelPackage]:
        """Disabled to prevent target-domain leakage in the fixed-base protocol."""
        print("[CloudAgentV2] aggregate_models() disabled: model refresh reuses the original source-trained package")
        return self.source_model_package
    
    def _create_archive(self, pkg: ModelPackage) -> Optional[Path]:
        """创建 tar.gz"""
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                
                src = self.models_dir / pkg.weights_file
                if src.exists():
                    shutil.copy2(src, tmp / "model_weights.pth")
                
                with open(tmp / "model_config.json", "w") as f:
                    json.dump(pkg.model_config, f)
                with open(tmp / "baseline_thresholds.json", "w") as f:
                    json.dump(pkg.baseline_thresholds, f)
                with open(tmp / "signal_augment.json", "w") as f:
                    json.dump(pkg.signal_augment, f)
                with open(tmp / "metadata.json", "w") as f:
                    json.dump({"version": pkg.version, "avg_metrics": pkg.avg_metrics}, f)
                
                pkg_path = self.models_dir / f"model_package_{pkg.version}.tar.gz"
                with tarfile.open(pkg_path, "w:gz") as tar:
                    for f in tmp.iterdir():
                        tar.add(f, arcname=f.name)
                
                shutil.copy2(pkg_path, self.models_dir / "model_package_latest.tar.gz")
                return pkg_path
        except Exception as e:
            print(f"[CloudAgentV2] Archive failed: {e}")
            return None
    
    def _get_or_create_model_package(self) -> Optional[ModelPackage]:
        if self.source_model_package is None:
            self._ensure_initial_model()
        return self.source_model_package
    

    def _save_state(self):
        state = {
            "experience_bank": dict(self.experience_bank),
            "episodes": [self._episode_to_dict(ep) for ep in self.episode_store.episodes],
            "threshold_priors": {k: asdict(v) for k, v in self.threshold_priors.items()},
            "model_packages": {k: asdict(v) for k, v in self.model_packages.items()},
            "edge_states": {k: asdict(v) for k, v in self.edge_states.items()},
            "latest_model_version": self.latest_model_package.version if self.latest_model_package else None,
            "policy_audit": self.policy_audit,
        }
        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def _episode_to_dict(self, ep: EpisodeRecord) -> Dict:
        return {
            "episode_id": ep.episode_id,
            "terminal_id": ep.terminal_id,
            "timestamp": ep.timestamp,
            "state_before": asdict(ep.state_before),
            "action": ep.action,
            "state_after": asdict(ep.state_after),
            "success": ep.success,
            "improvement": ep.improvement,
            "model_config": ep.model_config,
            "weights_file": ep.weights_file,
        }
    
    def _load_state(self):
        path = self.data_dir / "state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self.experience_bank = defaultdict(list, state.get("experience_bank", {}))
            saved_audit = state.get("policy_audit", {}) or {}
            for key in self.policy_audit:
                if key in saved_audit:
                    self.policy_audit[key] = int(saved_audit[key])

            for ep_dict in state.get("episodes", []):
                ep = EpisodeRecord(
                    episode_id=ep_dict["episode_id"],
                    terminal_id=ep_dict["terminal_id"],
                    timestamp=ep_dict["timestamp"],
                    state_before=StateVector.from_dict(ep_dict.get("state_before", {}), self.objectives),
                    action=ep_dict.get("action", {}),
                    state_after=StateVector.from_dict(ep_dict.get("state_after", {}), self.objectives),
                    success=bool(ep_dict.get("success", False)),
                    improvement=ep_dict.get("improvement", {}),
                    model_config=ep_dict.get("model_config", {}),
                    weights_file=None,
                )
                self.episode_store.add(ep)

            for key, value in state.get("threshold_priors", {}).items():
                try:
                    self.threshold_priors[key] = ThresholdPriorInterval(**value)
                except TypeError:
                    pass

            source = state.get("model_packages", {}).get("v0000_initial")
            if source:
                try:
                    pkg = ModelPackage(**source)
                    self.model_packages["v0000_initial"] = pkg
                    self.source_model_package = pkg
                    self.latest_model_package = pkg
                except TypeError:
                    pass

            for key, value in state.get("edge_states", {}).items():
                try:
                    self.edge_states[key] = EdgeNodeState(**value)
                except TypeError:
                    pass
        except Exception as exc:
            print(f"[CloudAgentV2] Load failed: {exc}")
    

    def get_assistance(self, terminal_id: str) -> Dict:
        return self._determine_assistance(terminal_id)
    
    def set_mode(self, mode: str):
        self.assistance_mode = AssistanceMode(mode)
    
    def get_model_path(self) -> Optional[str]:
        pkg = self._get_or_create_model_package()
        if pkg is None:
            return None
        if pkg.package_path and os.path.exists(pkg.package_path):
            return str(pkg.package_path)
        p = self.models_dir / "model_package_v0000_initial.tar.gz"
        return str(p) if p.exists() else None
    
    def get_stats(self) -> Dict:
        audit = dict(self.policy_audit)
        reviewed = audit.get("reviewed_proposals", 0)
        audit["direct_acceptance_rate"] = (
            audit.get("directly_accepted", 0) / reviewed if reviewed else 0.0
        )
        audit["guard_intervention_rate"] = (
            audit.get("guard_adjusted", 0) / reviewed if reviewed else 0.0
        )
        return {
            "mode": self.assistance_mode.value,
            "guidance_variant": self.guidance_variant,
            "llm_enabled": self.llm_enabled,
            "episodes": len(self.episode_store.episodes),
            "success_rate": self.episode_store.get_statistics().get("success_rate", 0),
            "source_model": self.source_model_package.version if self.source_model_package else None,
            "model_refresh_policy": "source_only_reprovisioning",
            "policy_audit": audit,
        }



def create_app(agent: CloudAgentV2):
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "mode": agent.assistance_mode.value,
            "guidance_variant": agent.guidance_variant,
        })

    @app.route("/episode", methods=["POST"])
    def episode():
        try:
            if request.content_type and "multipart/form-data" in request.content_type:
                data = json.loads(request.form.get("json", "{}"))
            else:
                data = request.json or {}
            # Older clients may still send a weight file. It is ignored by the
            # source-only refresh protocol in receive_episode().
            weights = request.files.get("weights_file")
            weights_data = weights.read() if weights else None
            return jsonify(agent.receive_episode(data, weights_data))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 400

    @app.route("/assistance/<tid>")
    def assistance(tid):
        return jsonify(agent.get_assistance(tid))

    def _guidance_impl():
        data = request.json or {}
        terminal_id = data.get("terminal_id", "unknown")
        current_state = data.get("current_state", data)
        return jsonify(agent.get_guidance(terminal_id, current_state))

    @app.route("/guidance", methods=["POST"])
    def guidance():
        try:
            return _guidance_impl()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 400

    @app.route("/rag_guidance", methods=["POST"])
    def rag_guidance():
        # Backward-compatible alias.
        try:
            return _guidance_impl()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/model/download")
    def download():
        path = agent.get_model_path()
        if path and os.path.exists(path):
            return send_file(path, as_attachment=True)
        return jsonify({"error": "source model package unavailable"}), 404

    @app.route("/aggregate", methods=["POST"])
    def aggregate():
        return jsonify({
            "status": "disabled",
            "reason": "fixed-base model refresh re-provisions the original source-trained package",
        }), 409

    @app.route("/mode/<mode>", methods=["POST"])
    def set_mode(mode):
        try:
            agent.set_mode(mode)
            return jsonify({"status": "ok", "mode": mode})
        except Exception:
            return jsonify({"error": "invalid mode"}), 400

    @app.route("/stats")
    def stats():
        return jsonify(agent.get_stats())

    return app



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cloud coordination agent")
    parser.add_argument("--config", default="cloud_config.yaml")
    parser.add_argument("--mode", choices=["lightweight", "model_reprovision", "auto"], default=None)
    parser.add_argument("--guidance-variant", choices=["rag", "llm_only", "deterministic"], default=None)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    if args.mode:
        cfg["assistance_mode"] = args.mode
    if args.guidance_variant:
        cfg["guidance_variant"] = args.guidance_variant

    agent = CloudAgentV2(cfg)
    if HAS_FLASK:
        app = create_app(agent)
        print(f"\n[CloudAgentV2] Starting on port {args.port}")
        print(f"[CloudAgentV2] Mode: {agent.assistance_mode.value}")
        print(f"[CloudAgentV2] Guidance variant: {agent.guidance_variant}")
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    else:
        print("[CloudAgentV2] Flask not installed")


if __name__ == "__main__":
    main()

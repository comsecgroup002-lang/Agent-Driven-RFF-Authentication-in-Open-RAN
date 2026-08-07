# cloud_agent_v2.py
# -*- coding: utf-8 -*-
"""
云端 Agent V2 - RAG-based 轻量级指导系统

核心架构（RAG 风格）：
1. Episode Store: 每个历史 episode 存为结构化向量记录
2. Retriever: 根据当前状态检索 top-k 相似案例
3. LLM Generator: 结合 summary + retrieved cases 生成建议
4. Policy Guard: 对 LLM 建议做二次审查
5. 返回安全的调参建议给边缘端

工作流程：
Edge 上传当前状态 → 云端检索相似案例 → LLM 生成建议 → Policy Guard 审查 → 返回

协助模式：
- lightweight: RAG-based 轻量级指导（阈值区间 + 检索案例 + LLM建议）
- model_reprovision: 完整模型包 M_j
- auto: 根据状态自动选择
"""

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


# ============================================================================
# 协助级别枚举
# ============================================================================

class AssistanceMode(Enum):
    """云端协助模式"""
    LIGHTWEIGHT = "lightweight"
    MODEL_REPROVISION = "model_reprovision"
    AUTO = "auto"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StateVector:
    """
    状态向量 - 用于相似度检索
    包含：性能指标、参数配置、分布特征
    """
    # 性能指标
    closed_acc: float = 0.0
    open_auc: float = 0.0
    reject_rate: float = 0.0
    
    # 性能差距（相对于目标）
    closed_gap: float = 0.0
    open_gap: float = 0.0
    reject_excess: float = 0.0
    
    # 关键参数
    retain_ratio: float = 0.5
    engineering_aug_prob: float = 0.7
    
    # 分布特征
    fused_mean: float = 0.0
    fused_std: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """转换为数值向量（用于相似度计算）"""
        return np.array([
            self.closed_acc,
            self.open_auc,
            self.reject_rate,
            self.closed_gap,
            self.open_gap,
            self.reject_excess,
            self.retain_ratio,
            self.engineering_aug_prob,
            self.fused_mean,
            self.fused_std,
        ], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, d: Dict, objectives: Dict = None) -> 'StateVector':
        """从字典创建"""
        obj = objectives or {"min_closed_acc": 0.9, "target_open_auc": 0.85, "max_reject_rate": 0.6}
        
        closed_acc = d.get("closed_acc", d.get("metrics", {}).get("closed_acc", 0))
        open_auc = d.get("open_auc", d.get("metrics", {}).get("open_auc", 0))
        reject_rate = d.get("reject_rate", d.get("metrics", {}).get("reject_rate", 0))
        
        sig_aug = d.get("signal_augment", d.get("domain_state", {}))
        
        return cls(
            closed_acc=closed_acc,
            open_auc=open_auc,
            reject_rate=reject_rate,
            closed_gap=max(0, obj["min_closed_acc"] - closed_acc),
            open_gap=max(0, obj["target_open_auc"] - open_auc),
            reject_excess=max(0, reject_rate - obj["max_reject_rate"]),
            retain_ratio=sig_aug.get("retain_ratio", 0.5),
            engineering_aug_prob=sig_aug.get("engineering_aug_prob", 0.7),
            fused_mean=d.get("fused_mean", 0),
            fused_std=d.get("fused_std", 1),
        )


@dataclass
class EpisodeRecord:
    """
    Episode 记录 - RAG 的"文档"
    每条记录包含：状态、动作、结果、是否成功
    """
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
    """检索到的案例"""
    episode: EpisodeRecord
    similarity: float
    relevance_reason: str


@dataclass
class RAGGuidance:
    """RAG 生成的指导"""
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
    """阈值先验区间"""
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
    """模型包"""
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
    """边缘节点状态"""
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
    """
    Episode 存储 - 类似向量数据库
    支持：存储、检索、相似度计算
    """
    
    def __init__(self, objectives: Dict):
        self.objectives = objectives
        self.episodes: List[EpisodeRecord] = []
        self.success_episodes: List[EpisodeRecord] = []
        self.failure_episodes: List[EpisodeRecord] = []
        
        # 用于快速检索的索引
        self._state_vectors: Optional[np.ndarray] = None
        self._needs_reindex = True
    
    def add(self, episode: EpisodeRecord):
        """添加 episode"""
        self.episodes.append(episode)
        if episode.success:
            self.success_episodes.append(episode)
        else:
            self.failure_episodes.append(episode)
        self._needs_reindex = True
    
    def _build_index(self):
        """构建向量索引"""
        if not self._needs_reindex or not self.episodes:
            return
        
        vectors = []
        for ep in self.episodes:
            vectors.append(ep.state_before.to_vector())
        
        self._state_vectors = np.array(vectors)
        self._needs_reindex = False
    
    def retrieve(self, query_state: StateVector, k: int = 5, 
                 success_only: bool = None) -> List[RetrievedCase]:
        """
        检索相似案例
        
        Args:
            query_state: 查询状态向量
            k: 返回 top-k 个结果
            success_only: None=全部, True=只成功, False=只失败
        """
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
        """计算余弦相似度"""
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _explain_relevance(self, query: StateVector, ep: EpisodeRecord) -> str:
        """解释相关性"""
        reasons = []
        
        # 性能相似
        if abs(query.closed_acc - ep.state_before.closed_acc) < 0.05:
            reasons.append("similar closed_acc")
        if abs(query.open_auc - ep.state_before.open_auc) < 0.05:
            reasons.append("similar open_auc")
        
        # 差距相似
        if abs(query.closed_gap - ep.state_before.closed_gap) < 0.03:
            reasons.append("similar closed_gap")
        if abs(query.open_gap - ep.state_before.open_gap) < 0.03:
            reasons.append("similar open_gap")
        
        # 参数相似
        if abs(query.retain_ratio - ep.state_before.retain_ratio) < 0.1:
            reasons.append("similar retain_ratio")
        
        return "; ".join(reasons) if reasons else "general similarity"
    
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


# ============================================================================
# Policy Guard - 建议审查
# ============================================================================

class PolicyGuard:
    """
    Policy Guard - 对 LLM 建议进行二次审查
    确保建议安全、合理、不极端
    """
    
    # 安全范围
    SAFE_RANGES = {
        "retain_ratio": (0.35, 0.65),
        "engineering_aug_prob": (0.5, 0.85),
        "channel_aug_prob": (0.7, 1.0),
    }
    
    # 最大单步变化
    MAX_STEP = {
        "retain_ratio": 0.08,
        "engineering_aug_prob": 0.1,
    }
    
    def __init__(self, objectives: Dict):
        self.objectives = objectives
    
    def review(self, suggestion: Dict, current_state: StateVector,
               retrieved_cases: List[RetrievedCase]) -> Tuple[bool, Dict, List[str]]:
        """
        审查建议
        
        Returns:
            (approved, adjusted_suggestion, warnings)
        """
        warnings = []
        adjusted = {}
        approved = True
        
        param_changes = suggestion.get("param_changes", {})
        
        for key, new_value in param_changes.items():
            # 检查是否在安全范围内
            if key in self.SAFE_RANGES:
                min_val, max_val = self.SAFE_RANGES[key]
                
                if new_value < min_val:
                    warnings.append(f"{key}={new_value:.3f} below safe range, clamped to {min_val}")
                    new_value = min_val
                    approved = False
                elif new_value > max_val:
                    warnings.append(f"{key}={new_value:.3f} above safe range, clamped to {max_val}")
                    new_value = max_val
                    approved = False
            
            # 检查单步变化是否过大
            if key in self.MAX_STEP:
                current_value = getattr(current_state, key, 0.5)
                max_step = self.MAX_STEP[key]
                delta = new_value - current_value
                
                if abs(delta) > max_step:
                    clamped = current_value + (max_step if delta > 0 else -max_step)
                    warnings.append(f"{key} step too large ({delta:.3f}), clamped to {clamped:.3f}")
                    new_value = clamped
                    approved = False
            
            adjusted[key] = new_value
        
        # 检查是否与历史失败案例相似
        if retrieved_cases:
            for case in retrieved_cases:
                if not case.episode.success:
                    # 检查是否在向失败方向调整
                    failed_action = case.episode.action
                    for key, value in adjusted.items():
                        if key in failed_action:
                            failed_value = failed_action[key]
                            current_value = getattr(current_state, key, 0.5)
                            
                            # 如果调整方向与失败案例相同，警告
                            if (value > current_value and failed_value > current_value) or \
                               (value < current_value and failed_value < current_value):
                                if abs(value - failed_value) < 0.05:
                                    warnings.append(f"Similar to failed case: {key}={value:.3f}")
        
        return approved, adjusted, warnings
    
    def suggest_safe_adjustment(self, current_state: StateVector,
                                retrieved_success: List[RetrievedCase]) -> Dict:
        """
        基于成功案例建议安全调整
        """
        if not retrieved_success:
            # 没有成功案例，返回保守建议
            return self._conservative_suggestion(current_state)
        
        # 从成功案例中提取参数建议
        success_retain_ratios = []
        success_eng_probs = []
        
        for case in retrieved_success:
            action = case.episode.action
            if "retain_ratio" in action:
                success_retain_ratios.append(action["retain_ratio"])
            if "engineering_aug_prob" in action:
                success_eng_probs.append(action["engineering_aug_prob"])
        
        suggestion = {}
        
        # 使用成功案例的中位数作为建议
        if success_retain_ratios:
            target = float(np.median(success_retain_ratios))
            current = current_state.retain_ratio
            # 向目标方向小步调整
            step = min(0.05, abs(target - current))
            if target > current:
                suggestion["retain_ratio"] = current + step
            else:
                suggestion["retain_ratio"] = current - step
        
        return suggestion
    
    def _conservative_suggestion(self, current_state: StateVector) -> Dict:
        """保守建议：向中间值靠近"""
        suggestion = {}
        
        # retain_ratio 向 0.5 靠近
        current = current_state.retain_ratio
        if current < 0.5:
            suggestion["retain_ratio"] = min(0.5, current + 0.03)
        elif current > 0.5:
            suggestion["retain_ratio"] = max(0.5, current - 0.03)
        
        return suggestion


# ============================================================================
# RAG Retriever - 检索器
# ============================================================================

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


# ============================================================================
# 云端 Agent V2 - RAG-based
# ============================================================================

class CloudAgentV2:
    """云端 Agent V2 - RAG-based 轻量级指导"""
    
    def __init__(self, cfg: dict = None):
        self.config = cfg or self._default_config()
        
        # 协助模式
        mode_str = self.config.get("assistance_mode", "auto")
        self.assistance_mode = AssistanceMode(mode_str)
        
        # 目录
        self.data_dir = Path(self.config.get("data_dir", "cloud_data"))
        self.uploads_dir = self.data_dir / "uploads"
        self.models_dir = self.data_dir / "models"
        
        for d in [self.data_dir, self.uploads_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.objectives = self.config.get("objectives", {
            "min_closed_acc": 0.9, "target_open_auc": 0.85, "max_reject_rate": 0.6
        })
        
        # RAG 组件
        self.episode_store = EpisodeStore(self.objectives)
        self.retriever = RAGRetriever(self.episode_store, self.objectives)
        self.policy_guard = PolicyGuard(self.objectives)
        
        # 其他存储
        self.threshold_priors: Dict[str, ThresholdPriorInterval] = {}
        self.model_packages: Dict[str, ModelPackage] = {}
        self.latest_model_package: Optional[ModelPackage] = None
        self.edge_states: Dict[str, EdgeNodeState] = {}
        
        # 旧格式兼容
        self.experience_bank: Dict[str, List[Dict]] = defaultdict(list)
        
        self.lock = threading.Lock()
        self._load_state()
        
        # 初始化初始模型包
        self._ensure_initial_model()
        
        print(f"[CloudAgentV2] Initialized (RAG-based)")
        print(f"[CloudAgentV2] ★ Assistance mode: {self.assistance_mode.value}")
        print(f"[CloudAgentV2] ★ Episode store: {len(self.episode_store.episodes)} episodes")
    
    def _default_config(self) -> dict:
        return {
            "data_dir": "cloud_data",
            "assistance_mode": "auto",
            "initial_model": {},
            "rag": {
                "k_success": 3,
                "k_failure": 2,
            },
            "prior": {"min_samples": 3, "interval_expansion": 0.05},
            "mismatch": {"mild_threshold": 0.05, "moderate_threshold": 0.10, 
                        "severe_threshold": 0.20, "max_consecutive_failures": 3},
            "aggregation": {"method": "weighted_avg", "min_contributors": 2},
            "objectives": {"min_closed_acc": 0.9, "target_open_auc": 0.85, 
                          "max_reject_rate": 0.6},
        }
    
    # ========================================================================
    # Episode 处理
    # ========================================================================
    
    def receive_episode(self, data: Dict, weights_data: bytes = None) -> Dict:
        """接收边缘适应记录"""
        with self.lock:
            terminal_id = data.get("terminal_id", "unknown")
            
            print(f"\n[CloudAgentV2] Received from {terminal_id}")
            print(f"  closed={data.get('closed_acc', 0):.4f}, open={data.get('open_auc', 0):.4f}")
            
            # 保存权重
            weights_file = None
            if weights_data and data.get("recovery_success"):
                filename = f"{terminal_id}_ep{data.get('episode_id', 0)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
                with open(self.uploads_dir / filename, "wb") as f:
                    f.write(weights_data)
                weights_file = filename
                print(f"  Weights saved: {filename}")
            
            # 创建 Episode 记录
            episode = self._create_episode_record(data, weights_file)
            
            # 添加到 Episode Store
            self.episode_store.add(episode)
            
            # 兼容旧格式
            self.experience_bank[terminal_id].append(data)
            
            # 更新边缘状态
            self._update_edge_state(data)
            
            # 更新先验
            self._try_update_priors()
            
            # 保存状态
            self._save_state()
            
            # 返回协助信息
            assistance = self._determine_assistance(terminal_id)
            
            # 异步聚合
            if self.assistance_mode == AssistanceMode.MODEL_REPROVISION and data.get("recovery_success"):
                self._trigger_async_aggregation()
            
            return {"status": "success", "assistance": assistance}
    
    def _create_episode_record(self, data: Dict, weights_file: str = None) -> EpisodeRecord:
        """创建 Episode 记录"""
        terminal_id = data.get("terminal_id", "unknown")
        episode_id = f"{terminal_id}_{data.get('episode_id', 0)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 状态向量
        state = StateVector.from_dict(data, self.objectives)
        
        # 动作（参数变化）
        sig_aug = data.get("signal_augment", data.get("domain_state", {}))
        action = {
            "retain_ratio": sig_aug.get("retain_ratio", 0.5),
            "engineering_aug_prob": sig_aug.get("engineering_aug_prob", 0.7),
        }
        
        # 改善程度（如果有历史对比）
        improvement = {}
        prev_episodes = [ep for ep in self.episode_store.episodes 
                        if ep.terminal_id == terminal_id]
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
            state_before=state,  # 当前状态作为"之前"状态
            action=action,
            state_after=state,   # 下次上报时更新
            success=data.get("recovery_success", False),
            improvement=improvement,
            model_config=data.get("model_config", {}),
            weights_file=weights_file,
        )
    
    # ========================================================================
    # RAG-based 轻量级指导
    # ========================================================================
    
    def _provide_lightweight(self, terminal_id: str, query_state: StateVector = None) -> Dict:
        """
        RAG-based 轻量级指导
        
        流程：
        1. 检索相似案例
        2. 构建全局摘要
        3. 格式化为 LLM 输入
        4. Policy Guard 提供安全建议
        """
        print(f"[CloudAgentV2] → RAG-based LIGHTWEIGHT guidance for {terminal_id}")
        
        # 获取或构建查询状态
        if query_state is None:
            # 使用该终端最近的状态
            terminal_episodes = [ep for ep in self.episode_store.episodes 
                                if ep.terminal_id == terminal_id]
            if terminal_episodes:
                query_state = terminal_episodes[-1].state_after
            else:
                query_state = StateVector()
        
        # RAG 配置
        rag_cfg = self.config.get("rag", {})
        k_success = rag_cfg.get("k_success", 3)
        k_failure = rag_cfg.get("k_failure", 2)
        
        # 检索相似案例
        retrieved = self.retriever.retrieve_for_guidance(
            query_state, k_success=k_success, k_failure=k_failure
        )
        
        # 格式化为 LLM 输入
        rag_context = self.retriever.format_for_llm(retrieved)
        
        # Policy Guard 建议
        success_cases = retrieved.get("success_cases", [])
        safe_suggestion = self.policy_guard.suggest_safe_adjustment(query_state, success_cases)
        
        # 阈值先验
        prior = self._get_or_build_prior("default")
        threshold_interval = {
            "delta_fused_range": [-0.15, 0.35],
            "delta_margin_range": [0.05, 0.35],
            "accept_quantile_range": [0.80, 0.95],
            "margin_quantile_range": [0.10, 0.30],
        }
        if prior:
            threshold_interval = {
                "delta_fused_range": [prior.delta_fused_min, prior.delta_fused_max],
                "delta_margin_range": [prior.delta_margin_min, prior.delta_margin_max],
                "accept_quantile_range": [prior.accept_quantile_min, prior.accept_quantile_max],
                "margin_quantile_range": [prior.margin_quantile_min, prior.margin_quantile_max],
            }
        
        # 全局摘要
        global_summary = self._build_global_summary()
        
        return {
            "type": "lightweight",
            "threshold_interval": threshold_interval,
            
            # RAG 组件
            "rag_context": rag_context,  # 格式化的检索结果（给 LLM）
            "retrieved_cases": {
                "success": [self._case_to_dict(c) for c in success_cases[:k_success]],
                "failure": [self._case_to_dict(c) for c in retrieved.get("failure_cases", [])[:k_failure]],
            },
            
            # Policy Guard 建议
            "policy_suggestion": safe_suggestion,
            "safe_param_ranges": PolicyGuard.SAFE_RANGES,
            "max_step_sizes": PolicyGuard.MAX_STEP,
            
            # 全局摘要
            "global_summary": global_summary,
            
            "confidence": prior.confidence if prior else 0.5,
            "num_episodes": len(self.episode_store.episodes),
            "message": "RAG-based lightweight guidance with retrieved cases",
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
        """构建全局摘要"""
        stats = self.episode_store.get_statistics()
        
        summary = {
            "total_episodes": stats.get("total", 0),
            "success_rate": stats.get("success_rate", 0),
            "success_count": stats.get("success", 0),
            "failure_count": stats.get("failure", 0),
        }
        
        # 成功案例的参数统计
        if self.episode_store.success_episodes:
            retain_ratios = [ep.action.get("retain_ratio", 0.5) 
                           for ep in self.episode_store.success_episodes]
            summary["successful_retain_ratio"] = {
                "min": float(min(retain_ratios)),
                "max": float(max(retain_ratios)),
                "mean": float(np.mean(retain_ratios)),
                "median": float(np.median(retain_ratios)),
            }
        
        return summary
    
    def get_rag_guidance(self, terminal_id: str, current_state: Dict) -> Dict:
        """
        获取 RAG 指导（供边缘端调用）
        
        Args:
            terminal_id: 终端 ID
            current_state: 当前状态字典
        
        Returns:
            RAG 指导信息
        """
        query_state = StateVector.from_dict(current_state, self.objectives)
        return self._provide_lightweight(terminal_id, query_state)
    
    # ========================================================================
    # 模型重配置
    # ========================================================================
    
    def _provide_model(self, terminal_id: str) -> Dict:
        """模型重配置"""
        print(f"[CloudAgentV2] → MODEL REPROVISIONING for {terminal_id}")
        
        pkg = self._get_or_create_model_package()
        
        if pkg is None:
            return {"type": "model_reprovision", "available": False, 
                    "message": "No model available"}
        
        return {
            "type": "model_reprovision",
            "available": True,
            "model_package": {
                "version": pkg.version,
                "download_endpoint": "/model/download",
                "model_config": pkg.model_config,
                "baseline_thresholds": pkg.baseline_thresholds,
                "signal_augment": pkg.signal_augment,
                "avg_metrics": pkg.avg_metrics,
            },
            "message": "Model reprovisioning",
        }
    
    def _determine_assistance(self, terminal_id: str) -> Dict:
        """决定协助类型"""
        if self.assistance_mode == AssistanceMode.LIGHTWEIGHT:
            return self._provide_lightweight(terminal_id)
        elif self.assistance_mode == AssistanceMode.MODEL_REPROVISION:
            return self._provide_model(terminal_id)
        else:
            state = self.edge_states.get(terminal_id)
            if state is None or state.needs_reprovisioning or state.mismatch_severity == "severe":
                return self._provide_model(terminal_id)
            return self._provide_lightweight(terminal_id)
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
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
        """构建先验区间"""
        min_samples = self.config["prior"]["min_samples"]
        expansion = self.config["prior"]["interval_expansion"]
        
        if len(self.episode_store.episodes) < min_samples:
            return None
        
        # 使用所有记录
        all_records = []
        for tid, records in self.experience_bank.items():
            all_records.extend(records)
        
        if not all_records:
            return None
        
        delta_f = [r.get("delta_fused", r.get("threshold_config", {}).get("rho", 0)) for r in all_records]
        accept_q = [r.get("accept_quantile", r.get("threshold_config", {}).get("accept_quantile", 0.9)) for r in all_records]
        margin_q = [r.get("margin_quantile", r.get("threshold_config", {}).get("margin_quantile", 0.2)) for r in all_records]
        
        prior = ThresholdPriorInterval(
            domain_cluster=domain,
            delta_fused_min=float(np.percentile(delta_f, 10)) - expansion,
            delta_fused_max=float(np.percentile(delta_f, 90)) + expansion,
            delta_margin_min=0.05,
            delta_margin_max=0.35,
            accept_quantile_min=float(np.percentile(accept_q, 10)),
            accept_quantile_max=float(np.percentile(accept_q, 90)),
            margin_quantile_min=float(np.percentile(margin_q, 10)),
            margin_quantile_max=float(np.percentile(margin_q, 90)),
            confidence=min(1.0, len(all_records) / 20),
            num_samples=len(all_records),
            created_at=datetime.now().isoformat(),
        )
        
        self.threshold_priors[domain] = prior
        return prior
    
    def _try_update_priors(self):
        self._build_prior("default")
    
    # ========================================================================
    # 初始模型
    # ========================================================================
    
    def _ensure_initial_model(self):
        """确保有初始模型包"""
        if self.latest_model_package is not None:
            return
        
        initial_cfg = self.config.get("initial_model", {})
        weights_path = initial_cfg.get("weights_path", "")
        
        if not weights_path or not os.path.exists(weights_path):
            print(f"[CloudAgentV2] ⚠️ No initial model at: {weights_path}")
            return
        
        print(f"[CloudAgentV2] Creating initial model package from: {weights_path}")
        
        try:
            initial_weights = self.models_dir / "initial_model.pth"
            shutil.copy2(weights_path, initial_weights)
            
            pkg = ModelPackage(
                version="v0000_initial",
                domain_cluster="default",
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
            
            self.model_packages["v0000_initial"] = pkg
            self.latest_model_package = pkg
            self._save_state()
            
            print(f"[CloudAgentV2] ✅ Initial model package created")
            
        except Exception as e:
            print(f"[CloudAgentV2] ❌ Failed to create initial model: {e}")
    
    def _trigger_async_aggregation(self):
        def async_aggregate():
            try:
                self.aggregate_models()
            except Exception as e:
                print(f"[CloudAgentV2] Aggregation failed: {e}")
        
        thread = threading.Thread(target=async_aggregate, daemon=True)
        thread.start()
    
    def aggregate_models(self) -> Optional[ModelPackage]:
        """聚合模型"""
        with self.lock:
            agg = self.config["aggregation"]
            
            available = []
            for ep in self.episode_store.success_episodes:
                if ep.weights_file:
                    path = self.uploads_dir / ep.weights_file
                    if path.exists():
                        available.append({"episode": ep, "path": str(path)})
            
            if len(available) < agg.get("min_contributors", 2):
                return None
            
            print(f"[CloudAgentV2] Aggregating {len(available)} models...")
            
            import torch
            
            perfs = [m["episode"].state_after.open_auc for m in available]
            total = sum(perfs) if sum(perfs) > 0 else 1
            weights = [p / total for p in perfs]
            
            all_sd = [torch.load(m["path"], map_location="cpu") for m in available]
            avg_sd = {}
            for key in all_sd[0]:
                tensors = [sd[key].float() * w for sd, w in zip(all_sd, weights)]
                avg_sd[key] = sum(tensors)
            
            version = f"v{len(self.model_packages) + 1:04d}"
            filename = f"global_{version}.pth"
            torch.save(avg_sd, self.models_dir / filename)
            
            # 聚合参数
            signal_augment = {}
            retain_ratios = [m["episode"].action.get("retain_ratio", 0.5) for m in available]
            if retain_ratios:
                signal_augment["retain_ratio"] = float(np.mean(retain_ratios))
            
            pkg = ModelPackage(
                version=version,
                domain_cluster="default",
                created_at=datetime.now().isoformat(),
                weights_file=filename,
                model_config=available[0]["episode"].model_config,
                baseline_thresholds=[{"accept_quantile": 0.90, "margin_quantile": 0.20, "rho": 0.0}],
                signal_augment=signal_augment,
                avg_metrics={
                    "closed_acc": float(np.mean([m["episode"].state_after.closed_acc for m in available])),
                    "open_auc": float(np.mean([m["episode"].state_after.open_auc for m in available])),
                },
                num_contributors=len(available),
            )
            
            pkg.package_path = str(self._create_archive(pkg))
            
            self.model_packages[version] = pkg
            self.latest_model_package = pkg
            self._save_state()
            
            print(f"[CloudAgentV2] ✅ Aggregated: {version}")
            return pkg
    
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
        if self.latest_model_package:
            return self.latest_model_package
        return self.aggregate_models()
    
    # ========================================================================
    # 状态管理
    # ========================================================================
    
    def _save_state(self):
        state = {
            "experience_bank": dict(self.experience_bank),
            "episodes": [self._episode_to_dict(ep) for ep in self.episode_store.episodes],
            "threshold_priors": {k: asdict(v) for k, v in self.threshold_priors.items()},
            "model_packages": {k: asdict(v) for k, v in self.model_packages.items()},
            "edge_states": {k: asdict(v) for k, v in self.edge_states.items()},
            "latest_model_version": self.latest_model_package.version if self.latest_model_package else None,
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
            
            # 恢复 episodes
            for ep_dict in state.get("episodes", []):
                ep = EpisodeRecord(
                    episode_id=ep_dict["episode_id"],
                    terminal_id=ep_dict["terminal_id"],
                    timestamp=ep_dict["timestamp"],
                    state_before=StateVector(**ep_dict["state_before"]),
                    action=ep_dict["action"],
                    state_after=StateVector(**ep_dict["state_after"]),
                    success=ep_dict["success"],
                    improvement=ep_dict.get("improvement", {}),
                    model_config=ep_dict.get("model_config", {}),
                    weights_file=ep_dict.get("weights_file"),
                )
                self.episode_store.add(ep)
            
            for k, v in state.get("threshold_priors", {}).items():
                self.threshold_priors[k] = ThresholdPriorInterval(**v)
            
            for k, v in state.get("model_packages", {}).items():
                self.model_packages[k] = ModelPackage(**v)
            
            for k, v in state.get("edge_states", {}).items():
                self.edge_states[k] = EdgeNodeState(**v)
            
            latest_ver = state.get("latest_model_version")
            if latest_ver and latest_ver in self.model_packages:
                self.latest_model_package = self.model_packages[latest_ver]
                
        except Exception as e:
            print(f"[CloudAgentV2] Load failed: {e}")
    
    # ========================================================================
    # API
    # ========================================================================
    
    def get_assistance(self, terminal_id: str) -> Dict:
        return self._determine_assistance(terminal_id)
    
    def set_mode(self, mode: str):
        self.assistance_mode = AssistanceMode(mode)
    
    def get_model_path(self) -> Optional[str]:
        p = self.models_dir / "model_package_latest.tar.gz"
        return str(p) if p.exists() else None
    
    def get_stats(self) -> Dict:
        return {
            "mode": self.assistance_mode.value,
            "episodes": len(self.episode_store.episodes),
            "success_rate": self.episode_store.get_statistics().get("success_rate", 0),
            "latest_model": self.latest_model_package.version if self.latest_model_package else None,
        }


# ============================================================================
# Flask API
# ============================================================================

def create_app(agent: CloudAgentV2) -> Flask:
    app = Flask(__name__)
    
    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "mode": agent.assistance_mode.value})
    
    @app.route("/episode", methods=["POST"])
    def episode():
        try:
            if request.content_type and 'multipart/form-data' in request.content_type:
                data = json.loads(request.form.get("json", "{}"))
            else:
                data = request.json or {}
            
            weights = request.files.get("weights_file")
            weights_data = weights.read() if weights else None
            
            return jsonify(agent.receive_episode(data, weights_data))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/assistance/<tid>")
    def assistance(tid):
        return jsonify(agent.get_assistance(tid))
    
    @app.route("/rag_guidance", methods=["POST"])
    def rag_guidance():
        """RAG 指导接口"""
        try:
            data = request.json or {}
            terminal_id = data.get("terminal_id", "unknown")
            current_state = data.get("current_state", {})
            return jsonify(agent.get_rag_guidance(terminal_id, current_state))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/model/download")
    def download():
        path = agent.get_model_path()
        if path and os.path.exists(path):
            return send_file(path, as_attachment=True)
        return jsonify({"error": "no model"}), 404
    
    @app.route("/aggregate", methods=["POST"])
    def aggregate():
        pkg = agent.aggregate_models()
        if pkg:
            return jsonify({"status": "success", "version": pkg.version})
        return jsonify({"status": "insufficient"})
    
    @app.route("/mode/<mode>", methods=["POST"])
    def set_mode(mode):
        try:
            agent.set_mode(mode)
            return jsonify({"status": "ok", "mode": mode})
        except:
            return jsonify({"error": "invalid mode"}), 400
    
    @app.route("/stats")
    def stats():
        return jsonify(agent.get_stats())
    
    return app


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cloud_config.yaml")
    parser.add_argument("--mode", choices=["lightweight", "model_reprovision", "auto"], default=None)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    
    if args.mode:
        cfg["assistance_mode"] = args.mode
    
    agent = CloudAgentV2(cfg)
    
    if HAS_FLASK:
        app = create_app(agent)
        print(f"\n[CloudAgentV2] Starting on port {args.port}")
        print(f"[CloudAgentV2] Mode: {agent.assistance_mode.value}")
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    else:
        print("[CloudAgentV2] Flask not installed")


if __name__ == "__main__":
    main()

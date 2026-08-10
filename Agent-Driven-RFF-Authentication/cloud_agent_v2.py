# -*- coding: utf-8 -*-
"""Cloud coordination agent for the revised edge-cloud decision framework.

Responsibilities:
- accumulate compact escalation/recovery episodes;
- retrieve top-3 successful and top-2 failed cases;
- run the RAG-enabled LLM advisor on the cloud side;
- provide deterministic retrieval-only guidance for ablation/fallback;
- publish policy-guard ranges and maximum step sizes;
- reprovision the original source-trained model package when lightweight
  guidance remains insufficient;
- maintain auditable policy-guard outcome statistics.

The module intentionally contains no cross-edge model aggregation. Model
refresh in the revised manuscript is source-model reprovisioning, not FedAvg.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import tempfile
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from llm_advisor import LLMAdvisor, MAX_STEP_SIZE, PARAM_RANGES

try:
    from flask import Flask, jsonify, request, send_file
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


class AssistanceMode(Enum):
    LIGHTWEIGHT = "lightweight"
    MODEL_REPROVISION = "model_reprovision"
    AUTO = "auto"


@dataclass
class StateVector:
    closed_acc: float = 0.0
    open_auc: float = 0.0
    reject_rate: float = 0.0
    closed_gap: float = 0.0
    open_gap: float = 0.0
    accept_quantile: float = 0.90
    margin_quantile: float = 0.20
    delta_fused: float = 0.10
    delta_margin: float = 0.10
    fused_mean: float = 0.0
    fused_std: float = 1.0
    margin_mean: float = 0.0
    margin_std: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.asarray([
            self.closed_acc,
            self.open_auc,
            self.reject_rate,
            self.closed_gap,
            self.open_gap,
            self.accept_quantile,
            self.margin_quantile,
            self.delta_fused,
            self.delta_margin,
            self.fused_mean,
            self.fused_std,
            self.margin_mean,
            self.margin_std,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: Optional[Dict], objectives: Optional[Dict] = None) -> "StateVector":
        d = data or {}
        obj = objectives or {"min_closed_acc": 0.90, "target_open_auc": 0.85}
        metrics = d.get("metrics", {}) if isinstance(d.get("metrics"), dict) else {}
        score_stats = d.get("score_stats", {}) if isinstance(d.get("score_stats"), dict) else {}
        config = d.get("threshold_config", d.get("search_state", {}))
        if not isinstance(config, dict):
            config = {}

        def f(name: str, default: float) -> float:
            raw = d.get(name, metrics.get(name, default))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        closed = f("closed_acc", 0.0)
        open_auc = f("open_auc", 0.0)
        reject = f("reject_rate", 0.0)
        delta_fused = config.get("delta_fused", config.get("rho", d.get("delta_fused", d.get("rho", 0.10))))

        return cls(
            closed_acc=closed,
            open_auc=open_auc,
            reject_rate=reject,
            closed_gap=max(0.0, float(obj["min_closed_acc"]) - closed),
            open_gap=max(0.0, float(obj["target_open_auc"]) - open_auc),
            accept_quantile=float(config.get("accept_quantile", d.get("accept_quantile", 0.90))),
            margin_quantile=float(config.get("margin_quantile", d.get("margin_quantile", 0.20))),
            delta_fused=float(delta_fused),
            delta_margin=float(config.get("delta_margin", d.get("delta_margin", 0.10))),
            fused_mean=float(d.get("fused_mean", score_stats.get("fused_mean", 0.0))),
            fused_std=max(1e-6, float(d.get("fused_std", score_stats.get("fused_std", 1.0)))),
            margin_mean=float(d.get("margin_mean", score_stats.get("margin_mean", 0.0))),
            margin_std=max(1e-6, float(d.get("margin_std", score_stats.get("margin_std", 1.0)))),
        )

    def search_state(self) -> Dict[str, float]:
        return {
            "accept_quantile": self.accept_quantile,
            "margin_quantile": self.margin_quantile,
            "delta_fused": self.delta_fused,
            "delta_margin": self.delta_margin,
        }


@dataclass
class EpisodeRecord:
    episode_id: str
    terminal_id: str
    timestamp: str
    state_before: StateVector
    action: Dict[str, float]
    state_after: StateVector
    success: bool
    improvement: Dict[str, float] = field(default_factory=dict)
    event_descriptor: str = "domain_shift"


@dataclass
class RetrievedCase:
    episode: EpisodeRecord
    similarity: float
    relevance_reason: str


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
    created_at: str
    weights_file: str
    model_config: Dict
    baseline_thresholds: List[Dict]
    package_path: str


@dataclass
class EdgeNodeState:
    terminal_id: str
    last_seen: str
    current_metrics: Dict = field(default_factory=dict)
    consecutive_failures: int = 0
    num_episodes: int = 0


class EpisodeStore:
    def __init__(self, objectives: Dict):
        self.objectives = objectives
        self.episodes: List[EpisodeRecord] = []
        self.success_episodes: List[EpisodeRecord] = []
        self.failure_episodes: List[EpisodeRecord] = []

    def add(self, episode: EpisodeRecord) -> None:
        # Keep each episode exactly once. The previous implementation appended
        # twice, which biased retrieval and statistics.
        self.episodes.append(episode)
        if episode.success:
            self.success_episodes.append(episode)
        else:
            self.failure_episodes.append(episode)

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def retrieve(self, query_state: StateVector, k: int, success_only: Optional[bool]) -> List[RetrievedCase]:
        if success_only is True:
            candidates = self.success_episodes
        elif success_only is False:
            candidates = self.failure_episodes
        else:
            candidates = self.episodes
        if not candidates or k <= 0:
            return []

        q = query_state.to_vector()
        results: List[RetrievedCase] = []
        for ep in candidates:
            sim = self._cosine_similarity(q, ep.state_before.to_vector())
            reasons = []
            if abs(query_state.closed_gap - ep.state_before.closed_gap) <= 0.03:
                reasons.append("similar closed-set feasibility gap")
            if abs(query_state.open_gap - ep.state_before.open_gap) <= 0.03:
                reasons.append("similar open-set feasibility gap")
            if abs(query_state.reject_rate - ep.state_before.reject_rate) <= 0.05:
                reasons.append("similar rejection behavior")
            results.append(RetrievedCase(
                episode=ep,
                similarity=sim,
                relevance_reason="; ".join(reasons) if reasons else "state-vector cosine similarity",
            ))
        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[:k]

    def statistics(self) -> Dict:
        total = len(self.episodes)
        success = len(self.success_episodes)
        return {
            "total": total,
            "success": success,
            "failure": len(self.failure_episodes),
            "success_rate": (success / total) if total else 0.0,
        }


class RAGRetriever:
    def __init__(self, store: EpisodeStore):
        self.store = store

    def retrieve_for_guidance(self, query: StateVector, k_success: int = 3, k_failure: int = 2) -> Dict:
        return {
            "success_cases": self.store.retrieve(query, k_success, True),
            "failure_cases": self.store.retrieve(query, k_failure, False),
            "statistics": self.store.statistics(),
        }

    @staticmethod
    def format_for_llm(retrieved: Dict) -> str:
        stats = retrieved.get("statistics", {})
        lines = [
            "### Historical experience",
            f"Total episodes: {stats.get('total', 0)}; success rate: {stats.get('success_rate', 0.0):.1%}",
        ]
        successes = retrieved.get("success_cases", [])
        if successes:
            lines.append("### Top similar successful cases")
            for i, case in enumerate(successes, 1):
                ep = case.episode
                lines.append(
                    f"Case {i}: similarity={case.similarity:.4f}; "
                    f"before Ac/Ao/R={ep.state_before.closed_acc:.4f}/{ep.state_before.open_auc:.4f}/{ep.state_before.reject_rate:.4f}; "
                    f"score mean(F/M)={ep.state_before.fused_mean:.4f}/{ep.state_before.margin_mean:.4f}; "
                    f"action={ep.action}; after Ac/Ao/R={ep.state_after.closed_acc:.4f}/{ep.state_after.open_auc:.4f}/{ep.state_after.reject_rate:.4f}"
                )
        else:
            lines.append("### No suitable successful case is currently available")

        failures = retrieved.get("failure_cases", [])
        if failures:
            lines.append("### Similar failed cases (negative contextual evidence only)")
            for i, case in enumerate(failures, 1):
                ep = case.episode
                lines.append(
                    f"Failed case {i}: similarity={case.similarity:.4f}; "
                    f"score mean(F/M)={ep.state_before.fused_mean:.4f}/{ep.state_before.margin_mean:.4f}; "
                    f"action={ep.action}; result Ac/Ao={ep.state_after.closed_acc:.4f}/{ep.state_after.open_auc:.4f}"
                )
        return "\n".join(lines)


class PolicyGuard:
    """Cloud-published deterministic safety policy and retrieval-only advisor."""

    SAFE_RANGES = dict(PARAM_RANGES)
    MAX_STEP = dict(MAX_STEP_SIZE)

    @classmethod
    def clamp_to_policy(cls, proposal: Dict, current: Dict) -> Tuple[Dict, List[str]]:
        adjusted: Dict[str, float] = {}
        notes: List[str] = []
        for raw_key, raw_value in (proposal or {}).items():
            key = "delta_fused" if raw_key == "rho" else raw_key
            if key not in cls.SAFE_RANGES:
                notes.append(f"unsupported:{raw_key}")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                notes.append(f"non_numeric:{raw_key}")
                continue
            if not math.isfinite(value):
                notes.append(f"non_finite:{raw_key}")
                continue

            current_value = float(current.get(key, value))
            step = cls.MAX_STEP[key]
            delta = value - current_value
            if abs(delta) > step:
                value = current_value + (step if delta > 0 else -step)
                notes.append(f"step_clamp:{key}")
            lo, hi = cls.SAFE_RANGES[key]
            clipped = min(hi, max(lo, value))
            if clipped != value:
                notes.append(f"range_clamp:{key}")
            adjusted[key] = float(clipped)
        return adjusted, notes

    @classmethod
    def deterministic_guidance(cls, success_cases: List[RetrievedCase], current: Dict) -> Optional[Dict]:
        """Top-3 non-generative retrieval guidance with nonnegative cosine weights."""
        usable = []
        for case in success_cases[:3]:
            sim = max(0.0, float(case.similarity))
            action = {
                ("delta_fused" if k == "rho" else k): v
                for k, v in case.episode.action.items()
                if ("delta_fused" if k == "rho" else k) in {"delta_fused", "delta_margin"}
            }
            if sim > 0.0 and action:
                usable.append((sim, action))
        if not usable:
            return None

        total = sum(sim for sim, _ in usable)
        if total <= 0.0:
            return None

        keys = set().union(*(action.keys() for _, action in usable))
        proposal: Dict[str, float] = {}
        for key in keys:
            weighted_sum = 0.0
            weight_sum = 0.0
            for sim, action in usable:
                if key in action:
                    w = sim / total
                    weighted_sum += w * float(action[key])
                    weight_sum += w
            if weight_sum > 0:
                proposal[key] = weighted_sum / weight_sum

        guarded, _ = cls.clamp_to_policy(proposal, current)
        return guarded or None


class CloudAgentV2:
    def __init__(self, cfg: Optional[dict] = None):
        self.config = cfg or self._default_config()
        self.assistance_mode = AssistanceMode(self.config.get("assistance_mode", "auto"))
        self.objectives = self.config.get("objectives", {
            "min_closed_acc": 0.90,
            "target_open_auc": 0.85,
        })

        self.data_dir = Path(self.config.get("data_dir", "cloud_data"))
        self.models_dir = self.data_dir / "models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        audit_cfg = self.config.get("audit", {})
        self.audit_path = self.data_dir / audit_cfg.get("policy_guard_log", "policy_guard_events.jsonl")

        self.episode_store = EpisodeStore(self.objectives)
        self.retriever = RAGRetriever(self.episode_store)
        self.policy_guard = PolicyGuard()
        self.threshold_priors: Dict[str, ThresholdPriorInterval] = {}
        self.edge_states: Dict[str, EdgeNodeState] = {}
        self.experience_bank: Dict[str, List[Dict]] = defaultdict(list)
        self.experience_read_only = bool(self.config.get("experience_bank", {}).get("read_only", False))
        self.pending_audits: Dict[str, Dict] = {}
        self.model_package: Optional[ModelPackage] = None
        self.lock = threading.RLock()

        llm_cfg = self.config.get("llm", {})
        self.llm_enabled = bool(llm_cfg.get("enable", True))
        self.llm_advisor = LLMAdvisor(llm_cfg) if self.llm_enabled else None

        self._load_state()
        self._ensure_initial_model_package()

        print(f"[CloudAgentV2] mode={self.assistance_mode.value}")
        print(f"[CloudAgentV2] episodes={len(self.episode_store.episodes)}")
        print(f"[CloudAgentV2] cloud-side LLM enabled={self.llm_enabled}")

    @staticmethod
    def _default_config() -> dict:
        return {
            "data_dir": "cloud_data",
            "assistance_mode": "auto",
            "rag": {"k_success": 3, "k_failure": 2},
            "experience_bank": {"read_only": False},
            "prior": {"min_samples": 3, "interval_expansion": 0.05},
            "mismatch": {"max_consecutive_failures": 3},
            "objectives": {"min_closed_acc": 0.90, "target_open_auc": 0.85},
            "llm": {"enable": False},
            "initial_model": {},
            "audit": {"policy_guard_log": "policy_guard_events.jsonl"},
        }

    def _state_file(self) -> Path:
        return self.data_dir / "state.json"

    def receive_episode(self, data: Dict) -> Dict:
        """Store compact telemetry only; raw IQ and learned edge weights are not uploaded.

        ``experience_bank.read_only`` can be enabled during the four-arm
        ablation so all guidance strategies query the same frozen historical
        corpus.
        """
        with self.lock:
            if self.experience_read_only:
                return {"status": "read_only", "message": "experience bank is frozen; episode not stored"}
            terminal_id = str(data.get("terminal_id", "unknown"))
            before = StateVector.from_dict(data.get("state_before", data), self.objectives)
            after = StateVector.from_dict(data.get("state_after", data), self.objectives)
            action_raw = data.get("threshold_config", data.get("action", {}))
            # Historical episodes store the operating point that was actually
            # applied. Step-size limits govern *new advisory events* and must
            # not retroactively alter the recorded successful action. We only
            # sanitize type/finite/range here.
            action: Dict[str, float] = {}
            for raw_key, raw_value in (action_raw or {}).items():
                key = "delta_fused" if raw_key == "rho" else raw_key
                if key not in PolicyGuard.SAFE_RANGES:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                lo, hi = PolicyGuard.SAFE_RANGES[key]
                action[key] = float(min(hi, max(lo, value)))
            success = bool(data.get("recovery_success", False))
            improvement = {
                "closed_acc_delta": after.closed_acc - before.closed_acc,
                "open_auc_delta": after.open_auc - before.open_auc,
                "reject_rate_delta": after.reject_rate - before.reject_rate,
            }
            episode = EpisodeRecord(
                episode_id=str(data.get("episode_id") or uuid.uuid4().hex),
                terminal_id=terminal_id,
                timestamp=str(data.get("timestamp", datetime.now().isoformat())),
                state_before=before,
                action=action,
                state_after=after,
                success=success,
                improvement=improvement,
                event_descriptor=str(data.get("event_descriptor", "domain_shift")),
            )
            self.episode_store.add(episode)
            self.experience_bank[terminal_id].append(data)
            self._update_edge_state(terminal_id, after, success)
            self._try_update_prior("default")
            self._save_state()
            return {"status": "success", "episode_id": episode.episode_id}

    def _update_edge_state(self, terminal_id: str, state: StateVector, success: bool) -> None:
        node = self.edge_states.get(terminal_id)
        if node is None:
            node = EdgeNodeState(terminal_id=terminal_id, last_seen=datetime.now().isoformat())
            self.edge_states[terminal_id] = node
        node.last_seen = datetime.now().isoformat()
        node.current_metrics = {
            "closed_acc": state.closed_acc,
            "open_auc": state.open_auc,
            "reject_rate": state.reject_rate,
        }
        node.num_episodes += 1
        node.consecutive_failures = 0 if success else node.consecutive_failures + 1

    def _retrieve(self, query: StateVector, use_retrieval: bool = True) -> Dict:
        if not use_retrieval:
            return {"success_cases": [], "failure_cases": [], "statistics": self.episode_store.statistics()}
        rag_cfg = self.config.get("rag", {})
        return self.retriever.retrieve_for_guidance(
            query,
            k_success=int(rag_cfg.get("k_success", 3)),
            k_failure=int(rag_cfg.get("k_failure", 2)),
        )

    def _case_to_dict(self, case: RetrievedCase) -> Dict:
        ep = case.episode
        return {
            "episode_id": ep.episode_id,
            "similarity": float(case.similarity),
            "relevance": case.relevance_reason,
            "state_before": asdict(ep.state_before),
            "action": ep.action,
            "state_after": asdict(ep.state_after),
            "success": ep.success,
            "improvement": ep.improvement,
        }

    def _global_summary(self) -> Dict:
        return self.episode_store.statistics()

    def _default_threshold_interval(self) -> Dict:
        return {
            "accept_quantile_range": list(PARAM_RANGES["accept_quantile"]),
            "margin_quantile_range": list(PARAM_RANGES["margin_quantile"]),
            "delta_fused_range": list(PARAM_RANGES["delta_fused"]),
            "delta_margin_range": list(PARAM_RANGES["delta_margin"]),
        }

    def _prior_interval(self, domain: str = "default") -> Dict:
        prior = self.threshold_priors.get(domain)
        if not prior:
            return self._default_threshold_interval()
        return {
            "accept_quantile_range": [prior.accept_quantile_min, prior.accept_quantile_max],
            "margin_quantile_range": [prior.margin_quantile_min, prior.margin_quantile_max],
            "delta_fused_range": [prior.delta_fused_min, prior.delta_fused_max],
            "delta_margin_range": [prior.delta_margin_min, prior.delta_margin_max],
        }

    def _try_update_prior(self, domain: str) -> None:
        cfg = self.config.get("prior", {})
        min_samples = int(cfg.get("min_samples", 3))
        successes = [ep for ep in self.episode_store.success_episodes if ep.action]
        if len(successes) < min_samples:
            return

        def values(key: str, default: float) -> np.ndarray:
            arr = [float(ep.action.get(key, default)) for ep in successes if key in ep.action]
            return np.asarray(arr, dtype=np.float64)

        defaults = {"accept_quantile": 0.90, "margin_quantile": 0.20, "delta_fused": 0.10, "delta_margin": 0.10}
        bounds = {}
        for key, default in defaults.items():
            arr = values(key, default)
            if arr.size == 0:
                bounds[key] = PARAM_RANGES[key]
                continue
            lo, hi = np.quantile(arr, [0.10, 0.90]) if arr.size > 1 else (arr[0], arr[0])
            if key == "delta_fused":
                expansion = float(cfg.get("interval_expansion", 0.05))
                lo -= expansion
                hi += expansion
            glo, ghi = PARAM_RANGES[key]
            bounds[key] = (max(glo, float(lo)), min(ghi, float(hi)))

        self.threshold_priors[domain] = ThresholdPriorInterval(
            domain_cluster=domain,
            accept_quantile_min=bounds["accept_quantile"][0],
            accept_quantile_max=bounds["accept_quantile"][1],
            margin_quantile_min=bounds["margin_quantile"][0],
            margin_quantile_max=bounds["margin_quantile"][1],
            delta_fused_min=bounds["delta_fused"][0],
            delta_fused_max=bounds["delta_fused"][1],
            delta_margin_min=bounds["delta_margin"][0],
            delta_margin_max=bounds["delta_margin"][1],
            confidence=min(0.95, 0.5 + 0.05 * len(successes)),
            num_samples=len(successes),
            created_at=datetime.now().isoformat(),
        )


    def _build_prior(self, domain: str = "default") -> Optional[ThresholdPriorInterval]:
        """Build/update the empirical threshold prior and return it.

        The resulting interval is always intersected with the manuscript-level
        global safe set by ``_try_update_prior``.
        """
        self._try_update_prior(domain)
        return self.threshold_priors.get(domain)

    def _record_immediate_no_executable(self, source: str, notes: Optional[List[str]] = None) -> str:
        proposal_id = uuid.uuid4().hex
        event = {
            "proposal_id": proposal_id,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "outcome": "no_executable",
            "first_pass_adjusted": False,
            "validation_notes": notes or [],
            "range_clamp_events": sum(1 for n in (notes or []) if str(n).startswith("range_clamp:")),
            "step_clamp_events": sum(1 for n in (notes or []) if str(n).startswith("step_clamp:")),
        }
        self._append_audit(event)
        return proposal_id

    def _register_pending_audit(self, terminal_id: str, advice, strategy: str) -> None:
        self.pending_audits[advice.proposal_id] = {
            "proposal_id": advice.proposal_id,
            "terminal_id": terminal_id,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "first_pass_adjusted": bool(advice.first_pass_adjusted),
            "validation_notes": list(advice.validation_notes),
            "raw_proposal": dict(advice.raw_param_changes),
            "post_first_pass_params": dict(advice.param_changes),
        }

    def finalize_policy_audit(self, data: Dict) -> Dict:
        proposal_id = str(data.get("proposal_id", ""))
        if not proposal_id:
            raise ValueError("proposal_id is required")
        pending = self.pending_audits.pop(proposal_id, None)
        if pending is None:
            raise KeyError(f"Unknown or already finalized proposal_id: {proposal_id}")

        edge_adjusted = not bool(data.get("policy_approved", True))
        executable = bool(data.get("executable", bool(data.get("final_params", {}))))
        if not executable:
            outcome = "no_executable"
        else:
            outcome = "guard_adjusted" if (pending["first_pass_adjusted"] or edge_adjusted) else "direct"
        notes = list(pending.get("validation_notes", [])) + list(data.get("policy_warnings", []) or [])
        event = {
            **pending,
            "outcome": outcome,
            "edge_policy_approved": not edge_adjusted,
            "final_params": data.get("final_params", {}),
            "range_clamp_events": sum(1 for n in notes if "range_clamp" in str(n) or "range" in str(n).lower()),
            "step_clamp_events": sum(1 for n in notes if "step_clamp" in str(n) or "step" in str(n).lower()),
            "validation_notes": notes,
        }
        self._append_audit(event)
        return {"status": "recorded", "outcome": outcome}

    def _append_audit(self, event: Dict) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _policy_stats(self) -> Dict:
        counts = {"attempts": 0, "direct": 0, "guard_adjusted": 0, "no_executable": 0,
                  "range_clamp_events": 0, "step_clamp_events": 0,
                  "per_parameter_clamp_counts": {}}
        if not self.audit_path.exists():
            return counts
        with open(self.audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                outcome = event.get("outcome")
                if outcome not in {"direct", "guard_adjusted", "no_executable"}:
                    continue
                counts["attempts"] += 1
                counts[outcome] += 1
                counts["range_clamp_events"] += int(event.get("range_clamp_events", 0))
                counts["step_clamp_events"] += int(event.get("step_clamp_events", 0))
                for note in event.get("validation_notes", []) or []:
                    text = str(note)
                    if text.startswith("range_clamp:") or text.startswith("step_clamp:"):
                        _, key = text.split(":", 1)
                        per_param = counts["per_parameter_clamp_counts"]
                        per_param[key] = int(per_param.get(key, 0)) + 1
        n = counts["attempts"]
        counts["direct_acceptance_rate"] = counts["direct"] / n if n else 0.0
        counts["guard_adjusted_rate"] = counts["guard_adjusted"] / n if n else 0.0
        counts["no_executable_rate"] = counts["no_executable"] / n if n else 0.0
        counts["executable_after_guard_rate"] = (counts["direct"] + counts["guard_adjusted"]) / n if n else 0.0
        return counts

    def _provide_lightweight(self, terminal_id: str, current_state: Dict, strategy: str = "rag") -> Dict:
        query = StateVector.from_dict(current_state, self.objectives)
        current_search = query.search_state()
        if strategy == "rag":
            retrieved = self._retrieve(query, use_retrieval=True)
        elif strategy == "deterministic":
            # Retrieval-only comparator uses successful cases only; no failed
            # contextual evidence and no LLM are involved in this arm.
            rag_cfg = self.config.get("rag", {})
            retrieved = self.retriever.retrieve_for_guidance(
                query, k_success=int(rag_cfg.get("k_success", 3)), k_failure=0
            )
        else:
            retrieved = self._retrieve(query, use_retrieval=False)
        success_cases = retrieved.get("success_cases", [])
        failure_cases = retrieved.get("failure_cases", [])

        common = {
            "type": "lightweight",
            "strategy": strategy,
            # Only the RAG arm receives the empirical retrieval-derived prior.
            # LLM-only uses no retrieval information; deterministic guidance
            # uses only the weighted top-3 threshold offsets.
            "threshold_interval": (self._prior_interval("default") if strategy == "rag" else self._default_threshold_interval()),
            "safe_param_ranges": {k: list(v) for k, v in PolicyGuard.SAFE_RANGES.items()},
            "max_step_sizes": dict(PolicyGuard.MAX_STEP),
            "retrieved_cases": {
                "success": [self._case_to_dict(c) for c in success_cases],
                "failure": [self._case_to_dict(c) for c in failure_cases],
            },
            "global_summary": self._global_summary(),
        }

        if strategy == "deterministic":
            det = self.policy_guard.deterministic_guidance(success_cases, current_search)
            return {
                **common,
                "param_changes": det or {},
                "source": "deterministic_retrieval" if det else "no_evidence_supported_guidance",
                "policy_approved": True,
                "requires_reprocessing": False,
                "message": "Retrieval-only deterministic guidance; no LLM call was made.",
            }

        rag_context = self.retriever.format_for_llm(retrieved) if strategy == "rag" else ""
        if self.llm_advisor is None:
            det = self.policy_guard.deterministic_guidance(success_cases, current_search) if strategy == "rag" else None
            return {
                **common,
                "param_changes": det or {},
                "source": "deterministic_fallback" if det else "no_executable_guidance",
                "policy_approved": True,
                "requires_reprocessing": False,
                "message": "LLM unavailable; deterministic retrieval guidance used when evidence exists.",
            }

        try:
            advice = self.llm_advisor.get_guidance(
                current_state=current_search,
                performance={
                    "closed_acc": query.closed_acc,
                    "open_auc": query.open_auc,
                    "reject_rate": query.reject_rate,
                },
                objectives=self.objectives,
                rag_context=rag_context,
                history=current_state.get("history", []),
            )
        except Exception as exc:
            proposal_id = self._record_immediate_no_executable("llm_exception", [f"exception:{type(exc).__name__}"])
            det = self.policy_guard.deterministic_guidance(success_cases, current_search) if strategy == "rag" else None
            return {
                **common,
                "proposal_id": proposal_id,
                "param_changes": det or {},
                "source": "deterministic_fallback" if det else "no_executable_guidance",
                "policy_approved": True,
                "requires_reprocessing": False,
                "message": f"LLM advisory failed; deterministic alternative used when available: {exc}",
            }

        if advice is None or not advice.param_changes:
            proposal_id = advice.proposal_id if advice is not None else self._record_immediate_no_executable("llm_parse_or_validation")
            if advice is not None:
                self._append_audit({
                    "proposal_id": proposal_id,
                    "timestamp": datetime.now().isoformat(),
                    "source": strategy,
                    "outcome": "no_executable",
                    "raw_proposal": dict(advice.raw_param_changes),
                    "post_first_pass_params": dict(advice.param_changes),
                    "validation_notes": advice.validation_notes,
                    "range_clamp_events": sum(1 for n in advice.validation_notes if str(n).startswith("range_clamp:")),
                    "step_clamp_events": sum(1 for n in advice.validation_notes if str(n).startswith("step_clamp:")),
                })
            det = self.policy_guard.deterministic_guidance(success_cases, current_search) if strategy == "rag" else None
            return {
                **common,
                "proposal_id": proposal_id,
                "param_changes": det or {},
                "source": "deterministic_fallback" if det else "no_executable_guidance",
                "policy_approved": True,
                "requires_reprocessing": False,
                "message": "No executable LLM-derived update remained after deterministic validation.",
            }

        self._register_pending_audit(terminal_id, advice, strategy)
        return {
            **common,
            "proposal_id": advice.proposal_id,
            "param_changes": advice.param_changes,
            "analysis": advice.analysis,
            "reasoning": advice.reasoning,
            "confidence": advice.confidence,
            "first_pass_adjusted": advice.first_pass_adjusted,
            "validation_notes": advice.validation_notes,
            "source": "rag_llm" if strategy == "rag" else "llm_only",
            "policy_approved": not advice.first_pass_adjusted,
            "requires_reprocessing": False,
            "message": "Cloud-side bounded LLM guidance; final edge-side policy review is required.",
        }

    def get_guidance(self, terminal_id: str, current_state: Dict, strategy: str = "rag", stage: str = "post_edge") -> Dict:
        if strategy not in {"rag", "llm_only", "deterministic"}:
            raise ValueError("strategy must be rag, llm_only, or deterministic")

        # In AUTO mode, model refresh is the final mechanism after lightweight
        # guidance remains infeasible. It re-provisions the original source model.
        if self.assistance_mode == AssistanceMode.MODEL_REPROVISION or (
            self.assistance_mode == AssistanceMode.AUTO and stage == "post_lightweight"
        ):
            return self._provide_model()

        return self._provide_lightweight(terminal_id, current_state, strategy)

    def _ensure_initial_model_package(self) -> None:
        initial = self.config.get("initial_model", {})
        weights_path = Path(str(initial.get("weights_path", "")))
        if not weights_path.is_file():
            self.model_package = None
            return

        dest_weights = self.models_dir / "source_model.pth"
        if not dest_weights.exists() or dest_weights.stat().st_mtime < weights_path.stat().st_mtime:
            shutil.copy2(weights_path, dest_weights)

        model_cfg = dict(initial.get("model_config", {}))
        thresholds = list(initial.get("baseline_thresholds", [
            {"accept_quantile": 0.90, "margin_quantile": 0.20, "delta_fused": 0.10, "delta_margin": 0.10}
        ]))
        package_path = self.models_dir / "source_model_package.tar.gz"
        self._create_source_archive(dest_weights, model_cfg, thresholds, package_path)
        self.model_package = ModelPackage(
            version="source_initial",
            created_at=datetime.now().isoformat(),
            weights_file=str(dest_weights),
            model_config=model_cfg,
            baseline_thresholds=thresholds,
            package_path=str(package_path),
        )

    @staticmethod
    def _create_source_archive(weights_path: Path, model_config: Dict,
                               thresholds: List[Dict], package_path: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy2(weights_path, tmp / "model_weights.pth")
            (tmp / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
            (tmp / "baseline_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
            (tmp / "metadata.json").write_text(json.dumps({
                "version": "source_initial",
                "refresh_type": "source_model_reprovisioning",
            }, indent=2), encoding="utf-8")
            with tarfile.open(package_path, "w:gz") as tar:
                for path in sorted(tmp.iterdir()):
                    tar.add(path, arcname=path.name)

    def _provide_model(self) -> Dict:
        available = self.model_package is not None and Path(self.model_package.package_path).is_file()
        return {
            "type": "model_reprovision",
            "available": available,
            "version": self.model_package.version if available else None,
            "refresh_type": "source_model_reprovisioning",
            "message": "Original source-trained model package is re-provisioned; no edge-model aggregation is used.",
        }

    def get_model_path(self) -> Optional[str]:
        if self.model_package and Path(self.model_package.package_path).is_file():
            return self.model_package.package_path
        return None

    def get_stats(self) -> Dict:
        ep = self.episode_store.statistics()
        return {
            "mode": self.assistance_mode.value,
            "episodes": ep,
            "policy_guard": self._policy_stats(),
            "pending_policy_audits": len(self.pending_audits),
            "source_model_available": self.get_model_path() is not None,
        }

    def set_mode(self, mode: str) -> None:
        self.assistance_mode = AssistanceMode(mode)

    def _save_state(self) -> None:
        state = {
            "episodes": [self._episode_to_dict(ep) for ep in self.episode_store.episodes],
            "threshold_priors": {k: asdict(v) for k, v in self.threshold_priors.items()},
            "edge_states": {k: asdict(v) for k, v in self.edge_states.items()},
            "experience_bank": dict(self.experience_bank),
        }
        self._state_file().write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _episode_to_dict(ep: EpisodeRecord) -> Dict:
        return {
            "episode_id": ep.episode_id,
            "terminal_id": ep.terminal_id,
            "timestamp": ep.timestamp,
            "state_before": asdict(ep.state_before),
            "action": ep.action,
            "state_after": asdict(ep.state_after),
            "success": ep.success,
            "improvement": ep.improvement,
            "event_descriptor": ep.event_descriptor,
        }

    def _load_state(self) -> None:
        path = self._state_file()
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self.experience_bank = defaultdict(list, state.get("experience_bank", {}))
            for item in state.get("episodes", []):
                ep = EpisodeRecord(
                    episode_id=item["episode_id"],
                    terminal_id=item["terminal_id"],
                    timestamp=item["timestamp"],
                    state_before=StateVector(**item["state_before"]),
                    action=item.get("action", {}),
                    state_after=StateVector(**item["state_after"]),
                    success=bool(item.get("success", False)),
                    improvement=item.get("improvement", {}),
                    event_descriptor=item.get("event_descriptor", "domain_shift"),
                )
                self.episode_store.add(ep)
            for key, value in state.get("threshold_priors", {}).items():
                self.threshold_priors[key] = ThresholdPriorInterval(**value)
            for key, value in state.get("edge_states", {}).items():
                self.edge_states[key] = EdgeNodeState(**value)
        except Exception as exc:
            print(f"[CloudAgentV2] state load skipped: {exc}")


def create_app(agent: CloudAgentV2) -> Flask:
    if not HAS_FLASK:
        raise RuntimeError("Flask is not installed")
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "mode": agent.assistance_mode.value})

    @app.route("/episode", methods=["POST"])
    def episode():
        try:
            data = request.get_json(silent=True) or {}
            return jsonify(agent.receive_episode(data))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/guidance", methods=["POST"])
    def guidance():
        try:
            data = request.get_json(silent=True) or {}
            terminal_id = str(data.get("terminal_id", "unknown"))
            current_state = data.get("current_state", {}) or {}
            strategy = str(data.get("strategy", "rag"))
            stage = str(data.get("stage", "post_edge"))
            return jsonify(agent.get_guidance(terminal_id, current_state, strategy, stage))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    # Backward-compatible route name; execution is still cloud-side.
    @app.route("/rag_guidance", methods=["POST"])
    def rag_guidance():
        return guidance()

    @app.route("/policy_guard/audit", methods=["POST"])
    def policy_audit():
        try:
            return jsonify(agent.finalize_policy_audit(request.get_json(silent=True) or {}))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/model/download")
    def download_model():
        path = agent.get_model_path()
        if path and os.path.exists(path):
            return send_file(path, as_attachment=True)
        return jsonify({"error": "source model package unavailable"}), 404

    @app.route("/mode/<mode>", methods=["POST"])
    def set_mode(mode: str):
        try:
            agent.set_mode(mode)
            return jsonify({"status": "ok", "mode": mode})
        except ValueError:
            return jsonify({"error": "invalid mode"}), 400

    @app.route("/stats")
    def stats():
        return jsonify(agent.get_stats())

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cloud coordination agent")
    parser.add_argument("--config", default="cloud_config.yaml")
    parser.add_argument("--mode", choices=["lightweight", "model_reprovision", "auto"], default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if args.mode:
        cfg["assistance_mode"] = args.mode

    agent = CloudAgentV2(cfg)
    if not HAS_FLASK:
        raise RuntimeError("Flask is required to run the cloud service")
    server = cfg.get("server", {})
    host = server.get("host", "0.0.0.0")
    port = int(args.port or server.get("port", 5000))
    app = create_app(agent)
    app.run(host=host, port=port, debug=bool(server.get("debug", False)), threaded=True)


if __name__ == "__main__":
    main()

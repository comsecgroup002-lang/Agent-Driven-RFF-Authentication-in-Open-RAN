# -*- coding: utf-8 -*-
"""Cached-score threshold replay for the edge control agent.

The module implements the operating-point search described in the revised
manuscript. Candidate selection is performed on a runtime control buffer and
final reported metrics are computed on a disjoint evaluation buffer. This
prevents the reported evaluation labels from being used to select the
operating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


DEFAULT_SAFE_RANGES = {
    "accept_quantile": (0.80, 0.95),
    "margin_quantile": (0.10, 0.30),
    "delta_fused": (-0.15, 0.35),
    "delta_margin": (0.05, 0.35),
}


@dataclass(frozen=True)
class SearchResult:
    config: Dict[str, float]
    control_metrics: Dict[str, float]
    metrics: Dict[str, float]
    utility: float


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _unique_sorted(values: Iterable[float]) -> List[float]:
    return sorted({round(float(v), 10) for v in values})


def build_search_grid(cfg: dict, guidance: Optional[Dict] = None) -> Dict[str, List[float]]:
    """Build the globally safe search grid, optionally narrowed by cloud guidance.

    The empirical/cloud interval may restrict the global safe set but never
    enlarge it. A validated point proposal is inserted into the corresponding
    grid so that it can be evaluated directly.
    """
    ts_cfg = cfg.get("threshold_search", {})
    grids = {
        "accept_quantile": list(ts_cfg.get(
            "accept_quantile_grid", [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
        )),
        "margin_quantile": list(ts_cfg.get(
            "margin_quantile_grid", [0.10, 0.15, 0.20, 0.25, 0.30]
        )),
        "delta_fused": list(ts_cfg.get(
            "delta_fused_grid", [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        )),
        "delta_margin": list(ts_cfg.get(
            "delta_margin_grid", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        )),
    }

    # Enforce the manuscript-level global safe set first.
    for key, (lo, hi) in DEFAULT_SAFE_RANGES.items():
        grids[key] = _unique_sorted(v for v in grids[key] if lo <= float(v) <= hi)

    if not guidance:
        return grids

    intervals = guidance.get("threshold_interval", {}) or {}
    interval_keys = {
        "accept_quantile": "accept_quantile_range",
        "margin_quantile": "margin_quantile_range",
        "delta_fused": "delta_fused_range",
        "delta_margin": "delta_margin_range",
    }
    for key, interval_name in interval_keys.items():
        raw = intervals.get(interval_name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        glo, ghi = DEFAULT_SAFE_RANGES[key]
        lo = max(glo, _finite_float(raw[0], glo))
        hi = min(ghi, _finite_float(raw[1], ghi))
        if lo > hi:
            continue
        narrowed = [v for v in grids[key] if lo <= v <= hi]
        if narrowed:
            grids[key] = narrowed

    proposal = guidance.get("param_changes", {}) or {}
    if "rho" in proposal and "delta_fused" not in proposal:
        proposal = dict(proposal)
        proposal["delta_fused"] = proposal["rho"]

    # Keep the validated proposal itself in the candidate set.
    for key in DEFAULT_SAFE_RANGES:
        if key not in proposal:
            continue
        lo, hi = DEFAULT_SAFE_RANGES[key]
        value = min(hi, max(lo, _finite_float(proposal[key], (lo + hi) / 2)))
        grids[key] = _unique_sorted([*grids[key], value])

    return grids


def _class_thresholds(values: np.ndarray, predictions: np.ndarray, quantile: float,
                      default: float) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    for c in np.unique(predictions):
        idx = predictions == c
        if np.any(idx):
            thresholds[int(c)] = float(np.quantile(values[idx], quantile))
        else:
            thresholds[int(c)] = default
    return thresholds


def evaluate_candidate(features: Dict[str, np.ndarray], split: str,
                       accept_quantile: float, margin_quantile: float,
                       delta_fused: float, delta_margin: float) -> Dict[str, float]:
    """Evaluate one cached-score operating point on ``control`` or ``eval`` data."""
    if split not in {"control", "eval"}:
        raise ValueError("split must be 'control' or 'eval'")

    alpha = float(features["alpha"])
    fused_val = alpha * features["ez_val"] + (1.0 - alpha) * features["dz_val"]
    pred_val = features["log_val"].argmax(axis=1)

    tau_f = _class_thresholds(fused_val, pred_val, accept_quantile, 1e9)
    tau_m = _class_thresholds(features["mz_val"], pred_val, margin_quantile, -1e9)

    logits = features[f"log_{split}"]
    labels = features[f"y_{split}"]
    ez = features[f"ez_{split}"]
    dz = features[f"dz_{split}"]
    mz = features[f"mz_{split}"]

    fused = alpha * ez + (1.0 - alpha) * dz
    pred_closed = logits.argmax(axis=1)
    thr_f = np.asarray([tau_f.get(int(c), 1e9) for c in pred_closed], dtype=np.float64)
    thr_m = np.asarray([tau_m.get(int(c), -1e9) for c in pred_closed], dtype=np.float64)

    # Larger delta_fused makes the novelty gate more permissive; larger
    # delta_margin makes the margin gate more conservative.
    d_fused = fused - (thr_f + float(delta_fused))
    d_margin = (thr_m + float(delta_margin)) - mz
    gate_score = np.maximum(d_fused, d_margin)

    accepted = gate_score <= 0.0
    pred_open = np.where(accepted, pred_closed, -1)
    reject_rate = float(np.mean(pred_open == -1)) if len(pred_open) else 1.0

    known_accepted = (labels != -1) & (pred_open != -1)
    if np.any(known_accepted):
        closed_acc = float(accuracy_score(labels[known_accepted], pred_open[known_accepted]))
    else:
        closed_acc = 0.0

    unknown_target = (labels == -1).astype(np.int32)
    if np.unique(unknown_target).size == 2:
        try:
            open_auc = float(roc_auc_score(unknown_target, gate_score))
        except ValueError:
            open_auc = 0.5
    else:
        open_auc = 0.5

    return {
        "closed_acc": closed_acc,
        "open_auc": open_auc,
        "reject_rate": reject_rate,
    }


def is_feasible(metrics: Dict[str, float], objectives: Dict[str, float]) -> bool:
    """Authentication feasibility is defined only by Ac and Ao.

    Rejection rate remains an auxiliary service-availability indicator used in
    Pareto filtering and operating-point selection, as stated in the manuscript.
    """
    return (
        float(metrics.get("closed_acc", 0.0)) >= float(objectives["min_closed_acc"])
        and float(metrics.get("open_auc", 0.0)) >= float(objectives["target_open_auc"])
    )


def _dominates(a: Dict, b: Dict) -> bool:
    am, bm = a["control_metrics"], b["control_metrics"]
    weak = (
        am["closed_acc"] >= bm["closed_acc"]
        and am["open_auc"] >= bm["open_auc"]
        and am["reject_rate"] <= bm["reject_rate"]
    )
    strict = (
        am["closed_acc"] > bm["closed_acc"]
        or am["open_auc"] > bm["open_auc"]
        or am["reject_rate"] < bm["reject_rate"]
    )
    return weak and strict


def pareto_front(results: Sequence[Dict]) -> List[Dict]:
    front: List[Dict] = []
    for candidate in results:
        if not any(_dominates(other, candidate) for other in results if other is not candidate):
            front.append(candidate)
    return front


def _config_distance(config: Dict[str, float], previous: Optional[Dict[str, float]]) -> float:
    if not previous:
        return 0.0
    keys = ("delta_fused", "delta_margin")
    return float(np.sqrt(sum((float(config.get(k, 0.0)) - float(previous.get(k, 0.0))) ** 2 for k in keys)))


def _selection_utility(candidate: Dict, cfg: dict, previous: Optional[Dict[str, float]]) -> float:
    sel = cfg.get("threshold_search", {}).get("selection", {})
    lambda_r = float(sel.get("lambda_r", 0.25))
    lambda_o = float(sel.get("lambda_o", 1.0))
    lambda_s = float(sel.get("lambda_s", 0.20))
    m = candidate["control_metrics"]
    # Eq.-style smoothness-aware utility: lower is better.
    return (
        lambda_r * float(m["reject_rate"])
        - lambda_o * float(m["open_auc"])
        + lambda_s * _config_distance(candidate["config"], previous)
    )


def select_operating_point(front: Sequence[Dict], objectives: Dict[str, float],
                           cfg: dict, previous: Optional[Dict[str, float]] = None) -> Dict:
    if not front:
        raise ValueError("Pareto front is empty")

    feasible = [r for r in front if is_feasible(r["control_metrics"], objectives)]
    pool = feasible if feasible else list(front)

    for item in pool:
        item["utility"] = _selection_utility(item, cfg, previous)

    if feasible:
        return min(pool, key=lambda r: r["utility"])

    # If no candidate is feasible, minimize normalized feasibility gaps first;
    # the smoothness-aware utility is used as the tie-breaker.
    gc = max(float(objectives["min_closed_acc"]), 1e-9)
    go = max(float(objectives["target_open_auc"]), 1e-9)

    def residual_key(r: Dict) -> Tuple[float, float]:
        m = r["control_metrics"]
        gap = max(0.0, objectives["min_closed_acc"] - m["closed_acc"]) / gc
        gap += max(0.0, objectives["target_open_auc"] - m["open_auc"]) / go
        return gap, r["utility"]

    return min(pool, key=residual_key)


def search_thresholds(features: Dict[str, np.ndarray], cfg: dict,
                      objectives: Dict[str, float], guidance: Optional[Dict] = None,
                      previous_config: Optional[Dict[str, float]] = None,
                      grid_override: Optional[Dict[str, List[float]]] = None) -> Tuple[Dict, List[Dict]]:
    """Run cached-score replay and return the selected point and Pareto front."""
    grid = grid_override if grid_override is not None else build_search_grid(cfg, guidance)
    for name, values in grid.items():
        if not values:
            raise ValueError(f"Search grid for {name} is empty after safe-set restriction")

    candidates: List[Dict] = []
    for aq in grid["accept_quantile"]:
        for mq in grid["margin_quantile"]:
            for df in grid["delta_fused"]:
                for dm in grid["delta_margin"]:
                    control_metrics = evaluate_candidate(
                        features, "control", aq, mq, df, dm
                    )
                    candidates.append({
                        "config": {
                            "accept_quantile": float(aq),
                            "margin_quantile": float(mq),
                            "delta_fused": float(df),
                            "delta_margin": float(dm),
                        },
                        "control_metrics": control_metrics,
                        "utility": 0.0,
                    })

    front = pareto_front(candidates)
    selected = select_operating_point(front, objectives, cfg, previous_config)
    c = selected["config"]
    eval_metrics = evaluate_candidate(
        features,
        "eval",
        c["accept_quantile"],
        c["margin_quantile"],
        c["delta_fused"],
        c["delta_margin"],
    )
    best = {
        "config": dict(c),
        "control_metrics": dict(selected["control_metrics"]),
        "metrics": eval_metrics,
        "utility": float(selected["utility"]),
    }
    return best, front

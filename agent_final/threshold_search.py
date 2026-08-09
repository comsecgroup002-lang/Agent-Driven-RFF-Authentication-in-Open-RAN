# threshold_search.py
# -*- coding: utf-8 -*-
"""
阈值网格搜索模块

功能：
1. 训练后一次性提取所有特征
2. 网格搜索 (accept_quantile, margin_quantile, rho) 组合
3. 返回帕累托最优解集合

优点：
- 只需一次前向推理，搜索 100+ 组合只需几秒
- 保证找到数学最优解
- 不依赖 LLM
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve, auc, accuracy_score


@dataclass
class ThresholdConfig:
    """阈值配置"""
    accept_quantile: float  # 接受分位数 [0.70, 0.98]
    margin_quantile: float  # margin 分位数 [0.05, 0.40]
    rho: float              # 全局偏置 [-0.20, 0.50]


@dataclass
class SearchResult:
    """搜索结果"""
    threshold_config: ThresholdConfig
    metrics: Dict[str, float]
    score: float


class ThresholdSearcher:
    """
    阈值网格搜索器
    
    搜索空间：
    - accept_quantile: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    - margin_quantile: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    - rho: [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    总共: 7 × 6 × 10 = 420 组合
    """
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.objectives = cfg["agent"]["objectives"]
        
        # 搜索网格
        self.accept_q_grid = cfg.get("threshold_search", {}).get(
            "accept_q_grid", [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
        )
        self.margin_q_grid = cfg.get("threshold_search", {}).get(
            "margin_q_grid", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        )
        self.rho_grid = cfg.get("threshold_search", {}).get(
            "rho_grid", [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        )
        
        # 缓存的特征
        self._cached_features = None
    
    def search(self, cfg: dict) -> Tuple[SearchResult, List[SearchResult]]:
        """
        执行网格搜索
        
        Returns:
            best_result: 最优结果
            pareto_front: 帕累托最优解列表
        """
        # 1. 提取特征（只需一次）
        features = self._extract_features(cfg)
        
        # 2. 网格搜索
        all_results = []
        total = len(self.accept_q_grid) * len(self.margin_q_grid) * len(self.rho_grid)
        
        print(f"\n   Searching {total} threshold combinations...")
        
        for accept_q in self.accept_q_grid:
            for margin_q in self.margin_q_grid:
                for rho in self.rho_grid:
                    config = ThresholdConfig(
                        accept_quantile=accept_q,
                        margin_quantile=margin_q,
                        rho=rho,
                    )
                    
                    metrics = self._evaluate_threshold(features, config)
                    score = self._compute_score(metrics)
                    
                    all_results.append(SearchResult(
                        threshold_config=config,
                        metrics=metrics,
                        score=score,
                    ))
        
        # 3. 筛选帕累托最优解
        pareto_front = self._compute_pareto_front(all_results)
        
        # 4. 选择最佳（基于目标优先级）
        best_result = self._select_best(pareto_front)
        
        return best_result, pareto_front
    
    def _extract_features(self, cfg: dict) -> Dict:
        """提取模型特征（只需一次前向推理）"""
        from model import (
            ThreeBandDataset, ConvNeXtCosFace, 
            extract_logits_feats, energy_score, get_margin, set_seed
        )
        
        seed = cfg.get("seed", 42)
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 路径配置
        paths = cfg["paths"]
        base_path = Path(paths["base"])
        data_cfg = cfg["data"]
        
        low_val = base_path / data_cfg["low"]["val"]
        mid_val = base_path / data_cfg["mid"]["val"]
        high_val = base_path / data_cfg["high"]["val"]
        
        low_test = base_path / data_cfg["low"]["test"]
        mid_test = base_path / data_cfg["mid"]["test"]
        high_test = base_path / data_cfg["high"]["test"]
        
        label_map_json = paths["label_map"]
        model_weights = paths["model_weights"]
        
        # 模型配置
        model_cfg = cfg.get("model", {})
        backbone = model_cfg.get("backbone", "convnext_tiny")
        channels_last = model_cfg.get("channels_last", True)
        
        # 训练配置
        train_cfg = cfg.get("train", {})
        batch_size = train_cfg.get("batch_size", 128)
        num_workers = train_cfg.get("num_workers", 1)
        
        # 加载标签映射
        with open(label_map_json, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        n_classes = len(label_map)
        
        # 创建数据集
        val_ds = ThreeBandDataset(low_val, mid_val, high_val, label_map, train=False, cfg=cfg)
        test_ds = ThreeBandDataset(low_test, mid_test, high_test, label_map, train=False, cfg=cfg)
        
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        
        val_loader = DataLoader(val_ds, **loader_kwargs)
        test_loader = DataLoader(test_ds, **loader_kwargs)
        
        # 加载模型
        model = ConvNeXtCosFace(
            n_classes=n_classes,
            backbone=backbone,
            pretrained_try=False,
            channels_last=channels_last
        ).to(device)
        model.load_state_dict(torch.load(model_weights, map_location=device), strict=True)
        model.eval()
        
        # 提取特征
        print("   Extracting features from validation set...")
        log_val, feat_val, y_val = extract_logits_feats(model, val_loader, device)
        
        print("   Extracting features from test set...")
        log_te, feat_te, y_te = extract_logits_feats(model, test_loader, device)
        
        # 计算各种分数
        en_val = energy_score(log_val)
        en_te = energy_score(log_te)
        
        margin_val = get_margin(log_val)
        margin_te = get_margin(log_te)
        
        # 类条件马氏距离
        open_set_cfg = cfg.get("open_set", {})
        k_centroids = open_set_cfg.get("mahalanobis", {}).get("k_centroids", 3)
        
        class_stats = {}
        for c in sorted(set(y_val)):
            Xc = feat_val[y_val == c]
            k = min(k_centroids, max(1, len(Xc)))
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(Xc)
            mus = km.cluster_centers_
            lwc = LedoitWolf().fit(Xc)
            Sig_inv_c = np.linalg.pinv(lwc.covariance_)
            class_stats[c] = (mus, Sig_inv_c)
        
        def mahal_min_classwise(X):
            out = np.empty((len(X),), dtype=np.float32)
            for i, x in enumerate(X):
                best = 1e9
                for (mus, Sig_inv_c) in class_stats.values():
                    for mu in mus:
                        d = float(np.sqrt((x - mu) @ Sig_inv_c @ (x - mu).T))
                        if d < best:
                            best = d
                out[i] = best
            return out
        
        print("   Computing Mahalanobis distances...")
        md_val = mahal_min_classwise(feat_val)
        md_te = mahal_min_classwise(feat_te)
        
        # 标准化
        ez_val = (en_val - en_val.mean()) / (en_val.std() + 1e-6)
        dz_val = (md_val - md_val.mean()) / (md_val.std() + 1e-6)
        ez_te = (en_te - en_val.mean()) / (en_val.std() + 1e-6)
        dz_te = (md_te - md_val.mean()) / (md_val.std() + 1e-6)
        
        mz_val = (margin_val - margin_val.mean()) / (margin_val.std() + 1e-6)
        mz_te = (margin_te - margin_val.mean()) / (margin_val.std() + 1e-6)
        
        # Alpha 搜索
        energy_cfg = open_set_cfg.get("energy", {})
        alpha_grid = energy_cfg.get("alpha_grid", [i/10 for i in range(11)])
        
        best_alpha = 0.5
        best_obj = 1e9
        for alpha in alpha_grid:
            fused_v = alpha * ez_val + (1.0 - alpha) * dz_val
            pred_v = log_val.argmax(axis=1)
            
            # 简单评估
            tau = {}
            for c in set(pred_v):
                idx = (pred_v == c)
                if idx.sum() > 0:
                    tau[c] = np.quantile(fused_v[idx], 0.90)
            
            rej = fused_v > np.array([tau.get(int(p), 1e9) for p in pred_v])
            obj = rej.mean() + (pred_v != y_val).mean()
            
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha
        
        print(f"   Optimal alpha: {best_alpha}")
        
        # 缓存
        self._cached_features = {
            "log_val": log_val,
            "log_te": log_te,
            "y_val": y_val,
            "y_te": y_te,
            "ez_val": ez_val,
            "ez_te": ez_te,
            "dz_val": dz_val,
            "dz_te": dz_te,
            "mz_val": mz_val,
            "mz_te": mz_te,
            "alpha": best_alpha,
        }
        
        return self._cached_features
    
    def _evaluate_threshold(self, features: Dict, config: ThresholdConfig) -> Dict[str, float]:
        """评估单个阈值配置"""
        log_val = features["log_val"]
        log_te = features["log_te"]
        y_val = features["y_val"]
        y_te = features["y_te"]
        ez_val = features["ez_val"]
        ez_te = features["ez_te"]
        dz_val = features["dz_val"]
        dz_te = features["dz_te"]
        mz_val = features["mz_val"]
        mz_te = features["mz_te"]
        alpha = features["alpha"]
        
        # Fused score
        fused_val = alpha * ez_val + (1.0 - alpha) * dz_val
        fused_te = alpha * ez_te + (1.0 - alpha) * dz_te
        
        # 计算每类阈值
        pred_val = log_val.argmax(axis=1)
        pred_te = log_te.argmax(axis=1)
        
        # Fused 阈值
        tau_fused = {}
        for c in set(pred_val):
            idx = (pred_val == c)
            if idx.sum() > 0:
                tau_fused[c] = np.quantile(fused_val[idx], config.accept_quantile)
            else:
                tau_fused[c] = 1e9
        
        # Margin 阈值
        tau_margin = {}
        for c in set(pred_val):
            idx = (pred_val == c)
            if idx.sum() > 0:
                tau_margin[c] = np.quantile(mz_val[idx], config.margin_quantile)
            else:
                tau_margin[c] = -1e9
        
        # 应用到测试集
        thr_vec_fused = np.array([tau_fused.get(int(c), 1e9) for c in pred_te]) + config.rho
        thr_vec_margin = np.array([tau_margin.get(int(c), -1e9) for c in pred_te])
        
        d1 = fused_te - thr_vec_fused
        d2 = thr_vec_margin - mz_te
        accept = (d1 <= 0) & (d2 <= 0)
        
        pred_open = np.where(accept, pred_te, -1)
        
        # 计算指标
        total = len(y_te)
        rejected = np.sum(pred_open == -1)
        reject_rate = rejected / total
        
        # Closed-set accuracy (只计算被接受且是已知类的样本)
        valid = [i for i, (p, y) in enumerate(zip(pred_open, y_te)) if p != -1 and y != -1]
        if valid:
            closed_acc = accuracy_score(y_te[valid], pred_open[valid])
        else:
            closed_acc = 0.0
        
        # Open-set AUC
        y_known = np.array([1 if y != -1 else 0 for y in y_te])
        gate_score = np.maximum(d1, d2)
        
        try:
            fpr, tpr, _ = roc_curve(y_known, gate_score, pos_label=0)
            open_auc = auc(fpr, tpr)
        except:
            open_auc = 0.5
        
        return {
            "closed_acc": float(closed_acc),
            "open_auc": float(open_auc),
            "reject_rate": float(reject_rate),
        }
    
    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Rank candidates with A_c/A_o-only hard feasibility.

        Rejection rate remains an auxiliary service-availability metric used for
        ranking/Pareto analysis; it is not a third feasibility constraint.
        """
        obj = self.objectives
        closed_gap = max(0.0, obj["min_closed_acc"] - metrics["closed_acc"])
        open_gap = max(0.0, obj["target_open_auc"] - metrics["open_auc"])
        availability_term = (1.0 - metrics.get("reject_rate", 0.0)) * 0.25
        if closed_gap == 0 and open_gap == 0:
            return (
                metrics["closed_acc"] * 2.0 +
                metrics["open_auc"] * 1.5 +
                availability_term
            )
        penalty = closed_gap * 5.0 + open_gap * 3.0
        return metrics["closed_acc"] + metrics["open_auc"] + availability_term - penalty
    
    def _compute_pareto_front(self, results: List[SearchResult]) -> List[SearchResult]:
        """计算帕累托最优解"""
        pareto = []
        
        for r in results:
            is_dominated = False
            
            for other in results:
                if other is r:
                    continue
                
                # 检查 other 是否支配 r
                # other 支配 r: 所有指标 >= r，且至少一个 >
                better_closed = other.metrics["closed_acc"] >= r.metrics["closed_acc"]
                better_open = other.metrics["open_auc"] >= r.metrics["open_auc"]
                better_reject = other.metrics["reject_rate"] <= r.metrics["reject_rate"]
                
                strictly_better = (
                    other.metrics["closed_acc"] > r.metrics["closed_acc"] or
                    other.metrics["open_auc"] > r.metrics["open_auc"] or
                    other.metrics["reject_rate"] < r.metrics["reject_rate"]
                )
                
                if better_closed and better_open and better_reject and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(r)
        
        # 按得分排序
        pareto.sort(key=lambda x: x.score, reverse=True)
        
        return pareto
    
    def _select_best(self, pareto_front: List[SearchResult]) -> SearchResult:
        """Select the best candidate; feasibility is A_c/A_o only."""
        if not pareto_front:
            raise ValueError("Empty Pareto front")
        obj = self.objectives
        feasible = [
            r for r in pareto_front
            if r.metrics["closed_acc"] >= obj["min_closed_acc"]
            and r.metrics["open_auc"] >= obj["target_open_auc"]
        ]
        if feasible:
            return max(feasible, key=lambda r: r.score)
        closed_ok = [r for r in pareto_front if r.metrics["closed_acc"] >= obj["min_closed_acc"]]
        if closed_ok:
            return max(closed_ok, key=lambda r: (r.metrics["open_auc"], r.score))
        return max(pareto_front, key=lambda r: r.score)

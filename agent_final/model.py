# -*- coding: utf-8 -*-

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 问题
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _build_label_map_from_triple(mid_root: Path, save_path: str, num_known_devices: int = None) -> Dict[str, int]:

    if num_known_devices is not None:
        # 使用 config 中指定的已知设备数量，确保标签映射一致
        mp = {f"Device{i}": i - 1 for i in range(1, num_known_devices + 1)}
        print(f"[LabelMap] Using config: {num_known_devices} known devices")
    else:
        # 回退：从目录扫描（不推荐，可能导致不一致）
        names = sorted([d.name for d in mid_root.iterdir() if d.is_dir()], key=natural_key)
        mp = {c: i for i, c in enumerate(names)}
        print(f"[LabelMap] Scanned from directory: {len(mp)} classes")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2, ensure_ascii=False)
    
    print(f"[LabelMap] Saved to {save_path}: {mp}")
    return mp


# ============================================================================
# 三路灰度数据集（对称填充到共同最大尺寸）
# ============================================================================
class ThreeBandDataset(Dataset):

    def __init__(self, root_low: Path, root_mid: Path, root_high: Path,
                 label_map: Dict[str, int], train: bool = True, cfg: dict = None,
                 filter_unknown: bool = False):

        self.root_low = Path(root_low)
        self.root_mid = Path(root_mid)
        self.root_high = Path(root_high)
        self.train = train
        self.label_map = label_map
        self.cfg = cfg or {}
        self.filter_unknown = filter_unknown
        self.samples: List[Tuple[str, str, str, int]] = []
        
        # 从 config 读取参数
        model_cfg = self.cfg.get("model", {})
        self.img_size = model_cfg.get("img_size", 224)
        
        # 增强参数
        aug_cfg = self.cfg.get("augment", {})
        self.aug_enable = aug_cfg.get("enable", True) and train
        self.aug_strength = aug_cfg.get("aug_strength", 0.35)
        self.spec_fmask_p = aug_cfg.get("spec_fmask_p", 0.03)
        self.spec_tmask_p = aug_cfg.get("spec_tmask_p", 0.03)
        
        # 弱几何变换参数
        self.weak_geo_max_angle = aug_cfg.get("weak_geo_max_angle", 2.0)
        self.weak_geo_translate = aug_cfg.get("weak_geo_translate", 0.03)
        self.weak_geo_scale = aug_cfg.get("weak_geo_scale", 0.03)

        if not (self.root_low.exists() and self.root_mid.exists() and self.root_high.exists()):
            raise FileNotFoundError("One of the three-band roots does not exist.")

        # 统计
        total_samples = 0
        filtered_samples = 0
        
        for cname in sorted([d.name for d in self.root_mid.iterdir() if d.is_dir()], key=natural_key):
            y = self.label_map.get(cname, -1)
            low_dir = self.root_low / cname
            mid_dir = self.root_mid / cname
            high_dir = self.root_high / cname
            if not (low_dir.exists() and mid_dir.exists() and high_dir.exists()):
                continue
            for p_mid in sorted(mid_dir.glob("*.png"), key=lambda x: natural_key(x.name)):
                p_low = low_dir / p_mid.name
                p_high = high_dir / p_mid.name
                if p_low.exists() and p_high.exists():
                    total_samples += 1
                    # 如果需要过滤未知标签，跳过 label=-1 的样本
                    if self.filter_unknown and y == -1:
                        filtered_samples += 1
                        continue
                    self.samples.append((str(p_low), str(p_mid), str(p_high), y))
        
        if filter_unknown and filtered_samples > 0:
            print(f"[Dataset] Filtered {filtered_samples}/{total_samples} unknown samples, kept {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_gray(path: str) -> torch.Tensor:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)  # (H,W)
        t = (t - t.mean()) / (t.std() + 1e-6)  # per-image z-score
        return t

    @staticmethod
    def _pad_to_center(x: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        H, W = x.shape
        out = torch.zeros((Ht, Wt), dtype=x.dtype)
        sh = (Ht - H) // 2
        sw = (Wt - W) // 2
        out[sh:sh + H, sw:sw + W] = x
        return out

    def _stack3(self, pL: str, pM: str, pH: str) -> torch.Tensor:
        L = self._read_gray(pL)
        M = self._read_gray(pM)
        Ht = self._read_gray(pH)
        Hmax = int(max(L.shape[0], M.shape[0], Ht.shape[0]))
        Wmax = int(max(L.shape[1], M.shape[1], Ht.shape[1]))
        Lp = self._pad_to_center(L, Hmax, Wmax)
        Mp = self._pad_to_center(M, Hmax, Wmax)
        Hp = self._pad_to_center(Ht, Hmax, Wmax)
        return torch.stack([Lp, Mp, Hp], dim=0)  # [3,Hmax,Wmax]

    def _weak_geo(self, x: torch.Tensor) -> torch.Tensor:
        """弱几何变换 - 参数从 config 读取"""
        if not self.train or not self.aug_enable:
            return x
        H, W = x.shape[-2:]
        angle = float((torch.rand(1) * 2 - 1) * self.weak_geo_max_angle)
        tx = int(W * self.weak_geo_translate * (torch.rand(1) * 2 - 1))
        ty = int(H * self.weak_geo_translate * (torch.rand(1) * 2 - 1))
        scale = float(1.0 + (torch.rand(1) * 2 - 1) * self.weak_geo_scale)
        return TF.affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """SpecAugment - 参数从 config 读取"""
        if not self.train or not self.aug_enable:
            return x
        
        # 频率掩码
        if torch.rand(1).item() < self.spec_fmask_p:
            Fm = max(1, int(x.shape[-2] * self.aug_strength * 0.2))
            f0 = torch.randint(0, max(1, x.shape[-2] - Fm), (1,)).item()
            x[:, f0:f0 + Fm, :] = 0
        
        # 时间掩码
        if torch.rand(1).item() < self.spec_tmask_p:
            Tm = max(1, int(x.shape[-1] * self.aug_strength * 0.25))
            t0 = torch.randint(0, max(1, x.shape[-1] - Tm), (1,)).item()
            x[:, :, t0:t0 + Tm] = 0
        return x

    def __getitem__(self, idx):
        pL, pM, pH, y = self.samples[idx]
        x = self._stack3(pL, pM, pH)
        if self.train and self.aug_enable:
            x = self._weak_geo(x)
            x = self._spec_augment(x)
        x = TF.resize(x, [self.img_size, self.img_size], antialias=True)
        return x, y


# ============================================================================
# CosFace 头
# ============================================================================
class CosFace(nn.Module):
    def __init__(self, in_feats, n_classes, s=30.0, m=0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_feats))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m

    def forward(self, feats, labels=None):
        x = F.normalize(feats)
        w = F.normalize(self.W)
        cos = F.linear(x, w)  # (B,C)
        if labels is None:
            return self.s * cos
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (cos - one_hot * self.m)
        return logits


# ============================================================================
# 预训练权重路径映射
# ============================================================================
PRETRAINED_WEIGHTS_DIR = "/set/your/path/pretrained_models"

# backbone 名称到本地权重文件的映射
PRETRAINED_WEIGHTS_MAP = {
    "convnext_tiny": "convnext_tiny_pretrained.pth",
    "convnext_small": "convnext_small_pretrained.pth",
    "convnext_base": "convnext_base_pretrained.pth",
    "convnext_tiny.fb_in22k": "convnext_tiny_22k_pretrained.pth",
    "convnext_small.fb_in22k": "convnext_small_22k_pretrained.pth",
    "convnext_base.fb_in22k": "convnext_base_22k_pretrained.pth",
}


def load_pretrained_weights(model: nn.Module, backbone_name: str) -> bool:

    # 查找对应的权重文件
    weights_file = PRETRAINED_WEIGHTS_MAP.get(backbone_name)
    
    if weights_file is None:
        # 尝试模糊匹配
        for key in PRETRAINED_WEIGHTS_MAP:
            if backbone_name in key or key in backbone_name:
                weights_file = PRETRAINED_WEIGHTS_MAP[key]
                break
    
    if weights_file is None:
        print(f"[WARN] No pretrained weights mapping for: {backbone_name}")
        return False
    
    weights_path = os.path.join(PRETRAINED_WEIGHTS_DIR, weights_file)
    
    if not os.path.exists(weights_path):
        print(f"[WARN] Pretrained weights not found: {weights_path}")
        return False
    
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # 处理可能的 state_dict 包装
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # 加载权重（允许部分匹配）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"[INFO] Loaded pretrained weights from: {weights_path}")
        print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        return True
        
    except Exception as e:
        print(f"[WARN] Failed to load pretrained weights: {e}")
        return False


# ============================================================================
# Backbone + BNNeck + CosFace
# ============================================================================
class ConvNeXtCosFace(nn.Module):
    def __init__(self, n_classes, backbone="convnext_tiny", embed_dim=256, 
                 pretrained_try=True, channels_last=True, 
                 pretrained_dir=None):

        super().__init__()
        self.channels_last = channels_last
        
        # 先创建模型（不加载 timm 的在线预训练）
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=3, num_classes=0)
        
        # 从本地加载预训练权重
        if pretrained_try:
            if pretrained_dir:
                global PRETRAINED_WEIGHTS_DIR
                PRETRAINED_WEIGHTS_DIR = pretrained_dir
            
            loaded = load_pretrained_weights(self.backbone, backbone)
            if not loaded:
                print(f"[INFO] Using randomly initialized weights for {backbone}")

        feat_dim = int(getattr(self.backbone, "num_features", 768))
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.head = CosFace(feat_dim, n_classes, s=30.0, m=0.25)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x, labels=None, return_feats=False):
        if self.channels_last and x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        f = self.backbone(x)  # (B,feat_dim)
        f = self.bnneck(f)
        logits = self.head(f, labels)  # (B,C)
        if return_feats:
            return logits, f
        return logits


# ============================================================================
# 工具函数
# ============================================================================
def cosine_lr(optimizer, epoch, base_lr=3e-4, min_lr=1e-5, warmup=3, max_epochs=60):
    if epoch < warmup:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup))
    else:
        t = (epoch - warmup) / float(max(1, max_epochs - warmup))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def extract_logits_feats(model, loader, device):
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits, feats = model(x, labels=None, return_feats=True)
        all_logits.append(logits.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return np.vstack(all_logits), np.vstack(all_feats), np.array(all_labels)


def energy_score(logits, T=1.0):
    """Energy score (越大越 OOD)"""
    x = logits / T
    m = x.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    return (-T * lse).ravel()


def get_margin(logits):
    """获取 top1-top2 margin"""
    part = np.partition(logits, -2, axis=1)
    top2 = part[:, -2]
    top1 = logits.max(axis=1)
    return top1 - top2


# ============================================================================
# 训练函数 - 从 config 读取所有参数
# ============================================================================
def train(cfg: dict):
    """
    训练模型，所有参数从 cfg 读取
    
    Args:
        cfg: 配置字典，包含 paths, train, model, augment 等配置
    """
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 路径配置
    paths = cfg["paths"]
    base_path = Path(paths["base"])
    data_cfg = cfg["data"]
    
    low_train = base_path / data_cfg["low"]["train"]
    mid_train = base_path / data_cfg["mid"]["train"]
    high_train = base_path / data_cfg["high"]["train"]
    
    low_val = base_path / data_cfg["low"]["val"]
    mid_val = base_path / data_cfg["mid"]["val"]
    high_val = base_path / data_cfg["high"]["val"]
    
    label_map_json = paths["label_map"]
    model_weights = paths["model_weights"]
    
    # 模型配置
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "convnext_tiny")
    channels_last = model_cfg.get("channels_last", True)
    pretrained_dir = model_cfg.get("pretrained_dir", "/home/wangyb/pretrained_models")
    
    # 训练配置
    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 128)
    max_epochs = train_cfg.get("max_epochs", 60)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    base_lr = train_cfg.get("base_lr", 3e-4)
    min_lr = train_cfg.get("min_lr", 1e-5)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    early_patience = train_cfg.get("early_patience", 8)
    num_workers = train_cfg.get("num_workers", 1)
    pin_memory = train_cfg.get("pin_memory", True)
    persistent_workers = train_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    use_bf16 = train_cfg.get("use_bf16", True)
    
    # 从 config 获取已知设备数量
    dataset_split_cfg = cfg.get("dataset_split", {})
    num_known_devices = dataset_split_cfg.get("num_known_devices", 10)
    
    # 构建标签映射 - 使用 config 中的 num_known_devices 确保一致性
    label_map = _build_label_map_from_triple(mid_train, label_map_json, num_known_devices=num_known_devices)
    n_classes = len(label_map)
    print(f"[Train] Classes: {n_classes}, Device: {device}")
    
    # 创建数据集
    # - train: 过滤未知标签，只训练已知类别
    # - val: 过滤未知标签，只在已知类别上计算验证损失
    train_ds = ThreeBandDataset(low_train, mid_train, high_train, label_map, 
                                 train=True, cfg=cfg, filter_unknown=True)
    val_ds = ThreeBandDataset(low_val, mid_val, high_val, label_map, 
                               train=False, cfg=cfg, filter_unknown=True)
    
    print(f"[Train] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # 安全检查：确保数据集不为空
    if len(train_ds) == 0:
        raise ValueError("[Train] Training dataset is empty! Check your data paths and label_map.")
    if len(val_ds) == 0:
        print("[WARN] Validation dataset is empty, will skip validation.")
    
    # DataLoader 参数
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    # 创建模型（从本地加载预训练权重）
    model = ConvNeXtCosFace(
        n_classes=n_classes, 
        backbone=backbone, 
        pretrained_try=True,
        channels_last=channels_last,
        pretrained_dir=pretrained_dir
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # 混合精度
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not use_bf16)
    autocast_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float16
    
    best_val = 1e9
    bad = 0
    
    for epoch in range(max_epochs):
        model.train()
        lr = cosine_lr(optimizer, epoch, base_lr, min_lr, warmup_epochs, max_epochs)
        running = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            
            # 安全检查：训练数据不应包含无效标签
            if (y < 0).any() or (y >= n_classes).any():
                invalid = y[(y < 0) | (y >= n_classes)].unique().tolist()
                raise ValueError(f"[Train] Invalid labels in training batch: {invalid}. "
                               f"Expected [0, {n_classes-1}]. Check filter_unknown setting.")
            
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                logits = model(x, labels=y)
                loss = F.cross_entropy(logits, y)
            
            if autocast_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            n_batches += 1
        
        # 验证
        model.eval()
        val_loss, tot = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)
                
                # 安全检查：验证数据也不应包含无效标签（因为已经过滤）
                if (y < 0).any() or (y >= n_classes).any():
                    invalid = y[(y < 0) | (y >= n_classes)].unique().tolist()
                    raise ValueError(f"[Val] Invalid labels in validation batch: {invalid}. "
                                   f"Expected [0, {n_classes-1}]. Check filter_unknown setting.")
                
                logits = model(x, labels=y)
                val_loss += float(F.cross_entropy(logits, y).item()) * x.size(0)
                tot += x.size(0)
        
        val_loss = val_loss / max(1, tot)
        print(f"[{epoch + 1:03d}] train_ce={running / max(1, n_batches):.4f}  val_ce={val_loss:.4f}  lr={lr:.2e}")
        
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), model_weights)
            print(f"  ↳ saved new best weights: {model_weights}")
        else:
            bad += 1
            if bad >= early_patience:
                print(f"Early stopping. best val_ce={best_val:.4f}")
                break
    
    if not os.path.exists(model_weights):
        torch.save(model.state_dict(), model_weights)
    print(f"[Train] Finished. Weights saved to {model_weights}")


# ============================================================================
# 评估函数 - 从 config 读取所有参数，返回 metrics 字典
# ============================================================================
def evaluate(cfg: dict, save_plots: bool = False) -> dict:
    """
    评估模型，返回 metrics 字典
    
    Args:
        cfg: 配置字典
        save_plots: 是否保存图表
        
    Returns:
        dict: 包含 closed_acc, open_auc, reject_rate 等指标
    """
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
    
    # 训练配置 (用于 DataLoader)
    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 128)
    num_workers = train_cfg.get("num_workers", 1)
    pin_memory = train_cfg.get("pin_memory", True)
    persistent_workers = train_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    
    # 开集配置
    open_set_cfg = cfg.get("open_set", {})
    energy_cfg = open_set_cfg.get("energy", {})
    calib_cfg = open_set_cfg.get("calibration", {})
    mahal_cfg = open_set_cfg.get("mahalanobis", {})
    
    alpha_init = energy_cfg.get("alpha_init", 0.6)
    auto_alpha = energy_cfg.get("auto_alpha", True)
    alpha_grid = energy_cfg.get("alpha_grid", [i / 10 for i in range(0, 11)])
    
    accept_quantile = calib_cfg.get("accept_quantile", 0.90)
    margin_quantile = calib_cfg.get("margin_quantile", 0.20)
    rho = calib_cfg.get("rho", 0.10)
    
    k_centroids = mahal_cfg.get("k_centroids", 3)
    
    # 加载标签映射
    with open(label_map_json, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    n_classes = len(label_map)
    
    # 创建数据集
    # - val: 过滤未知标签（用于阈值校准，只用已知类别）
    # - test: 不过滤（包含未知设备，用于开集评估）
    val_ds = ThreeBandDataset(low_val, mid_val, high_val, label_map, 
                               train=False, cfg=cfg, filter_unknown=True)
    test_ds = ThreeBandDataset(low_test, mid_test, high_test, label_map, 
                                train=False, cfg=cfg, filter_unknown=False)
    
    print(f"[Eval] Val samples (known only): {len(val_ds)}, Test samples (all): {len(test_ds)}")
    
    # DataLoader 参数
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
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
    
    # 1) 提取 val/test 的 logits + feats
    log_val, feat_val, y_val = extract_logits_feats(model, val_loader, device)
    log_te, feat_te, y_te = extract_logits_feats(model, test_loader, device)
    
    # 2) Energy（越大越 OOD）
    en_val = energy_score(log_val)
    en_te = energy_score(log_te)
    
    # 3) 类条件马氏距离
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
    
    md_val = mahal_min_classwise(feat_val)
    md_te = mahal_min_classwise(feat_te)
    
    # 4) 标准化
    ez_val = (en_val - en_val.mean()) / (en_val.std() + 1e-6)
    dz_val = (md_val - md_val.mean()) / (md_val.std() + 1e-6)
    ez_te = (en_te - en_val.mean()) / (en_val.std() + 1e-6)
    dz_te = (md_te - md_val.mean()) / (md_val.std() + 1e-6)
    
    # 5) α 搜索
    def objective(alpha):
        fused_v = alpha * ez_val + (1.0 - alpha) * dz_val
        pred_v = log_val.argmax(axis=1)
        tau = {}
        for c in set(pred_v):
            idx = (pred_v == c)
            tau[c] = np.quantile(fused_v[idx], accept_quantile) if idx.sum() > 0 else 1e9
        rej = fused_v > np.array([tau.get(int(p), 1e9) for p in pred_v])
        rej_rate = rej.mean()
        miscls_rate = (pred_v != y_val).mean()
        return rej_rate + miscls_rate, tau
    
    alpha_opt = alpha_init
    tau_opt = {}
    if auto_alpha:
        best_obj = 1e9
        for a in alpha_grid:
            obj, tau = objective(a)
            if obj < best_obj:
                best_obj = obj
                alpha_opt, tau_opt = a, tau
    else:
        _, tau_opt = objective(alpha_init)
    
    # 6) margin 门控
    margin_val = get_margin(log_val)
    margin_te = get_margin(log_te)
    mz_val = (margin_val - margin_val.mean()) / (margin_val.std() + 1e-6)
    mz_te = (margin_te - margin_val.mean()) / (margin_val.std() + 1e-6)
    
    pred_v = log_val.argmax(axis=1)
    tau_margin = {}
    for c in set(pred_v):
        idx = (pred_v == c)
        tau_margin[c] = np.quantile(mz_val[idx], margin_quantile) if idx.sum() > 0 else -1e9
    
    # 7) threshold-free 诊断
    fused_te = alpha_opt * ez_te + (1.0 - alpha_opt) * dz_te
    y_known = np.array([1 if y != -1 else 0 for y in y_te])
    fpr_f, tpr_f, _ = roc_curve(y_known, fused_te, pos_label=0)
    auc_f = auc(fpr_f, tpr_f)
    
    # 8) 两级门控
    pred_closed = log_te.argmax(axis=1)
    thr_vec_fused = np.array([tau_opt.get(int(c), 1e9) for c in pred_closed]) + rho
    thr_vec_margin = np.array([tau_margin.get(int(c), -1e9) for c in pred_closed])
    
    d1 = fused_te - thr_vec_fused
    d2 = thr_vec_margin - mz_te
    accept_known = (d1 <= 0) & (d2 <= 0)
    pred_open = np.where(accept_known, pred_closed, -1)
    
    # 9) Gate-ROC
    gate_score = np.maximum(d1, d2)
    fpr_g, tpr_g, _ = roc_curve(y_known, gate_score, pos_label=0)
    auc_g = auc(fpr_g, tpr_g)
    
    # 10) 计算最终指标
    total = len(y_te)
    rejected = np.sum(pred_open == -1)
    reject_rate = rejected / total
    
    valid = [i for i, (p, y) in enumerate(zip(pred_open, y_te)) if p != -1 and y != -1]
    if valid:
        closed_acc = accuracy_score(y_te[valid], pred_open[valid])
    else:
        closed_acc = 0.0
    
    # 保存图表（可选）
    if save_plots:
        output_dir = Path(cfg.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩阵
        labels_all = sorted(list(set(y_te) | set(pred_open)))
        cm = confusion_matrix(y_te, pred_open, labels=labels_all)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_all)
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        ax.set_title("Open-set Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        
        # ROC 曲线
        fig, ax = plt.subplots()
        ax.plot(fpr_f, tpr_f, label=f"fused AUC={auc_f:.3f}")
        ax.plot(fpr_g, tpr_g, label=f"gate  AUC={auc_g:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR (unknown accepted)")
        ax.set_ylabel("TPR (unknown rejected)")
        ax.set_title("ROC comparison: fused vs. two-stage gate")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close()
    
    metrics = {
        "closed_acc": float(closed_acc),
        "open_auc": float(auc_g),  # 使用 gate AUC 作为 open_auc
        "reject_rate": float(reject_rate),
        "fused_auc": float(auc_f),
        "alpha_opt": float(alpha_opt),
    }
    
    print(f"[Eval] closed_acc={closed_acc:.4f}, open_auc={auc_g:.4f}, reject_rate={reject_rate:.4f}")
    
    return metrics


# ============================================================================
# 入口 - 用于独立运行
# ============================================================================
if __name__ == "__main__":
    import yaml
    
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass
    
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get("seed", 42))
    
    train(cfg)
    metrics = evaluate(cfg, save_plots=True)
    print(f"\nFinal metrics: {metrics}")

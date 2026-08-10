# data_pipeline.py
# -*- coding: utf-8 -*-


import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import yaml

from signal_processing import (
    SignalAugmentConfig, ProcessingConfig,
    run_signal_processing
)


@dataclass
class DatasetSplitConfig:
    """数据集划分配置"""
    train_window_start: int = 1
    train_window_end: int = 800
    test_window_start: int = 801
    test_window_end: int = 900
    val_window_start: int = 901
    val_window_end: int = 1000
    num_known_devices: int = 10
    total_devices: int = 11
    iq_file_start: int = 1
    iq_file_end: int = 10


class DataPipeline:
    """数据流水线"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.data_cfg = cfg["data"]

        self.proc_config = self._parse_proc_config(cfg)
        self.aug_config = self._parse_aug_config(cfg)
        self.split_config = self._parse_split_config(cfg)

    def _parse_proc_config(self, cfg: dict) -> ProcessingConfig:
        """解析处理配置"""
        sp_cfg = cfg.get("signal_processing", {})
        return ProcessingConfig(
            sample_count=sp_cfg.get("sample_count", int(2e7)),
            sample_rate=sp_cfg.get("sample_rate", 1e6),
            window_size=sp_cfg.get("window_size", 8192),
            stride=sp_cfg.get("stride", 8192),
            wavelet_type=sp_cfg.get("wavelet_type", "db4"),
            stft_win_size=sp_cfg.get("stft_win_size", 256),
            stft_overlap=sp_cfg.get("stft_overlap", 200),
            stft_nfft=sp_cfg.get("stft_nfft", 512),
            img_dpi=sp_cfg.get("img_dpi", 150),
        )

    def _parse_aug_config(self, cfg: dict) -> SignalAugmentConfig:
        """解析增强配置"""
        sa_cfg = cfg.get("signal_augment", {})

        return SignalAugmentConfig(
            retain_ratio=sa_cfg.get("retain_ratio", 0.5),
            channel_aug_prob=sa_cfg.get("channel_aug_prob", 1.0),
            engineering_aug_prob=sa_cfg.get("engineering_aug_prob", 0.7),
            drift_prob=sa_cfg.get("drift_prob", 0.05),
            shift_prob=sa_cfg.get("shift_prob", 0.10),
            tau_rms_min_ns=sa_cfg.get("tau_rms_min_ns", 20.0),
            tau_rms_max_ns=sa_cfg.get("tau_rms_max_ns", 150.0),
            doppler_min_hz=sa_cfg.get("doppler_min_hz", 0.1),
            doppler_max_hz=sa_cfg.get("doppler_max_hz", 5.0),
            k_factor_min_db=sa_cfg.get("k_factor_min_db", 3.0),
            k_factor_max_db=sa_cfg.get("k_factor_max_db", 10.0),
            snr_min_db=sa_cfg.get("snr_min_db", 30.0),
            snr_max_db=sa_cfg.get("snr_max_db", 40.0),
            gain_std=sa_cfg.get("gain_std", 0.05),
            nonlinear_coef=sa_cfg.get("nonlinear_coef", 0.003),
            iq_amplitude_std=sa_cfg.get("iq_amplitude_std", 0.02),
            iq_phase_std_deg=sa_cfg.get("iq_phase_std_deg", 1.0),
            max_time_shift=sa_cfg.get("max_time_shift", 2),
            drift_ppm_min=sa_cfg.get("drift_ppm_min", 300),
            drift_ppm_max=sa_cfg.get("drift_ppm_max", 1000),
        )

    def _parse_split_config(self, cfg: dict) -> DatasetSplitConfig:
        """解析划分配置"""
        ds_cfg = cfg.get("dataset_split", {})
        return DatasetSplitConfig(
            train_window_start=ds_cfg.get("train_window_start", 1),
            train_window_end=ds_cfg.get("train_window_end", 800),
            test_window_start=ds_cfg.get("test_window_start", 801),
            test_window_end=ds_cfg.get("test_window_end", 900),
            val_window_start=ds_cfg.get("val_window_start", 901),
            val_window_end=ds_cfg.get("val_window_end", 1000),
            num_known_devices=ds_cfg.get("num_known_devices", 10),
            total_devices=ds_cfg.get("total_devices", 11),
            iq_file_start=ds_cfg.get("iq_file_start", 1),
            iq_file_end=ds_cfg.get("iq_file_end", 10),
        )

    def check_raw_data(self, raw_dir: str) -> Tuple[bool, List[str]]:
        """检查原始数据"""
        if not raw_dir:
            return False, ["raw_dir not configured"]

        raw_path = Path(raw_dir)
        if not raw_path.exists():
            return False, [f"Directory not found: {raw_dir}"]

        missing = []
        for device_idx in range(1, self.split_config.total_devices + 1):
            device_dir = raw_path / f"Device{device_idx}"

            if not device_dir.exists():
                missing.append(str(device_dir))
                continue

            for file_idx in range(self.split_config.iq_file_start,
                                  self.split_config.iq_file_end + 1):
                dat_file = device_dir / f"IQ_{file_idx}.dat"
                if not dat_file.exists():
                    missing.append(str(dat_file))

        return len(missing) == 0, missing

    def check_processed_data(self, processed_dir: str) -> Tuple[bool, Dict[str, int]]:
        """检查处理后的数据（宽松检查，只看是否有内容）"""
        if not processed_dir:
            return False, {}

        proc_path = Path(processed_dir)
        if not proc_path.exists():
            return False, {}

        components = ['cA2', 'cD1', 'cD2']
        stats = {}
        has_data = False

        for device_idx in range(1, self.split_config.total_devices + 1):
            device_name = f"Device{device_idx}"

            for comp in components:
                comp_dir = proc_path / comp / device_name
                if comp_dir.exists():
                    png_count = len(list(comp_dir.glob("*.png")))
                    stats[f"{device_name}/{comp}"] = png_count
                    if png_count > 0:
                        has_data = True
                else:
                    stats[f"{device_name}/{comp}"] = 0

        return has_data, stats

    def check_split_data(self, target_dir: str) -> Tuple[bool, Dict[str, int]]:
        """检查划分后的数据集"""
        if not target_dir:
            return False, {}

        target_path = Path(target_dir)
        if not target_path.exists():
            return False, {}

        components = ['cA2', 'cD1', 'cD2']
        splits = ['train', 'val', 'test']
        stats = {}

        for comp in components:
            for split in splits:
                split_dir = target_path / comp / split
                if split_dir.exists():
                    png_count = len(list(split_dir.rglob("*.png")))
                    stats[f"{comp}/{split}"] = png_count
                else:
                    stats[f"{comp}/{split}"] = 0

        # Basic completeness plus the manuscript open-set protocol:
        # Device11 (the held-out unknown device) must be absent from source
        # train/validation and present in the open-set test split.
        complete = all(v > 0 for v in stats.values()) if stats else False
        unknown_name = f"Device{self.split_config.total_devices}"
        if complete and self.split_config.total_devices > self.split_config.num_known_devices:
            for comp in components:
                if (target_path / comp / "train" / unknown_name).exists():
                    complete = False
                if (target_path / comp / "val" / unknown_name).exists():
                    complete = False
                unknown_test = target_path / comp / "test" / unknown_name
                if not unknown_test.exists() or not any(unknown_test.glob("*.png")):
                    complete = False

        return complete, stats

    def run_signal_processing(self, raw_dir: str, output_dir: str,
                              num_workers: int = 4) -> Dict[str, int]:
        """运行信号处理"""
        print("\n" + "=" * 60)
        print("SIGNAL PROCESSING")
        print("=" * 60)

        print(f"\nSource: {raw_dir}")
        print(f"Output: {output_dir}")
        print(f"\nAugmentation config:")
        print(f"  retain_ratio: {self.aug_config.retain_ratio}")
        print(f"  channel_aug_prob: {self.aug_config.channel_aug_prob}")
        print(f"  engineering_aug_prob: {self.aug_config.engineering_aug_prob}")

        stats = run_signal_processing(
            base_dir=raw_dir,
            output_dir=output_dir,
            proc_config=self.proc_config,
            aug_config=self.aug_config,
            device_range=(1, self.split_config.total_devices),
            file_range=(self.split_config.iq_file_start, self.split_config.iq_file_end),
            num_workers=num_workers,
        )

        print(f"\n✅ Signal processing completed!")
        for device, count in sorted(stats.items()):
            print(f"   {device}: {count} windows")

        return stats

    def split_dataset(self, source_dir: str, target_dir: str) -> Dict[str, int]:
        """划分数据集"""
        print("\n" + "=" * 60)
        print("DATASET SPLITTING")
        print("=" * 60)

        source_path = Path(source_dir)
        target_path = Path(target_dir)

        components = ['cA2', 'cD1', 'cD2']
        sc = self.split_config

        splits = {
            'train': (sc.train_window_start, sc.train_window_end),
            'test': (sc.test_window_start, sc.test_window_end),
            'val': (sc.val_window_start, sc.val_window_end),
        }

        stats = {}

        # Rebuild the split directories from scratch so a previous legacy split
        # cannot leave the unknown device in train/validation.
        for comp in components:
            for split_name in splits:
                split_dir = target_path / comp / split_name
                if split_dir.exists():
                    shutil.rmtree(split_dir)

        for comp in components:
            for split_name, (win_start, win_end) in splits.items():
                # Manuscript protocol: train/validation contain enrolled devices only;
                # the held-out unknown device appears exclusively in open-set testing.
                if split_name == 'test':
                    device_range = range(1, sc.total_devices + 1)
                else:
                    device_range = range(1, sc.num_known_devices + 1)

                split_stats = 0

                for device_idx in device_range:
                    device_name = f"Device{device_idx}"

                    src_dir = source_path / comp / device_name
                    dst_dir = target_path / comp / split_name / device_name
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    if not src_dir.exists():
                        print(f"Warning: {src_dir} does not exist")
                        continue

                    for iq_idx in range(sc.iq_file_start, sc.iq_file_end + 1):
                        for win_idx in range(win_start, win_end + 1):
                            filename = f"IQ{iq_idx:02d}_win{win_idx:03d}.png"
                            src_file = src_dir / filename
                            dst_file = dst_dir / filename

                            if src_file.exists():
                                shutil.copy2(src_file, dst_file)
                                split_stats += 1

                stats[f"{comp}/{split_name}"] = split_stats
                print(f"   {comp}/{split_name}: {split_stats} files")

        print(f"\n✅ Dataset splitting completed!")
        return stats

    def generate_label_map(self, target_dir: str, output_file: str) -> Dict[str, int]:
        """生成标签映射文件"""
        label_map = {}

        for device_idx in range(1, self.split_config.num_known_devices + 1):
            device_name = f"Device{device_idx}"
            label_map[device_name] = device_idx - 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2)

        print(f"✅ Label map saved to {output_file}")
        print(f"   Known devices: {len(label_map)}")

        return label_map

    def prepare_dataset(self, raw_dir: str, processed_dir: str,
                        target_dir: str, label_map_file: str,
                        force_reprocess: bool = False,
                        num_workers: int = 4) -> bool:

        print("\n" + "=" * 60)
        print("DATA PREPARATION PIPELINE")
        print("=" * 60)

        # ===== Step 1: 检查最终数据集 (base) =====
        print("\n[Step 1] Checking final dataset (base)...")
        print(f"   Path: {target_dir}")

        if not force_reprocess:
            split_complete, split_stats = self.check_split_data(target_dir)

            if split_complete:
                print(f"✅ Final dataset found and complete!")
                total = sum(split_stats.values())
                print(f"   Total images: {total}")
                for key, count in sorted(split_stats.items()):
                    print(f"     {key}: {count}")

                # 确保标签映射存在
                if not os.path.exists(label_map_file):
                    print(f"\n[Step 2] Generating label map...")
                    self.generate_label_map(target_dir, label_map_file)
                else:
                    print(f"✅ Label map exists: {label_map_file}")

                print("\n" + "=" * 60)
                print("✅ DATA READY - Using existing dataset")
                print("=" * 60)
                return True
            else:
                print(f"⚠️ Final dataset incomplete or not found")
                if split_stats:
                    print(f"   Current stats: {split_stats}")

        # ===== Step 2: 检查中间数据 (processed_data) =====
        print("\n[Step 2] Checking processed data...")
        print(f"   Path: {processed_dir}")

        proc_exists, proc_stats = self.check_processed_data(processed_dir)

        if proc_exists and not force_reprocess:
            print(f"✅ Processed data found!")
            total = sum(proc_stats.values())
            print(f"   Total images: {total}")

            # 只需要划分
            print("\n[Step 3] Splitting dataset...")
            self.split_dataset(processed_dir, target_dir)

            print("\n[Step 4] Generating label map...")
            self.generate_label_map(target_dir, label_map_file)

            print("\n" + "=" * 60)
            print("✅ DATA PREPARATION COMPLETED!")
            print("=" * 60)
            return True
        else:
            print(f"⚠️ Processed data not found (this is OK if we have raw data)")

        # ===== Step 3: 检查原始数据 (raw_data) =====
        print("\n[Step 3] Checking raw data...")
        print(f"   Path: {raw_dir}")

        raw_exists, missing = self.check_raw_data(raw_dir)

        if not raw_exists:
            print(f"❌ Raw data not found or incomplete!")
            print(f"   Missing: {len(missing)} items")
            for m in missing[:5]:
                print(f"     - {m}")
            if len(missing) > 5:
                print(f"     ... and {len(missing) - 5} more")

            print("\n" + "=" * 60)
            print("❌ CANNOT PREPARE DATA")
            print("   Please provide one of:")
            print(f"   1. Final dataset at: {target_dir}")
            print(f"   2. Processed data at: {processed_dir}")
            print(f"   3. Raw IQ data at: {raw_dir}")
            print("=" * 60)
            return False

        print(f"✅ Raw data found!")

        # ===== Step 4: 运行信号处理 =====
        print("\n[Step 4] Running signal processing...")
        self.run_signal_processing(raw_dir, processed_dir, num_workers)

        # ===== Step 5: 划分数据集 =====
        print("\n[Step 5] Splitting dataset...")
        self.split_dataset(processed_dir, target_dir)

        # ===== Step 6: 生成标签映射 =====
        print("\n[Step 6] Generating label map...")
        self.generate_label_map(target_dir, label_map_file)

        print("\n" + "=" * 60)
        print("✅ DATA PREPARATION COMPLETED!")
        print("=" * 60)
        return True

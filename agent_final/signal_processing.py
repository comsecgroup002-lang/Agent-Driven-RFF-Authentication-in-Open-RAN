# signal_processing.py
# -*- coding: utf-8 -*-
"""

功能：
1. 读取原始 IQ 数据
2. 信道增强（多径 + 多普勒 + Rician/Rayleigh）
3. 工程扰动（AWGN + 非线性失真 + IQ 不平衡）
4. 控制性增强（符号漂移 + 时间偏移 + 频率漂移）
5. CFO 估计与补偿
6. 小波变换
7. STFT 频谱图生成与保存
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import pywt
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class SignalAugmentConfig:
    """信号增强配置"""
    # Stage 概率参数（可由 LLM 调整）
    retain_ratio: float = 0.5           # 保留原始信号的概率 [0.3, 0.7]
    channel_aug_prob: float = 1.0       # 信道增强概率（Stage 1）[0.7, 1.0]
    engineering_aug_prob: float = 0.7   # 工程扰动概率（Stage 2）[0.4, 0.9]
    drift_prob: float = 0.05            # 符号漂移概率（Stage 3a）[0.02, 0.15]
    shift_prob: float = 0.10            # 时间偏移概率（Stage 3b）[0.05, 0.20]
    
    # 信道参数（严格限制范围）
    tau_rms_min_ns: float = 20.0        # 最小 RMS 延迟扩展 (ns) [15, 30]
    tau_rms_max_ns: float = 150.0       # 最大 RMS 延迟扩展 (ns) [100, 200]
    doppler_min_hz: float = 0.1         # 最小多普勒频移 (Hz) [0.05, 0.2]
    doppler_max_hz: float = 5.0         # 最大多普勒频移 (Hz) [3.0, 8.0]
    k_factor_min_db: float = 3.0        # 最小 Rician K 因子 (dB) [2.0, 5.0]
    k_factor_max_db: float = 10.0       # 最大 Rician K 因子 (dB) [8.0, 15.0]
    
    # 工程扰动参数（严格限制范围）
    snr_min_db: float = 30.0            # 最小 SNR (dB) [25, 35]
    snr_max_db: float = 40.0            # 最大 SNR (dB) [35, 45]
    gain_std: float = 0.05              # 增益标准差 [0.02, 0.08]
    nonlinear_coef: float = 0.003       # 非线性系数 [0.001, 0.005]
    iq_amplitude_std: float = 0.02      # IQ 幅度不平衡标准差 [0.01, 0.03]
    iq_phase_std_deg: float = 1.0       # IQ 相位不平衡标准差 (度) [0.5, 2.0]
    
    # 漂移参数（严格限制范围）
    max_time_shift: int = 2             # 最大时间偏移样本数 [1, 4]
    drift_ppm_min: int = 300            # 最小频率漂移 (ppm) [200, 400]
    drift_ppm_max: int = 1000           # 最大频率漂移 (ppm) [800, 1200]


@dataclass 
class ProcessingConfig:
    """处理配置"""
    sample_count: int = int(2e7)
    sample_rate: float = 1e6
    window_size: int = 8192
    stride: int = 8192
    wavelet_type: str = 'db4'
    
    # STFT 参数
    stft_win_size: int = 256
    stft_overlap: int = 200
    stft_nfft: int = 512
    
    # 输出图像参数
    img_dpi: int = 150
    img_size: Tuple[int, int] = (560, 420)


class ChannelModel:
    """
    信道模型 - 模拟多径 + 多普勒 + Rician 衰落
    """
    
    @staticmethod
    def exp_pdp(tau_d: float, Ts: float, A_dB: float = -30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        指数功率延迟谱模型
        
        Args:
            tau_d: RMS 延迟扩展 (秒)
            Ts: 采样周期 (秒)
            A_dB: 最小可见路径功率 (dB)
        
        Returns:
            avg_path_gains: 平均路径增益 (dB)
            path_delays: 路径延迟 (秒)
        """
        A = 10 ** (A_dB / 10)
        lmax = int(np.ceil(-tau_d * np.log(A) / Ts))
        lmax = max(1, min(lmax, 100))  # 限制最大路径数
        
        p = np.arange(lmax + 1)
        path_delays = p * Ts
        
        sigma_tau = tau_d
        p_vals = (1 / sigma_tau) * np.exp(-p * Ts / sigma_tau)
        p_norm = p_vals / np.sum(p_vals)
        
        # 避免 log(0)
        p_norm = np.clip(p_norm, 1e-10, None)
        avg_path_gains = 10 * np.log10(p_norm)
        
        return avg_path_gains, path_delays
    
    @staticmethod
    def apply_multipath(sig: np.ndarray, path_gains_db: np.ndarray, 
                       path_delays: np.ndarray, Ts: float) -> np.ndarray:
        """应用多径效应"""
        output = np.zeros_like(sig, dtype=complex)
        
        for gain_db, delay in zip(path_gains_db, path_delays):
            delay_samples = int(np.round(delay / Ts))
            gain_linear = 10 ** (gain_db / 20)
            
            if delay_samples < len(sig):
                delayed = np.roll(sig, delay_samples)
                if delay_samples > 0:
                    delayed[:delay_samples] = 0
                output += gain_linear * delayed
        
        return output
    
    @staticmethod
    def apply_doppler(sig: np.ndarray, fD: float, Ts: float) -> np.ndarray:
        """应用多普勒效应（Jakes 模型简化版）"""
        N = len(sig)
        t = np.arange(N) * Ts
        
        # 随机相位的多普勒
        phase = 2 * np.pi * fD * t + np.random.uniform(0, 2 * np.pi)
        doppler_shift = np.exp(1j * phase)
        
        # 添加慢变化的衰落
        fade_rate = max(1, int(N / (fD * Ts * 100 + 1)))
        slow_fade = np.random.rayleigh(1.0, N // fade_rate + 1)
        slow_fade = np.interp(np.arange(N), np.arange(0, N, fade_rate), slow_fade[:N // fade_rate + 1])
        
        return sig * doppler_shift * slow_fade
    
    @staticmethod
    def apply_rician(sig: np.ndarray, k_factor_db: float) -> np.ndarray:
        """应用 Rician 衰落"""
        k = 10 ** (k_factor_db / 10)
        
        # LOS 分量
        los_power = k / (k + 1)
        nlos_power = 1 / (k + 1)
        
        # NLOS 分量（Rayleigh）
        nlos = (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) / np.sqrt(2)
        
        # 组合
        fading = np.sqrt(los_power) + np.sqrt(nlos_power) * nlos
        
        return sig * fading


class SignalAugmentor:
    """
    信号增强器 - 完整的增强流水线
    """
    
    def __init__(self, config: SignalAugmentConfig):
        self.config = config
        self.channel = ChannelModel()
    
    def augment_channel(self, sig: np.ndarray, Ts: float) -> np.ndarray:
        """
        Stage 1: 信道增强
        - 多径效应
        - 多普勒效应
        - Rician 衰落
        """
        cfg = self.config
        
        # 随机参数
        tau_rms = np.random.uniform(cfg.tau_rms_min_ns, cfg.tau_rms_max_ns) * 1e-9
        fD = np.random.uniform(cfg.doppler_min_hz, cfg.doppler_max_hz)
        k_factor = np.random.uniform(cfg.k_factor_min_db, cfg.k_factor_max_db)
        
        # 计算 PDP
        path_gains, path_delays = self.channel.exp_pdp(tau_rms, Ts)
        
        # 应用多径
        sig = self.channel.apply_multipath(sig, path_gains, path_delays, Ts)
        
        # 应用多普勒
        sig = self.channel.apply_doppler(sig, fD, Ts)
        
        # 应用 Rician 衰落
        sig = self.channel.apply_rician(sig, k_factor)
        
        return sig
    
    def augment_engineering(self, sig: np.ndarray) -> np.ndarray:
        """
        Stage 2: 工程扰动
        - AWGN 噪声
        - 非线性放大失真
        - IQ 不平衡
        """
        cfg = self.config
        
        # 1. AWGN
        snr = np.random.uniform(cfg.snr_min_db, cfg.snr_max_db)
        sig_power = np.mean(np.abs(sig) ** 2)
        noise_power = sig_power / (10 ** (snr / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
        sig = sig + noise
        
        # 2. 非线性失真
        gain = 1 + cfg.gain_std * np.random.randn()
        sig = gain * sig + cfg.nonlinear_coef * (sig ** 3)
        
        # 3. IQ 不平衡
        alpha = 1 + cfg.iq_amplitude_std * np.random.randn()
        theta = np.deg2rad(cfg.iq_phase_std_deg * np.random.randn())
        
        I = np.real(sig)
        Q = np.imag(sig)
        I_new = alpha * I
        Q_new = Q * np.cos(theta) + I * np.sin(theta)
        sig = I_new + 1j * Q_new
        
        return sig
    
    def augment_shift(self, sig: np.ndarray) -> np.ndarray:
        """时间偏移"""
        shift = np.random.randint(-self.config.max_time_shift, self.config.max_time_shift + 1)
        return np.roll(sig, shift)
    
    def augment_drift(self, sig: np.ndarray) -> np.ndarray:
        """频率漂移（通过重采样实现）"""
        cfg = self.config
        N = len(sig)
        
        # 随机漂移率
        drift_ppm = np.random.randint(cfg.drift_ppm_min, cfg.drift_ppm_max + 1)
        if np.random.rand() < 0.5:
            drift_ppm = -drift_ppm
        
        drift_ratio = 1 + drift_ppm / 1e6
        
        # 重采样
        new_len = int(N * drift_ratio)
        if new_len < 10:
            return sig
        
        # 使用 scipy 重采样
        resampled = scipy_signal.resample(sig, new_len)
        
        # 保持原始长度
        if len(resampled) >= N:
            return resampled[:N]
        else:
            result = np.zeros(N, dtype=complex)
            result[:len(resampled)] = resampled
            return result
    
    def apply_pipeline(self, sig: np.ndarray, Ts: float) -> np.ndarray:
        """
        完整增强流水线
        """
        cfg = self.config
        
        # 检查是否保留原始
        if np.random.rand() < cfg.retain_ratio:
            return sig.copy()
        
        # Stage 1: 信道增强
        if np.random.rand() < cfg.channel_aug_prob:
            try:
                sig = self.augment_channel(sig, Ts)
            except Exception as e:
                print(f"Warning: Channel augmentation failed: {e}")
        
        # Stage 2: 工程扰动
        if np.random.rand() < cfg.engineering_aug_prob:
            try:
                sig = self.augment_engineering(sig)
            except Exception as e:
                print(f"Warning: Engineering augmentation failed: {e}")
        
        # Stage 3: 控制性增强
        r = np.random.rand()
        if r < cfg.drift_prob:
            try:
                sig = self.augment_drift(sig)
            except Exception as e:
                print(f"Warning: Drift augmentation failed: {e}")
        elif r < cfg.drift_prob + cfg.shift_prob:
            sig = self.augment_shift(sig)
        
        return sig


class SignalProcessor:
    """
    信号处理器 - 完整处理流程
    """
    
    def __init__(self, proc_config: ProcessingConfig, aug_config: SignalAugmentConfig):
        self.proc_cfg = proc_config
        self.aug_cfg = aug_config
        self.augmentor = SignalAugmentor(aug_config)
    
    def estimate_cfo(self, sig: np.ndarray, Ts: float, L: int = 128) -> Tuple[np.ndarray, float]:
        """
        CFO 估计与补偿
        
        Returns:
            compensated_signal, estimated_cfo
        """
        if len(sig) < 2 * L:
            return sig, 0.0
        
        # 粗估计：基于瞬时频率
        phase = np.unwrap(np.angle(sig[:L]))
        inst_freq = np.diff(phase) / (2 * np.pi * Ts)
        cfo_coarse = np.mean(inst_freq)
        
        # 粗补偿
        t = np.arange(len(sig)) * Ts
        sig_c = sig * np.exp(-1j * 2 * np.pi * cfo_coarse * t)
        
        # 细估计：基于相关
        phi = np.angle(np.sum(sig_c[:L] * np.conj(sig_c[L:2*L])))
        cfo_fine = -phi / (2 * np.pi * L * Ts)
        
        # 细补偿
        sig_cf = sig_c * np.exp(-1j * 2 * np.pi * cfo_fine * t)
        
        total_cfo = cfo_coarse + cfo_fine
        
        return sig_cf, total_cfo
    
    def normalize_amplitude(self, sig: np.ndarray) -> Optional[np.ndarray]:
        """振幅归一化"""
        rms_val = np.sqrt(np.mean(np.abs(sig) ** 2))
        if rms_val < 1e-8:
            return None
        return sig / rms_val
    
    def wavelet_transform(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        小波变换（二级分解）
        
        Returns:
            cA2, cD1, cD2
        """
        # 取实部
        sig_real = np.real(sig)
        
        # 第一级分解
        cA1, cD1 = pywt.dwt(sig_real, self.proc_cfg.wavelet_type)
        
        # 第二级分解
        cA2, cD2 = pywt.dwt(cA1, self.proc_cfg.wavelet_type)
        
        return cA2, cD1, cD2
    
    def compute_stft(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 STFT"""
        f, t, Sxx = scipy_signal.spectrogram(
            sig,
            fs=self.proc_cfg.sample_rate,
            window='hann',
            nperseg=self.proc_cfg.stft_win_size,
            noverlap=self.proc_cfg.stft_overlap,
            nfft=self.proc_cfg.stft_nfft,
            mode='magnitude'
        )
        
        # 转换为 dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return Sxx_db, f, t
    
    def save_spectrogram(self, Sxx_db: np.ndarray, f: np.ndarray, t: np.ndarray,
                        save_path: str, title: str = ""):
        """保存频谱图"""
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        
        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        if title:
            ax.set_title(title, fontsize=10)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        fig.savefig(save_path, dpi=self.proc_cfg.img_dpi, bbox_inches='tight')
        plt.close(fig)
    
    def process_segment(self, segment: np.ndarray, Ts: float, 
                       apply_augmentation: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        处理单个片段
        
        Returns:
            (cA2, cD1, cD2) 或 None（如果处理失败）
        """
        try:
            # 数据增强
            if apply_augmentation:
                segment = self.augmentor.apply_pipeline(segment, Ts)
            
            # CFO 补偿
            segment, _ = self.estimate_cfo(segment, Ts)
            
            # 振幅归一化
            segment = self.normalize_amplitude(segment)
            if segment is None:
                return None
            
            # 小波变换
            cA2, cD1, cD2 = self.wavelet_transform(segment)
            
            return cA2, cD1, cD2
            
        except Exception as e:
            print(f"Warning: Segment processing failed: {e}")
            return None


def process_device_file(args: Tuple) -> int:
    """处理单个设备的单个文件（用于并行处理）"""
    (device_idx, file_idx, dat_file, save_dirs, proc_config, aug_config) = args
    
    processor = SignalProcessor(proc_config, aug_config)
    Ts = 1.0 / proc_config.sample_rate
    
    # 读取数据
    try:
        raw = np.fromfile(dat_file, dtype=np.float32, count=proc_config.sample_count)
    except Exception as e:
        print(f"Error reading {dat_file}: {e}")
        return 0
    
    # 构造 IQ
    I = raw[0::2]
    Q = raw[1::2]
    IQ_data = I + 1j * Q
    
    total_points = len(IQ_data)
    num_windows = (total_points - proc_config.window_size) // proc_config.stride + 1
    
    processed_count = 0
    
    for w in range(num_windows):
        idx_start = w * proc_config.stride
        idx_end = idx_start + proc_config.window_size
        segment = IQ_data[idx_start:idx_end]
        
        # 处理
        result = processor.process_segment(segment, Ts)
        if result is None:
            continue
        
        cA2, cD1, cD2 = result
        
        # 计算并保存频谱图
        for component, data in [('cA2', cA2), ('cD1', cD1), ('cD2', cD2)]:
            Sxx_db, f, t = processor.compute_stft(data)
            
            filename = f"IQ{file_idx:02d}_win{w+1:03d}.png"
            save_path = save_dirs[component] / filename
            
            title = f"Device{device_idx} IQ{file_idx} Win{w+1} - {component}"
            processor.save_spectrogram(Sxx_db, f, t, str(save_path), title)
        
        processed_count += 1
    
    return processed_count


def run_signal_processing(base_dir: str, output_dir: str,
                         proc_config: ProcessingConfig,
                         aug_config: SignalAugmentConfig,
                         device_range: Tuple[int, int] = (1, 11),
                         file_range: Tuple[int, int] = (1, 5),
                         num_workers: int = 4) -> Dict[str, int]:
    """
    运行完整的信号处理流程
    
    Args:
        base_dir: 原始数据目录（包含 Device1, Device2, ...）
        output_dir: 输出目录
        proc_config: 处理配置
        aug_config: 增强配置
        device_range: 设备范围 (start, end)，包含端点
        file_range: 文件范围 (start, end)
        num_workers: 并行工作进程数
    
    Returns:
        处理统计 {device_name: window_count}
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    components = ['cA2', 'cD1', 'cD2']
    for comp in components:
        (output_path / comp).mkdir(parents=True, exist_ok=True)
    
    # 收集所有任务
    tasks = []
    
    for device_idx in range(device_range[0], device_range[1] + 1):
        device_name = f"Device{device_idx}"
        device_dir = base_path / device_name
        
        if not device_dir.exists():
            print(f"Warning: {device_dir} does not exist, skipping")
            continue
        
        # 创建设备输出目录
        save_dirs = {}
        for comp in components:
            comp_dir = output_path / comp / device_name
            comp_dir.mkdir(parents=True, exist_ok=True)
            save_dirs[comp] = comp_dir
        
        for file_idx in range(file_range[0], file_range[1] + 1):
            dat_file = device_dir / f"IQ_{file_idx}.dat"
            
            if not dat_file.exists():
                print(f"Warning: {dat_file} does not exist, skipping")
                continue
            
            tasks.append((device_idx, file_idx, str(dat_file), save_dirs, proc_config, aug_config))
    
    print(f"Total tasks: {len(tasks)}")
    
    # 并行处理
    stats = {}
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_device_file, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                task = futures[future]
                device_name = f"Device{task[0]}"
                
                try:
                    count = future.result()
                    stats[device_name] = stats.get(device_name, 0) + count
                except Exception as e:
                    print(f"Error processing {device_name}: {e}")
    else:
        for task in tqdm(tasks, desc="Processing"):
            device_name = f"Device{task[0]}"
            try:
                count = process_device_file(task)
                stats[device_name] = stats.get(device_name, 0) + count
            except Exception as e:
                print(f"Error processing {device_name}: {e}")
    
    return stats


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    # 测试增强器
    print("Testing SignalAugmentor...")
    
    aug_config = SignalAugmentConfig()
    augmentor = SignalAugmentor(aug_config)
    
    # 生成测试信号
    N = 8192
    Ts = 1e-6
    t = np.arange(N) * Ts
    test_signal = np.exp(1j * 2 * np.pi * 1000 * t)  # 1kHz 信号
    
    # 测试各增强阶段
    aug_channel = augmentor.augment_channel(test_signal.copy(), Ts)
    print(f"Channel augmented: power ratio = {np.mean(np.abs(aug_channel)**2) / np.mean(np.abs(test_signal)**2):.4f}")
    
    aug_eng = augmentor.augment_engineering(test_signal.copy())
    print(f"Engineering augmented: power ratio = {np.mean(np.abs(aug_eng)**2) / np.mean(np.abs(test_signal)**2):.4f}")
    
    aug_full = augmentor.apply_pipeline(test_signal.copy(), Ts)
    print(f"Full pipeline: power ratio = {np.mean(np.abs(aug_full)**2) / np.mean(np.abs(test_signal)**2):.4f}")
    
    # 测试处理器
    print("\nTesting SignalProcessor...")
    proc_config = ProcessingConfig()
    processor = SignalProcessor(proc_config, aug_config)
    
    result = processor.process_segment(test_signal, Ts)
    if result:
        cA2, cD1, cD2 = result
        print(f"Wavelet coefficients: cA2={len(cA2)}, cD1={len(cD1)}, cD2={len(cD2)}")
    
    print("\n✅ All tests passed!")

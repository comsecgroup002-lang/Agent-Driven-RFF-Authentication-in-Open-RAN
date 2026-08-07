# llm_advisor.py
# -*- coding: utf-8 -*-
"""
LLM 顾问模块 - V2 增强版

核心优化：
1. 接收云端经验摘要，避免极端参数
2. 参数调整有边界保护
3. 考虑历史失败模式
4. 规则回退策略更加保守
"""

import os
import re
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TrainingAdvice:
    """训练建议"""
    analysis: str
    param_changes: Dict
    reasoning: str
    confidence: float
    requires_reprocessing: bool = False


# 参数范围定义（更保守的范围）
PARAM_RANGES = {
    # Stage 概率参数（收紧范围避免极端）
    "retain_ratio": (0.35, 0.65),  # 原 0.3-0.7，收紧
    "channel_aug_prob": (0.7, 1.0),
    "engineering_aug_prob": (0.5, 0.85),  # 原 0.4-0.9，收紧
    "drift_prob": (0.02, 0.15),
    "shift_prob": (0.05, 0.20),
    
    # 信道参数
    "tau_rms_min_ns": (15, 30),
    "tau_rms_max_ns": (100, 200),
    "doppler_min_hz": (0.05, 0.2),
    "doppler_max_hz": (3.0, 8.0),
    "k_factor_min_db": (2.0, 5.0),
    "k_factor_max_db": (8.0, 15.0),
    
    # 工程扰动参数
    "snr_min_db": (25, 35),
    "snr_max_db": (35, 45),
    "gain_std": (0.02, 0.08),
    "nonlinear_coef": (0.001, 0.005),
    "iq_amplitude_std": (0.01, 0.03),
    "iq_phase_std_deg": (0.5, 2.0),
}

# 单步最大调整量（避免剧烈变化）
MAX_STEP_SIZE = {
    "retain_ratio": 0.08,  # 每次最多调整 0.08
    "engineering_aug_prob": 0.1,
    "channel_aug_prob": 0.1,
}


class LLMAdvisor:
    """LLM 顾问 - 集成云端经验"""
    
    SYSTEM_PROMPT = """You are an expert advisor for RF fingerprint recognition optimization.

## Your Role
Analyze optimization results and recommend CONSERVATIVE signal augmentation adjustments.
You have access to cloud experience summary showing what worked for other nodes.

## CRITICAL RULES
1. **NEVER recommend extreme parameters** - stay within safe ranges
2. **Small incremental changes** - max 0.08 change per step for retain_ratio
3. **Balance all three metrics** - closed_acc, open_auc, AND reject_rate
4. **Learn from cloud experience** - follow successful parameter ranges

## Parameter Guidelines
- retain_ratio [0.35-0.65]: Controls augmentation intensity
  - Higher (0.55-0.65): Better closed_acc, worse open_auc
  - Lower (0.35-0.45): Better open_auc, worse closed_acc
  - BALANCED (0.45-0.55): Best for most cases
  
- engineering_aug_prob [0.5-0.85]: Engineering noise probability
  - Keep moderate, extreme values hurt both metrics

## Common Failure Patterns
- retain_ratio < 0.35: Causes closed_acc collapse
- retain_ratio > 0.65: Causes open_auc collapse  
- reject_rate > 0.6 usually means model is too conservative

## Output Format (JSON only)
{
  "analysis": "Brief analysis",
  "param_changes": {
    "retain_ratio": 0.48,
    "engineering_aug_prob": 0.72
  },
  "reasoning": "Why these conservative changes",
  "confidence": 0.7,
  "requires_reprocessing": true
}

IMPORTANT: Keep changes SMALL and BALANCED. Never push parameters to extremes."""

    def __init__(self, llm_cfg: dict):
        self.model_path = llm_cfg.get("model_path", llm_cfg.get("model", "Qwen/Qwen2.5-3B-Instruct"))
        self.device_map = llm_cfg.get("device_map", "auto")
        self.dtype = llm_cfg.get("dtype", "fp16")
        self.quantization = llm_cfg.get("quantization", "auto")
        self.max_new_tokens = llm_cfg.get("max_new_tokens", 400)
        self.offline_mode = llm_cfg.get("offline_mode", True)
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _lazy_load(self):
        """延迟加载模型"""
        if self._loaded:
            return
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        print(f"[LLM Advisor] Loading model from: {self.model_path}")
        
        is_local = os.path.isdir(self.model_path)
        print(f"[LLM Advisor] Local path: {is_local}, offline: {self.offline_mode}")
        
        accelerate_available = False
        try:
            import accelerate
            accelerate_available = True
        except ImportError:
            print("[LLM Advisor] accelerate not installed")
        
        tokenizer_kwargs = {"trust_remote_code": True}
        if is_local or self.offline_mode:
            tokenizer_kwargs["local_files_only"] = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        
        model_kwargs = {"trust_remote_code": True}
        if is_local or self.offline_mode:
            model_kwargs["local_files_only"] = True
        
        bnb_available = False
        try:
            import bitsandbytes
            bnb_available = True
        except ImportError:
            pass
        
        quant_mode = self.quantization
        if quant_mode == "auto":
            quant_mode = "4bit" if (bnb_available and accelerate_available) else "none"
        
        if quant_mode == "4bit" and bnb_available and accelerate_available:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = self.device_map
        elif accelerate_available and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = self.device_map
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        
        if not accelerate_available and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        self._loaded = True
        
        print("[LLM Advisor] ✅ Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"[LLM Advisor] GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _build_prompt(self, current_params: Dict, best_metrics: Dict,
                     objectives: Dict, pareto_summary: str,
                     history: List[Dict], diagnosis: Dict,
                     model_info: Dict = None,
                     cloud_experience: Dict = None) -> str:
        """构建提示 - 包含 RAG 检索结果"""
        
        sig_aug = current_params.get("signal_augment", {})
        
        prompt = f"""### Current Performance
- closed_acc: {best_metrics.get('closed_acc', 0):.4f} (target: {objectives.get('min_closed_acc', 0.9)})
- open_auc: {best_metrics.get('open_auc', 0):.4f} (target: {objectives.get('target_open_auc', 0.85)})
- reject_rate: {best_metrics.get('reject_rate', 0):.4f} (max: {objectives.get('max_reject_rate', 0.6)})

### Current Parameters
- retain_ratio: {sig_aug.get('retain_ratio', 0.5)}
- engineering_aug_prob: {sig_aug.get('engineering_aug_prob', 0.7)}
- channel_aug_prob: {sig_aug.get('channel_aug_prob', 1.0)}

### Diagnosis
- Improvement trend: {diagnosis.get('improvement_trend', 'unknown')}
- Training rounds: {diagnosis.get('training_rounds', 0)}
- Gaps: closed_acc={diagnosis.get('gaps', {}).get('closed_acc_gap', 0):.4f}, open_auc={diagnosis.get('gaps', {}).get('open_auc_gap', 0):.4f}
"""
        
        # ★ 添加 RAG 检索结果
        if cloud_experience:
            # RAG context（格式化的检索结果）
            rag_context = cloud_experience.get("rag_context")
            if rag_context:
                prompt += f"""
{rag_context}
"""
            
            # 检索到的具体案例
            retrieved_cases = cloud_experience.get("retrieved_cases", {})
            success_cases = retrieved_cases.get("success", [])
            failure_cases = retrieved_cases.get("failure", [])
            
            if success_cases and not rag_context:
                # 如果没有格式化的 context，手动构建
                prompt += "\n### Retrieved SUCCESSFUL Cases (follow these patterns):\n"
                for i, case in enumerate(success_cases[:3], 1):
                    prompt += f"""
Case {i} (similarity: {case.get('similarity', 0):.2f}):
  - Before: closed={case.get('state_before', {}).get('closed_acc', 0):.4f}, open={case.get('state_before', {}).get('open_auc', 0):.4f}
  - Action: retain_ratio={case.get('action', {}).get('retain_ratio', 'N/A')}
  - After:  closed={case.get('state_after', {}).get('closed_acc', 0):.4f}, open={case.get('state_after', {}).get('open_auc', 0):.4f}
  - Success: {case.get('success', False)}
"""
            
            if failure_cases and not rag_context:
                prompt += "\n### Retrieved FAILED Cases (AVOID these patterns):\n"
                for i, case in enumerate(failure_cases[:2], 1):
                    prompt += f"""
Failed Case {i}:
  - Action that FAILED: retain_ratio={case.get('action', {}).get('retain_ratio', 'N/A')}
  - Result: closed={case.get('state_after', {}).get('closed_acc', 0):.4f}, open={case.get('state_after', {}).get('open_auc', 0):.4f}
  ⚠️ DO NOT repeat this pattern
"""
            
            # Policy Guard 建议
            policy_suggestion = cloud_experience.get("policy_suggestion", {})
            if policy_suggestion:
                prompt += f"""
### Cloud Policy Guard Suggestion
Based on similar successful cases, the cloud suggests:
{policy_suggestion}
Consider following this suggestion or making similar adjustments.
"""
            
            # 安全范围
            safe_ranges = cloud_experience.get("safe_param_ranges", {})
            max_steps = cloud_experience.get("max_step_sizes", {})
            if safe_ranges:
                prompt += f"""
### Safety Constraints (MUST follow)
- Safe parameter ranges: {safe_ranges}
- Maximum step sizes: {max_steps}
"""
        
        prompt += """
### Task
Based on the retrieved cases and current state, recommend CONSERVATIVE parameter changes.
Learn from successful cases, avoid patterns that led to failure.
Stay within safety constraints.
Output JSON only:"""
        
        return prompt
    
    @torch.no_grad()
    def get_training_advice(self, current_params: Dict, best_metrics: Dict,
                           objectives: Dict, pareto_summary: str,
                           history: List[Dict], diagnosis: Dict,
                           model_info: Dict = None,
                           cloud_experience: Dict = None) -> Optional[TrainingAdvice]:
        """获取训练建议 - 集成云端经验"""
        try:
            self._lazy_load()
        except Exception as e:
            print(f"[LLM Advisor] Failed to load: {e}")
            return self._rule_based_fallback(current_params, best_metrics, objectives, diagnosis, cloud_experience)
        
        user_prompt = self._build_prompt(
            current_params, best_metrics, objectives,
            pareto_summary, history, diagnosis, model_info,
            cloud_experience
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
        except Exception as e:
            print(f"[LLM Advisor] Chat template failed: {e}")
            text = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"[LLM Advisor] Raw response:\n{raw_text[:600]}")
        
        advice = self._parse_response(raw_text, current_params, diagnosis, cloud_experience)
        
        if advice is None:
            print("[LLM Advisor] Parse failed, using rule-based fallback")
            return self._rule_based_fallback(current_params, best_metrics, objectives, diagnosis, cloud_experience)
        
        return advice
    
    def _parse_response(self, text: str, current_params: Dict, 
                       diagnosis: Dict, cloud_experience: Dict = None) -> Optional[TrainingAdvice]:
        """解析响应"""
        text = text.strip()
        
        # 尝试直接解析
        try:
            data = json.loads(text)
            return self._validate_advice(data, current_params, cloud_experience)
        except json.JSONDecodeError:
            pass
        
        # 查找代码块
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block:
            try:
                data = json.loads(code_block.group(1))
                return self._validate_advice(data, current_params, cloud_experience)
            except json.JSONDecodeError:
                pass
        
        # 查找 JSON 对象
        start_idx = text.find('{')
        if start_idx != -1:
            depth = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            json_str = text[start_idx:i + 1].replace("'", '"')
                            data = json.loads(json_str)
                            return self._validate_advice(data, current_params, cloud_experience)
                        except json.JSONDecodeError:
                            pass
                        break
        
        return None
    
    def _validate_advice(self, data: Dict, current_params: Dict, 
                        cloud_experience: Dict = None) -> TrainingAdvice:
        """验证并创建建议 - 带边界保护"""
        param_changes = data.get("param_changes", {})
        sig_aug = current_params.get("signal_augment", {})
        validated = {}
        
        for key, new_value in param_changes.items():
            if key not in PARAM_RANGES:
                continue
            
            try:
                new_value = float(new_value)
            except (ValueError, TypeError):
                continue
            
            # 获取当前值
            current_value = sig_aug.get(key, 0.5)
            
            # ★ 限制单步变化量
            max_step = MAX_STEP_SIZE.get(key, 0.1)
            delta = new_value - current_value
            if abs(delta) > max_step:
                new_value = current_value + (max_step if delta > 0 else -max_step)
                print(f"[LLM Advisor] Clamped {key} step: {delta:.3f} -> {new_value - current_value:.3f}")
            
            # ★ 限制在安全范围内
            min_val, max_val = PARAM_RANGES[key]
            new_value = max(min_val, min(max_val, new_value))
            
            # ★ 如果有云端推荐范围，进一步约束
            if cloud_experience and key == "retain_ratio":
                rec_range = cloud_experience.get("recommended_retain_ratio", (0.4, 0.6))
                # 不强制在范围内，但如果超出太多则警告
                if new_value < rec_range[0] - 0.1 or new_value > rec_range[1] + 0.1:
                    print(f"[LLM Advisor] ⚠️ {key}={new_value:.2f} outside recommended [{rec_range[0]:.2f}, {rec_range[1]:.2f}]")
            
            validated[key] = new_value
        
        # 检查是否需要重新处理
        signal_params = {"retain_ratio", "channel_aug_prob", "engineering_aug_prob",
                        "drift_prob", "shift_prob", "tau_rms_min_ns", "tau_rms_max_ns",
                        "doppler_min_hz", "doppler_max_hz", "snr_min_db", "snr_max_db"}
        requires_reprocessing = any(k in signal_params for k in validated.keys())
        
        return TrainingAdvice(
            analysis=str(data.get("analysis", "")),
            param_changes=validated,
            reasoning=str(data.get("reasoning", "")),
            confidence=float(data.get("confidence", 0.7)),
            requires_reprocessing=data.get("requires_reprocessing", requires_reprocessing),
        )
    
    def _rule_based_fallback(self, current_params: Dict, best_metrics: Dict,
                            objectives: Dict, diagnosis: Dict,
                            cloud_experience: Dict = None) -> TrainingAdvice:
        """
        规则回退 - 更保守的策略，考虑云端经验
        """
        gaps = diagnosis.get("gaps", {})
        
        sig_aug = current_params.get("signal_augment", {})
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
        else:
            rec_retain = (0.4, 0.6)
            rec_eng = (0.6, 0.8)
        
        # 计算目标值（在推荐范围内）
        target_retain = (rec_retain[0] + rec_retain[1]) / 2
        target_eng = (rec_eng[0] + rec_eng[1]) / 2
        
        # 策略选择（更保守）
        if closed_gap > 0.05:
            # closed_acc 差距大 - 适度增加 retain_ratio
            new_retain = min(rec_retain[1], current_retain + 0.05)
            changes["retain_ratio"] = new_retain
            analysis = f"Large closed_acc gap ({closed_gap:.4f})"
            reasoning = f"Increase retain_ratio conservatively: {current_retain:.2f} -> {new_retain:.2f}"
        
        elif open_gap > 0.05:
            # open_auc 差距大 - 适度降低 retain_ratio
            new_retain = max(rec_retain[0], current_retain - 0.05)
            changes["retain_ratio"] = new_retain
            analysis = f"Large open_auc gap ({open_gap:.4f})"
            reasoning = f"Decrease retain_ratio conservatively: {current_retain:.2f} -> {new_retain:.2f}"
        
        elif reject_excess > 0.1:
            # reject_rate 过高 - 往中间靠
            if current_retain < target_retain:
                changes["retain_ratio"] = min(target_retain, current_retain + 0.03)
            else:
                changes["retain_ratio"] = max(target_retain, current_retain - 0.03)
            analysis = f"High reject_rate ({reject_excess:.4f})"
            reasoning = "Move retain_ratio toward balanced value"
        
        elif open_gap > 0 or closed_gap > 0:
            # 小差距 - 微调
            if open_gap > closed_gap:
                changes["retain_ratio"] = max(rec_retain[0], current_retain - 0.03)
                analysis = f"Small open_auc gap ({open_gap:.4f})"
            else:
                changes["retain_ratio"] = min(rec_retain[1], current_retain + 0.03)
                analysis = f"Small closed_acc gap ({closed_gap:.4f})"
            reasoning = "Fine-tuning within recommended range"
        
        else:
            # 已经很好了，保持
            analysis = "Performance acceptable"
            reasoning = "No significant changes needed"
        
        # 确保在安全范围内
        if "retain_ratio" in changes:
            changes["retain_ratio"] = max(0.35, min(0.65, changes["retain_ratio"]))
        
        return TrainingAdvice(
            analysis=analysis,
            param_changes=changes,
            reasoning=reasoning,
            confidence=0.6,
            requires_reprocessing=bool(changes),
        )
    
    def unload(self):
        """释放资源"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[LLM Advisor] Unloaded.")

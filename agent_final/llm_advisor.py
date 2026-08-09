# llm_advisor.py
# -*- coding: utf-8 -*-
"""Cloud-side RAG-enabled LLM advisory module.

The LLM is deliberately outside the per-signal authentication path.  It only
produces bounded candidate guidance in the threshold/operating-point space.
All executable decisions remain subject to deterministic validation by the
cloud PolicyGuard and the edge control agent.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ThresholdAdvice:
    """A non-executable candidate guidance proposal produced by the LLM."""

    analysis: str
    threshold_patch: Dict[str, float]
    reasoning: str
    confidence: float
    requires_reprocessing: bool = False
    validation_adjusted: bool = False
    validation_events: list = field(default_factory=list)


# Backward-compatible alias for older imports.  The semantics are now threshold
# guidance rather than signal-augmentation/training advice.
TrainingAdvice = ThresholdAdvice


# Global admissible ranges for the cloud-side LLM proposal.  These ranges are
# intentionally compatible with the edge threshold-search grid.
THRESHOLD_RANGES = {
    "accept_quantile": (0.80, 0.95),
    "margin_quantile": (0.10, 0.30),
    "rho": (-0.15, 0.35),
}

# Maximum change relative to the current edge operating point for a single
# cloud advisory event.  The LLM never bypasses these limits.
MAX_STEP_SIZE = {
    "accept_quantile": 0.05,
    "margin_quantile": 0.05,
    "rho": 0.10,
}


class LLMAdvisor:
    """RAG-enabled LLM advisory agent executed on the cloud side."""

    SYSTEM_PROMPT = """You are a cloud-side advisory agent for RF fingerprint authentication.

Your role is limited to proposing conservative operating-point guidance after
edge-side cached-score correction remains infeasible. You do NOT authenticate
signals and you do NOT directly execute control actions.

HARD FEASIBILITY OBJECTIVES
- closed-set accuracy A_c >= gamma_c
- open-set AUROC A_o >= gamma_o

The rejection rate R is an auxiliary service-availability metric. It may be
considered when choosing among otherwise suitable operating points, but it is
NOT an additional hard feasibility condition.

ALLOWED ADVISORY VARIABLES
- accept_quantile
- margin_quantile
- rho

CRITICAL SAFETY RULES
1. Return only the allowed threshold-space variables.
2. Keep every value inside the supplied safe ranges.
3. Keep every single-event change inside the supplied maximum step limit.
4. Retrieved successful cases are positive evidence; retrieved failed cases
   are contextual negative evidence, not hard rejection rules.
5. The output is only a candidate proposal. Deterministic cloud and edge-side
   validation will decide what can be used.

OUTPUT JSON ONLY
{
  "analysis": "brief diagnosis",
  "threshold_patch": {
    "accept_quantile": 0.90,
    "margin_quantile": 0.20,
    "rho": 0.05
  },
  "reasoning": "why this bounded proposal may reduce the A_c/A_o gaps",
  "confidence": 0.70
}
"""

    def __init__(self, llm_cfg: dict):
        llm_cfg = llm_cfg or {}
        self.model_path = llm_cfg.get(
            "model_path", llm_cfg.get("model", "Qwen/Qwen2.5-3B-Instruct")
        )
        self.device_map = llm_cfg.get("device_map", "auto")
        self.dtype = llm_cfg.get("dtype", "fp16")
        self.quantization = llm_cfg.get("quantization", "auto")
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 400))
        self.offline_mode = bool(llm_cfg.get("offline_mode", True))

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print(f"[Cloud LLM Advisor] Loading model from: {self.model_path}")

        is_local = os.path.isdir(self.model_path)
        tokenizer_kwargs = {"trust_remote_code": True}
        if is_local or self.offline_mode:
            tokenizer_kwargs["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, **tokenizer_kwargs
        )

        model_kwargs = {"trust_remote_code": True}
        if is_local or self.offline_mode:
            model_kwargs["local_files_only"] = True

        try:
            import accelerate  # noqa: F401
            accelerate_available = True
        except ImportError:
            accelerate_available = False

        try:
            import bitsandbytes  # noqa: F401
            bnb_available = True
        except ImportError:
            bnb_available = False

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
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            if accelerate_available:
                model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )
        if (
            torch.cuda.is_available()
            and not accelerate_available
            and not hasattr(self.model, "hf_device_map")
        ):
            self.model = self.model.cuda()

        self.model.eval()
        self._loaded = True
        print("[Cloud LLM Advisor] Model loaded successfully")

    @staticmethod
    def _safe_thresholds(current_thresholds: Optional[Dict]) -> Dict[str, float]:
        current_thresholds = current_thresholds or {}
        return {
            "accept_quantile": float(current_thresholds.get("accept_quantile", 0.90)),
            "margin_quantile": float(current_thresholds.get("margin_quantile", 0.20)),
            "rho": float(
                current_thresholds.get(
                    "rho", current_thresholds.get("delta_fused", 0.0)
                )
            ),
        }

    def _build_prompt(
        self,
        current_thresholds: Dict,
        best_metrics: Dict,
        objectives: Dict,
        diagnosis: Dict,
        rag_context: Optional[str] = None,
        policy_suggestion: Optional[Dict] = None,
    ) -> str:
        cur = self._safe_thresholds(current_thresholds)
        gaps = diagnosis.get("gaps", {}) if diagnosis else {}

        prompt = f"""### Current authentication state
- A_c: {best_metrics.get('closed_acc', 0.0):.4f} (target {objectives.get('min_closed_acc', 0.90):.4f})
- A_o: {best_metrics.get('open_auc', 0.0):.4f} (target {objectives.get('target_open_auc', 0.85):.4f})
- R: {best_metrics.get('reject_rate', 0.0):.4f} (auxiliary only)

### Feasibility gaps
- A_c gap: {gaps.get('closed_acc_gap', 0.0):.4f}
- A_o gap: {gaps.get('open_auc_gap', 0.0):.4f}

### Current operating point
- accept_quantile: {cur['accept_quantile']:.4f}
- margin_quantile: {cur['margin_quantile']:.4f}
- rho: {cur['rho']:.4f}

### Deterministic safety constraints
- Safe ranges: {THRESHOLD_RANGES}
- Maximum single-event steps: {MAX_STEP_SIZE}
"""
        if rag_context:
            prompt += f"\n{rag_context}\n"
        if policy_suggestion:
            prompt += (
                "\n### Non-generative retrieval reference\n"
                f"A deterministic aggregation of retrieved successful cases suggests: "
                f"{policy_suggestion}\n"
            )

        prompt += """
### Task
Propose a small threshold-space patch that may reduce the A_c/A_o feasibility
gaps. Do not propose signal-augmentation parameters, retraining, model changes,
or direct authentication decisions. Output JSON only.
"""
        return prompt

    @torch.no_grad()
    def get_threshold_guidance(
        self,
        current_thresholds: Dict,
        best_metrics: Dict,
        objectives: Dict,
        diagnosis: Dict,
        rag_context: Optional[str] = None,
        policy_suggestion: Optional[Dict] = None,
    ) -> Optional[ThresholdAdvice]:
        """Generate a cloud-side bounded threshold-space candidate proposal."""
        try:
            self._lazy_load()
        except Exception as exc:
            print(f"[Cloud LLM Advisor] Model unavailable: {exc}")
            return None

        user_prompt = self._build_prompt(
            current_thresholds=current_thresholds,
            best_metrics=best_metrics,
            objectives=objectives,
            diagnosis=diagnosis,
            rag_context=rag_context,
            policy_suggestion=policy_suggestion,
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
        except Exception:
            text = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[Cloud LLM Advisor] Raw response:\n{raw_text[:600]}")
        return self._parse_response(raw_text, current_thresholds)

    # Compatibility wrapper for older code.  Any legacy caller is constrained
    # to the new threshold-space semantics.
    def get_training_advice(self, **kwargs) -> Optional[ThresholdAdvice]:
        current_params = kwargs.get("current_params", {}) or {}
        current_thresholds = kwargs.get("current_thresholds") or (
            current_params.get("open_set", {}).get("calibration", {})
        )
        cloud_experience = kwargs.get("cloud_experience", {}) or {}
        return self.get_threshold_guidance(
            current_thresholds=current_thresholds,
            best_metrics=kwargs.get("best_metrics", {}),
            objectives=kwargs.get("objectives", {}),
            diagnosis=kwargs.get("diagnosis", {}),
            rag_context=cloud_experience.get("rag_context"),
            policy_suggestion=cloud_experience.get("policy_suggestion"),
        )

    def _parse_response(
        self, text: str, current_thresholds: Dict
    ) -> Optional[ThresholdAdvice]:
        text = (text or "").strip()
        candidates = [text]

        block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if block:
            candidates.append(block.group(1))

        start_idx = text.find("{")
        if start_idx >= 0:
            depth = 0
            for idx, char in enumerate(text[start_idx:], start_idx):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start_idx : idx + 1])
                        break

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            advice = self._validate_advice(data, current_thresholds)
            if advice is not None:
                return advice
        return None

    def _validate_advice(
        self, data: Dict, current_thresholds: Dict
    ) -> Optional[ThresholdAdvice]:
        cur = self._safe_thresholds(current_thresholds)
        raw_patch = data.get("threshold_patch", data.get("param_changes", {}))
        if not isinstance(raw_patch, dict):
            return None

        validated: Dict[str, float] = {}
        events = []
        for key, proposed in raw_patch.items():
            if key not in THRESHOLD_RANGES:
                events.append(f"unsupported:{key}")
                continue
            try:
                proposed = float(proposed)
            except (TypeError, ValueError):
                events.append(f"non_numeric:{key}")
                continue

            current = cur[key]
            max_step = MAX_STEP_SIZE[key]
            raw_delta = proposed - current
            delta = max(-max_step, min(max_step, raw_delta))
            if delta != raw_delta:
                events.append(f"step_clamp:{key}")
            value = current + delta
            lo, hi = THRESHOLD_RANGES[key]
            clipped = max(lo, min(hi, value))
            if clipped != value:
                events.append(f"range_clamp:{key}")
            validated[key] = float(clipped)

        if not validated:
            return None

        try:
            confidence = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        return ThresholdAdvice(
            analysis=str(data.get("analysis", "")),
            threshold_patch=validated,
            reasoning=str(data.get("reasoning", "")),
            confidence=confidence,
            requires_reprocessing=False,
            validation_adjusted=bool(events),
            validation_events=events,
        )

    def unload(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Cloud LLM Advisor] Unloaded")

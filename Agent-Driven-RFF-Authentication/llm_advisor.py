# -*- coding: utf-8 -*-
"""Cloud-side RAG-enabled LLM advisory component.

The advisor only proposes bounded updates in the edge threshold/search space.
It never authenticates a signal, modifies model weights, changes signal
augmentation, or directly executes an edge action.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


PARAM_RANGES = {
    "accept_quantile": (0.80, 0.95),
    "margin_quantile": (0.10, 0.30),
    "delta_fused": (-0.15, 0.35),
    "delta_margin": (0.05, 0.35),
}

MAX_STEP_SIZE = {
    "accept_quantile": 0.05,
    "margin_quantile": 0.05,
    "delta_fused": 0.05,
    "delta_margin": 0.05,
}

ALIASES = {"rho": "delta_fused"}


@dataclass
class GuidanceAdvice:
    proposal_id: str
    analysis: str
    param_changes: Dict[str, float]
    reasoning: str
    confidence: float
    requires_reprocessing: bool = False
    raw_param_changes: Dict[str, object] = field(default_factory=dict)
    first_pass_adjusted: bool = False
    validation_notes: list[str] = field(default_factory=list)


class LLMAdvisor:
    """Cloud-side bounded proposal generator for threshold-space guidance."""

    SYSTEM_PROMPT = """You are the cloud-side advisory component of an RFF authentication decision framework.

Your role is strictly advisory. You DO NOT authenticate RF signals, DO NOT modify model weights,
DO NOT change signal augmentation, and DO NOT directly execute any edge action.

The edge control agent uses cached-score threshold replay. Recommend only conservative updates
in the following allowlisted search variables:
- accept_quantile: [0.80, 0.95]
- margin_quantile: [0.10, 0.30]
- delta_fused: [-0.15, 0.35]
- delta_margin: [0.05, 0.35]

Maximum single-step change for every variable is 0.05. The deterministic policy guard will
clip or reject unsupported output before it can affect the edge search.

Authentication feasibility is determined by BOTH closed-set accuracy Ac and open-set AUROC Ao.
Rejection rate R is an auxiliary service-availability indicator, not a hard feasibility condition.

Return JSON only:
{
  "analysis": "brief diagnosis",
  "param_changes": {
    "accept_quantile": 0.90,
    "margin_quantile": 0.20,
    "delta_fused": 0.10,
    "delta_margin": 0.10
  },
  "reasoning": "why this bounded search guidance is appropriate",
  "confidence": 0.70,
  "requires_reprocessing": false
}

Do not emit non-allowlisted parameters. Never request retraining or model refresh. Retrieved
failed cases are negative contextual evidence only, not hard rejection rules."""

    def __init__(self, llm_cfg: Optional[dict] = None):
        cfg = llm_cfg or {}
        self.model_path = cfg.get("model_path", cfg.get("model", "Qwen/Qwen2.5-3B-Instruct"))
        self.device_map = cfg.get("device_map", "auto")
        self.dtype = cfg.get("dtype", "fp16")
        self.quantization = cfg.get("quantization", "auto")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 400))
        self.offline_mode = bool(cfg.get("offline_mode", True))
        self.enabled = bool(cfg.get("enable", True))
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        if not self.enabled:
            raise RuntimeError("LLM advisor is disabled")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        is_local = os.path.isdir(self.model_path)
        common = {"trust_remote_code": True}
        if is_local or self.offline_mode:
            common["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **common)

        model_kwargs = dict(common)
        accelerate_available = False
        try:
            import accelerate  # noqa: F401
            accelerate_available = True
        except ImportError:
            pass

        bnb_available = False
        try:
            import bitsandbytes  # noqa: F401
            bnb_available = True
        except ImportError:
            pass

        quant_mode = self.quantization
        if quant_mode == "auto":
            quant_mode = "4bit" if (accelerate_available and bnb_available and torch.cuda.is_available()) else "none"

        if quant_mode == "4bit" and accelerate_available and bnb_available:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = self.device_map
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16 if self.dtype in {"fp16", "float16"} else torch.bfloat16
            if accelerate_available:
                model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if torch.cuda.is_available() and not accelerate_available and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.cuda()
        self.model.eval()
        self._loaded = True

    @staticmethod
    def _normalize_state(current_state: Dict) -> Dict[str, float]:
        defaults = {
            "accept_quantile": 0.90,
            "margin_quantile": 0.20,
            "delta_fused": 0.10,
            "delta_margin": 0.10,
        }
        state = dict(defaults)
        for key in defaults:
            if key in current_state:
                try:
                    value = float(current_state[key])
                    if math.isfinite(value):
                        state[key] = value
                except (TypeError, ValueError):
                    pass
        if "rho" in current_state and "delta_fused" not in current_state:
            try:
                value = float(current_state["rho"])
                if math.isfinite(value):
                    state["delta_fused"] = value
            except (TypeError, ValueError):
                pass
        return state

    def _build_prompt(self, current_state: Dict, performance: Dict, objectives: Dict,
                      rag_context: str = "", history: Optional[list] = None) -> str:
        state = self._normalize_state(current_state)
        closed = float(performance.get("closed_acc", 0.0))
        open_auc = float(performance.get("open_auc", 0.0))
        reject = float(performance.get("reject_rate", 0.0))
        closed_gap = max(0.0, float(objectives["min_closed_acc"]) - closed)
        open_gap = max(0.0, float(objectives["target_open_auc"]) - open_auc)

        prompt = f"""### Current authentication state
Ac={closed:.4f}, target={float(objectives['min_closed_acc']):.4f}, gap={closed_gap:.4f}
Ao={open_auc:.4f}, target={float(objectives['target_open_auc']):.4f}, gap={open_gap:.4f}
R={reject:.4f} (auxiliary only)

### Current threshold/search state
{json.dumps(state, sort_keys=True)}

### Safety constraints
Safe ranges: {json.dumps(PARAM_RANGES)}
Maximum step sizes: {json.dumps(MAX_STEP_SIZE)}
"""
        if rag_context:
            prompt += f"\n### Retrieved historical evidence\n{rag_context}\n"
        if history:
            compact = history[-3:]
            prompt += f"\n### Recent operating-point history\n{json.dumps(compact, default=str)}\n"
        prompt += "\nReturn one conservative JSON proposal."
        return prompt

    def get_guidance(self, current_state: Dict, performance: Dict, objectives: Dict,
                     rag_context: str = "", history: Optional[list] = None) -> Optional[GuidanceAdvice]:
        if not self.enabled:
            return None
        self._lazy_load()
        user_prompt = self._build_prompt(current_state, performance, objectives, rag_context, history)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt")
        if not hasattr(self.model, "hf_device_map"):
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_response(raw_text, current_state)

    def _extract_json(self, text: str) -> Optional[Dict]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None

    def _parse_response(self, text: str, current_state: Dict) -> Optional[GuidanceAdvice]:
        data = self._extract_json(text)
        if data is None:
            return None
        return self._validate_advice(data, current_state)

    def _validate_advice(self, data: Dict, current_state: Dict) -> GuidanceAdvice:
        """First deterministic validation pass described in the manuscript."""
        state = self._normalize_state(current_state)
        raw_changes = data.get("param_changes", {})
        if not isinstance(raw_changes, dict):
            raw_changes = {}

        validated: Dict[str, float] = {}
        notes: list[str] = []
        adjusted = False

        for raw_key, raw_value in raw_changes.items():
            key = ALIASES.get(raw_key, raw_key)
            if key not in PARAM_RANGES:
                notes.append(f"unsupported:{raw_key}")
                adjusted = True
                continue
            try:
                proposed = float(raw_value)
            except (TypeError, ValueError):
                notes.append(f"non_numeric:{raw_key}")
                adjusted = True
                continue
            if not math.isfinite(proposed):
                notes.append(f"non_finite:{raw_key}")
                adjusted = True
                continue

            current = state[key]
            max_step = MAX_STEP_SIZE[key]
            delta = proposed - current
            if abs(delta) > max_step:
                proposed = current + (max_step if delta > 0 else -max_step)
                notes.append(f"step_clamp:{key}")
                adjusted = True

            lo, hi = PARAM_RANGES[key]
            clipped = min(hi, max(lo, proposed))
            if clipped != proposed:
                notes.append(f"range_clamp:{key}")
                adjusted = True
            validated[key] = float(clipped)

        try:
            confidence = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = min(1.0, max(0.0, confidence))

        return GuidanceAdvice(
            proposal_id=str(data.get("proposal_id") or uuid.uuid4().hex),
            analysis=str(data.get("analysis", "")),
            param_changes=validated,
            reasoning=str(data.get("reasoning", "")),
            confidence=confidence,
            requires_reprocessing=False,
            raw_param_changes=dict(raw_changes),
            first_pass_adjusted=adjusted,
            validation_notes=notes,
        )

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

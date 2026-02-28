from dataclasses import dataclass
from collections import deque
import re
from typing import Dict, Tuple

from src.prompt_rl_gsm8k.autogen_env import AutoGenEpisodeResult


@dataclass
class AutoGenRewardConfig:
    prompt_weight: float = 0.30
    protocol_weight: float = 0.25
    task_weight: float = 0.45
    success_weight_in_task: float = 0.75
    efficiency_weight_in_task: float = 0.25
    target_prompt_words: int = 260
    target_protocol_words: int = 380
    max_rounds: int = 6
    role_names: tuple[str, ...] = ("strategist", "calculator", "verifier")
    adaptive_weights: bool = True
    adaptive_warmup: int = 20
    adaptive_history_size: int = 200
    adaptive_lr: float = 0.50
    min_component_weight: float = 0.15


class AutoGenRewardComputer:
    def __init__(self, config: AutoGenRewardConfig | None = None) -> None:
        self.config = config or AutoGenRewardConfig()
        self.episode_count = 0
        self.prompt_history = deque(maxlen=self.config.adaptive_history_size)
        self.protocol_history = deque(maxlen=self.config.adaptive_history_size)
        self.task_history = deque(maxlen=self.config.adaptive_history_size)

    def compute(
        self,
        result: AutoGenEpisodeResult,
        protocol_text: str,
        generated_prompt: str,
    ) -> Tuple[float, Dict[str, float]]:
        self.episode_count += 1

        prompt_score, prompt_meta = self._compute_prompt_quality(generated_prompt)
        protocol_score, protocol_meta = self._compute_protocol_quality(protocol_text, result)
        task_score, task_meta = self._compute_task_score(result)

        weights = self._get_component_weights()

        total = (
            weights["prompt"] * prompt_score
            + weights["protocol"] * protocol_score
            + weights["task"] * task_score
        )

        self.prompt_history.append(float(prompt_score))
        self.protocol_history.append(float(protocol_score))
        self.task_history.append(float(task_score))

        details = {
            "prompt_score": prompt_score,
            "protocol_score": protocol_score,
            "task_score": task_score,
            "weight_prompt": weights["prompt"],
            "weight_protocol": weights["protocol"],
            "weight_task": weights["task"],
            "prompt_meta": prompt_meta,
            "protocol_meta": protocol_meta,
            "task_meta": task_meta,
            "total": total,
        }
        return total, details

    def _get_component_weights(self) -> Dict[str, float]:
        base = {
            "prompt": float(self.config.prompt_weight),
            "protocol": float(self.config.protocol_weight),
            "task": float(self.config.task_weight),
        }
        base = self._normalize_weights(base)

        if not self.config.adaptive_weights:
            return base
        if self.episode_count <= self.config.adaptive_warmup:
            return base
        if len(self.task_history) < 8:
            return base

        prompt_signal = max(0.0, self._safe_corr(list(self.prompt_history), list(self.task_history)))
        protocol_signal = max(0.0, self._safe_corr(list(self.protocol_history), list(self.task_history)))
        task_signal = 1.0

        boosted = {
            "prompt": base["prompt"] * (1.0 + self.config.adaptive_lr * prompt_signal),
            "protocol": base["protocol"] * (1.0 + self.config.adaptive_lr * protocol_signal),
            "task": base["task"] * (1.0 + self.config.adaptive_lr * task_signal),
        }
        boosted = self._normalize_weights(boosted)

        floor = max(0.0, min(0.30, self.config.min_component_weight))
        for key in boosted:
            boosted[key] = max(floor, boosted[key])
        return self._normalize_weights(boosted)

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 1e-8:
            k = len(weights)
            return {name: 1.0 / k for name in weights}
        return {name: max(0.0, float(v)) / total for name, v in weights.items()}

    @staticmethod
    def _safe_corr(xs: list[float], ys: list[float]) -> float:
        n = min(len(xs), len(ys))
        if n < 3:
            return 0.0
        x = xs[-n:]
        y = ys[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        vx = sum((v - mx) ** 2 for v in x) / n
        vy = sum((v - my) ** 2 for v in y) / n
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
        return max(-1.0, min(1.0, cov / ((vx * vy) ** 0.5)))

    def _compute_collaboration_score(self, result: AutoGenEpisodeResult) -> float:
        if not result.transcript:
            return 0.0

        role_hits = 0
        for role_name in self.config.role_names:
            tag = f"[{role_name}]"
            if any(tag in line.lower() for line in result.transcript):
                role_hits += 1

        role_coverage = role_hits / max(1, len(self.config.role_names))

        json_lines = 0
        for line in result.transcript:
            content = line.split("]", 1)[-1].strip()
            if "{" in content and "}" in content:
                json_lines += 1
        json_ratio = json_lines / max(1, len(result.transcript))

        score = 0.7 * role_coverage + 0.3 * min(1.0, json_ratio * 2.0)
        return max(0.0, min(1.0, score))

    def _compute_prompt_quality(self, generated_prompt: str) -> Tuple[float, Dict[str, float]]:
        text = generated_prompt.strip()
        words = len(text.split())
        if words == 0:
            return 0.0, {
                "words": 0.0,
                "structure": 0.0,
                "clarity": 0.0,
                "diversity": 0.0,
                "generic_penalty": 0.0,
                "redundancy_penalty": 0.0,
            }

        dist = abs(words - self.config.target_prompt_words) / max(1, self.config.target_prompt_words)
        length_score = max(0.0, 1.0 - dist)

        has_sections = 1.0 if ("\n- " in text or "\n1." in text or "##" in text or "\n###" in text) else 0.0
        clarity_keywords = ["role", "commun", "verify", "error", "termination", "consistency"]
        hit = sum(1 for keyword in clarity_keywords if keyword in text.lower())
        clarity_score = min(1.0, hit / 4.0)

        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if tokens:
            diversity = min(1.0, len(set(tokens)) / max(1, len(tokens) * 0.6))
        else:
            diversity = 0.0

        generic_penalty = self._negative_keyword_penalty(
            text,
            [
                "be clear",
                "be concise",
                "communicate effectively",
                "work together",
                "collaborate",
            ],
        )
        redundancy_penalty = self._text_redundancy_penalty(text)

        total = (
            0.28 * length_score
            + 0.22 * has_sections
            + 0.28 * clarity_score
            + 0.22 * diversity
            - 0.20 * generic_penalty
            - 0.20 * redundancy_penalty
        )
        total = max(0.0, min(1.0, total))
        return total, {
            "words": float(words),
            "structure": has_sections,
            "clarity": clarity_score,
            "diversity": diversity,
            "generic_penalty": generic_penalty,
            "redundancy_penalty": redundancy_penalty,
        }

    def _compute_protocol_quality(
        self,
        protocol_text: str,
        result: AutoGenEpisodeResult,
    ) -> Tuple[float, Dict[str, float]]:
        words = len(protocol_text.split())
        dist = abs(words - self.config.target_protocol_words) / max(1, self.config.target_protocol_words)
        length_score = max(0.0, 1.0 - dist)

        role_hit = sum(1 for role_name in self.config.role_names if role_name in protocol_text.lower())
        role_score = role_hit / max(1, len(self.config.role_names))

        collaboration_score = self._compute_collaboration_score(result)

        lowered = protocol_text.lower()
        schema_keywords = ["json", "field", "schema", "message", "format"]
        schema_hit = sum(1 for key in schema_keywords if key in lowered)
        schema_score = min(1.0, schema_hit / 4.0)

        action_keywords = ["must", "should", "if", "then", "when", "otherwise"]
        action_hit = sum(1 for key in action_keywords if key in lowered)
        actionability_score = min(1.0, action_hit / 5.0)

        generic_penalty = self._negative_keyword_penalty(
            protocol_text,
            [
                "communicate clearly",
                "do your best",
                "help each other",
                "be concise",
                "collaborate efficiently",
            ],
        )
        transcript_redundancy = self._transcript_redundancy_penalty(result)

        total = (
            0.18 * length_score
            + 0.22 * role_score
            + 0.24 * collaboration_score
            + 0.18 * schema_score
            + 0.18 * actionability_score
            - 0.18 * generic_penalty
            - 0.16 * transcript_redundancy
        )
        total = max(0.0, min(1.0, total))
        return total, {
            "words": float(words),
            "length": length_score,
            "role_coverage": role_score,
            "collaboration": collaboration_score,
            "schema": schema_score,
            "actionability": actionability_score,
            "generic_penalty": generic_penalty,
            "transcript_redundancy": transcript_redundancy,
        }

    def _compute_task_score(self, result: AutoGenEpisodeResult) -> Tuple[float, Dict[str, float]]:
        if result.error is not None:
            return 0.0, {"success": 0.0, "efficiency": 0.0, "early_stop": 0.0}

        success_score = 1.0 if result.success else 0.0
        rounds = max(1, result.rounds)
        max_rounds = max(1, self.config.max_rounds)
        efficiency_score = max(0.0, 1.0 - ((rounds - 1) / max(1, max_rounds - 1)))
        early_stop_score = 1.0 if result.early_stop else 0.0
        blended_efficiency = 0.65 * efficiency_score + 0.35 * early_stop_score

        answer_format_score = 1.0 if self._is_numeric_like(result.final_answer) else 0.0
        stability_score = max(0.0, 1.0 - self._transcript_redundancy_penalty(result))

        core = (
            self.config.success_weight_in_task * success_score
            + self.config.efficiency_weight_in_task * blended_efficiency
        )
        aux = 0.6 * answer_format_score + 0.4 * stability_score
        total = 0.9 * core + 0.1 * aux

        return total, {
            "success": success_score,
            "efficiency": efficiency_score,
            "early_stop": early_stop_score,
            "answer_format": answer_format_score,
            "stability": stability_score,
        }

    @staticmethod
    def _negative_keyword_penalty(text: str, negative_phrases: list[str]) -> float:
        lowered = text.lower()
        if not negative_phrases:
            return 0.0
        hits = sum(1 for phrase in negative_phrases if phrase in lowered)
        return min(1.0, hits / max(1, len(negative_phrases) // 2))

    @staticmethod
    def _text_redundancy_penalty(text: str) -> float:
        lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return 0.0
        unique_ratio = len(set(lines)) / len(lines)
        repeat_ratio = 1.0 - unique_ratio
        return max(0.0, min(1.0, repeat_ratio))

    @staticmethod
    def _transcript_redundancy_penalty(result: AutoGenEpisodeResult) -> float:
        transcript = result.transcript or []
        if len(transcript) < 2:
            return 0.0

        cleaned = []
        for line in transcript:
            content = line.split("]", 1)[-1].strip().lower()
            cleaned.append(content)

        consecutive_repeat = 0
        for prev, cur in zip(cleaned[:-1], cleaned[1:]):
            if prev == cur:
                consecutive_repeat += 1
        consecutive_ratio = consecutive_repeat / max(1, len(cleaned) - 1)

        unique_ratio = len(set(cleaned)) / max(1, len(cleaned))
        global_repeat_ratio = 1.0 - unique_ratio

        penalty = 0.6 * consecutive_ratio + 0.4 * global_repeat_ratio
        return max(0.0, min(1.0, penalty))

    @staticmethod
    def _is_numeric_like(text: str) -> bool:
        raw = str(text).strip()
        if not raw:
            return False
        normalized = raw.replace(",", "")
        if "####" in normalized:
            normalized = normalized.split("####")[-1].strip()
        return re.fullmatch(r"-?\d+(?:\.\d+)?", normalized) is not None

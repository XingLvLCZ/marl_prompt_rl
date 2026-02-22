from dataclasses import dataclass
from typing import Dict, Tuple

from src.prompt_rl_autogen.autogen_env import AutoGenEpisodeResult


@dataclass
class AutoGenRewardConfig:
    success_weight: float = 1.0
    efficiency_weight: float = 0.25
    protocol_length_weight: float = 0.1
    target_protocol_words: int = 450
    max_rounds: int = 6


class AutoGenRewardComputer:
    def __init__(self, config: AutoGenRewardConfig | None = None) -> None:
        self.config = config or AutoGenRewardConfig()

    def compute(self, result: AutoGenEpisodeResult, protocol_text: str) -> Tuple[float, Dict[str, float]]:
        success_score = 1.0 if result.success else 0.0
        if result.error is not None:
            success_score = -0.5

        rounds = max(1, result.rounds)
        efficiency_score = max(0.0, 1.0 - (rounds / max(1, self.config.max_rounds)))

        protocol_words = len(protocol_text.split())
        dist = abs(protocol_words - self.config.target_protocol_words) / max(1, self.config.target_protocol_words)
        protocol_length_score = max(0.0, 1.0 - dist)

        total = (
            self.config.success_weight * success_score
            + self.config.efficiency_weight * efficiency_score
            + self.config.protocol_length_weight * protocol_length_score
        )

        details = {
            "success": success_score,
            "efficiency": efficiency_score,
            "protocol_length": protocol_length_score,
            "total": total,
        }
        return total, details

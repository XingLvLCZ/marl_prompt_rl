from dataclasses import dataclass
from typing import List


@dataclass
class AutoGenModelConfig:
    model: str = "Qwen/Qwen3-14B"
    temperature: float = 0.2
    max_rounds: int = 6


@dataclass
class AutoGenTask:
    task_id: str
    prompt: str
    expected_answer: str


def build_default_tasks() -> List[AutoGenTask]:
    return [
        AutoGenTask(
            task_id="arith_1",
            prompt=(
                "Solve collaboratively. Compute (17 * 6) + 25. "
                "The final responder must output exactly: FINAL_ANSWER: <value>."
            ),
            expected_answer="127",
        ),
        AutoGenTask(
            task_id="logic_1",
            prompt=(
                "A team has 4 members. Each member shakes hands exactly once with every other member. "
                "How many handshakes total? Output exactly: FINAL_ANSWER: <value>."
            ),
            expected_answer="6",
        ),
        AutoGenTask(
            task_id="arith_2",
            prompt=(
                "Solve collaboratively. Compute 144 / 12 + 9 * 3. "
                "The final responder must output exactly: FINAL_ANSWER: <value>."
            ),
            expected_answer="39",
        ),
    ]

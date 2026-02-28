from dataclasses import dataclass
from typing import List


@dataclass
class AutoGenModelConfig:
    model: str = "Qwen/Qwen3-14B"
    temperature: float = 0.2
    max_rounds: int = 9
    strategist_name: str = "strategist"
    calculator_name: str = "calculator"
    verifier_name: str = "verifier"


@dataclass
class AutoGenDatasetConfig:
    source: str = "hf"
    local_path: str = ""
    hf_path: str = ""
    hf_name: str = ""
    hf_split: str = "train"
    task_id_field: str = "task_id"
    prompt_field: str = "prompt"
    answer_field: str = "expected_answer"
    task_type: str = "generic"  # generic/gsm8k
    limit: int = 0
    shuffle: bool = False
    seed: int = 42
    train_ratio: float = 0.8


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

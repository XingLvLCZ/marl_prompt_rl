from __future__ import annotations

import threading
from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.provider.config import API_KEY, API_URL
from src.provider.qwen import QwenProvider
from src.generator.protocol_generator import ProtocolGenerator
from src.prompt_rl_gsm8k.autogen_env import AutoGenEpisodeEnv
from src.prompt_rl_gsm8k.config import AutoGenTask
from src.prompt_rl_gsm8k.reward import AutoGenRewardComputer, AutoGenRewardConfig


@dataclass
class RewardAdapterRuntime:
    prompt_history_dir: str
    protocol_history_dir: str
    eval_outputs_dir: str
    protocol_model: str
    env_model: str
    env_max_rounds: int
    env_temperature: float
    strategist_name: str
    calculator_name: str
    verifier_name: str


class TRLGSM8KRewardAdapter:
    _thread_lock = threading.Lock()

    def __init__(self, runtime: RewardAdapterRuntime) -> None:
        self.runtime = runtime
        self.prompt_history_dir = Path(runtime.prompt_history_dir)
        self.protocol_history_dir = Path(runtime.protocol_history_dir)
        self.eval_outputs_dir = Path(runtime.eval_outputs_dir)
        self.prompt_history_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_history_dir.mkdir(parents=True, exist_ok=True)
        self.eval_outputs_dir.mkdir(parents=True, exist_ok=True)

        provider = QwenProvider(api_key=API_KEY, base_url=API_URL, model=runtime.protocol_model)
        self.protocol_generator = ProtocolGenerator(provider=provider)
        self.env = AutoGenEpisodeEnv(
            model=runtime.env_model,
            api_key=API_KEY,
            base_url=API_URL,
            max_rounds=runtime.env_max_rounds,
            temperature=runtime.env_temperature,
            strategist_name=runtime.strategist_name,
            calculator_name=runtime.calculator_name,
            verifier_name=runtime.verifier_name,
        )
        self.reward_computer = AutoGenRewardComputer(
            AutoGenRewardConfig(
                max_rounds=runtime.env_max_rounds,
                role_names=(runtime.strategist_name, runtime.calculator_name, runtime.verifier_name),
            )
        )
        self.global_step = 0
        self.process_lock_file = self.eval_outputs_dir / ".reward_process.lock"
        self.process_lock_file.touch(exist_ok=True)

    def __call__(self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
        with self._sequential_guard():
            self.global_step += 1
            batch_size = len(completions)

            task_ids = self._expand_field(kwargs.get("task_id"), batch_size, default_prefix="task")
            task_prompts = self._expand_field(kwargs.get("task_prompt"), batch_size, default_prefix="prompt")
            expected_answers = self._expand_field(kwargs.get("expected_answer"), batch_size, default_prefix="answer")

            rewards: List[float] = []
            for index in range(batch_size):
                completion_text = self._completion_to_text(completions[index]).strip()
                if not completion_text:
                    rewards.append(-1.0)
                    continue

                sample_id = f"{self.global_step}_{index + 1}"
                prompt_path = self.prompt_history_dir / f"prompt_{sample_id}.md"
                prompt_path.write_text(completion_text, encoding="utf-8")

                try:
                    protocol_path = self.protocol_generator.generate_protocol(
                        prompt=completion_text,
                        protocol_name=f"protocol_{sample_id}",
                        save_dir=str(self.protocol_history_dir),
                    )
                    protocol_text = protocol_path.read_text(encoding="utf-8")

                    task = AutoGenTask(
                        task_id=str(task_ids[index]),
                        prompt=str(task_prompts[index]),
                        expected_answer=str(expected_answers[index]),
                    )
                    result = self.env.run_episode(protocol_text=protocol_text, task=task)
                    reward, details = self.reward_computer.compute(
                        result=result,
                        protocol_text=protocol_text,
                        generated_prompt=completion_text,
                    )
                    rewards.append(float(reward))

                    report = self._build_eval_report(
                        sample_id=sample_id,
                        task=task,
                        reward=reward,
                        details=details,
                        result=result,
                        prompt_path=prompt_path,
                        protocol_path=protocol_path,
                    )
                    (self.eval_outputs_dir / f"episode_{sample_id}.md").write_text(report, encoding="utf-8")
                except Exception as exc:
                    rewards.append(-1.0)
                    err = (
                        f"# Episode {sample_id}\n\n"
                        f"- task_id: {task_ids[index]}\n"
                        f"- error: {str(exc)}\n"
                        f"- reward: -1.0\n"
                    )
                    (self.eval_outputs_dir / f"episode_{sample_id}.md").write_text(err, encoding="utf-8")

            return rewards

    @contextmanager
    def _sequential_guard(self):
        with self._thread_lock:
            with self.process_lock_file.open("r+") as fp:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _expand_field(value: Any, n: int, default_prefix: str) -> List[Any]:
        if isinstance(value, list):
            if len(value) == n:
                return value
            if len(value) == 1:
                return value * n
            return [value[i % len(value)] for i in range(n)]
        if isinstance(value, tuple):
            seq = list(value)
            if len(seq) == n:
                return seq
            if len(seq) == 1:
                return seq * n
            return [seq[i % len(seq)] for i in range(n)]
        if value is None:
            return [f"{default_prefix}_{i + 1}" for i in range(n)]
        return [value for _ in range(n)]

    @staticmethod
    def _completion_to_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            for key in ("content", "text", "completion"):
                if key in completion and completion[key] is not None:
                    return str(completion[key])
            return str(completion)
        if isinstance(completion, list):
            parts: List[str] = []
            for item in completion:
                parts.append(TRLGSM8KRewardAdapter._completion_to_text(item))
            return "\n".join([x for x in parts if x]).strip()
        return str(completion)

    @staticmethod
    def _build_eval_report(
        sample_id: str,
        task: AutoGenTask,
        reward: float,
        details: Dict[str, Any],
        result: Any,
        prompt_path: Path,
        protocol_path: Path,
    ) -> str:
        return (
            f"# Episode {sample_id}\n\n"
            f"- task_id: {task.task_id}\n"
            f"- success: {result.success}\n"
            f"- rounds: {result.rounds}\n"
            f"- reward: {reward:.6f}\n"
            f"- prompt_score: {details.get('prompt_score', 0.0):.6f}\n"
            f"- protocol_score: {details.get('protocol_score', 0.0):.6f}\n"
            f"- task_score: {details.get('task_score', 0.0):.6f}\n"
            f"- final_answer: {result.final_answer}\n"
            f"- prompt_file: {prompt_path}\n"
            f"- protocol_file: {protocol_path}\n"
            f"- error: {result.error}\n"
        )

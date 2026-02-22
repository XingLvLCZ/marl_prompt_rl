from pathlib import Path
import random

import torch
from torch.optim import AdamW

from src.provider.config import API_KEY, API_URL
from src.provider.qwen import QwenProvider
from src.generator.prompt_generator import PromptGenerator
from src.generator.protocol_generator import ProtocolGenerator

from src.prompt_rl_autogen.autogen_env import AutoGenEpisodeEnv
from src.prompt_rl_autogen.config import AutoGenModelConfig, build_default_tasks
from src.prompt_rl_autogen.prompt_template import build_protocol_generation_prompt
from src.prompt_rl_autogen.reward import AutoGenRewardComputer, AutoGenRewardConfig


ROOT = Path("src/prompt_rl_autogen")
PROMPT_HISTORY_DIR = ROOT / "prompt_history"
PROTOCOL_HISTORY_DIR = ROOT / "protocol_history"
CHECKPOINT_DIR = ROOT / "checkpoints"
EVAL_OUTPUTS_DIR = ROOT / "eval_outputs"


def _prepare_dirs() -> None:
    PROMPT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def train(
    epochs: int = 5,
    base_model_path: str = "/root/aicloud-data/llms/Qwen3-4B-Instruct-2507",
    save_every: int = 5,
) -> None:
    _prepare_dirs()

    cfg = AutoGenModelConfig()
    tasks = build_default_tasks()

    prompt_generator = PromptGenerator(
        model_name=base_model_path,
        device_map="auto"
    )

    protocol_provider = QwenProvider(api_key=API_KEY, base_url=API_URL, model=cfg.model)
    protocol_generator = ProtocolGenerator(provider=protocol_provider)

    env = AutoGenEpisodeEnv(
        model=cfg.model,
        api_key=API_KEY,
        base_url=API_URL,
        max_rounds=cfg.max_rounds,
        temperature=cfg.temperature,
    )

    reward_computer = AutoGenRewardComputer(
        AutoGenRewardConfig(max_rounds=cfg.max_rounds)
    )

    optimizer = AdamW(prompt_generator.model.parameters(), lr=3e-5)

    for ep in range(epochs):
        task = random.choice(tasks)

        protocol_prompt_task = build_protocol_generation_prompt(task, prior="")
        generated_prompt, _, _, avg_log_prob = prompt_generator.generate_prompt(
            prompt=protocol_prompt_task,
            temperature=0.7,
            max_new_tokens=2048,
        )

        prompt_path = PROMPT_HISTORY_DIR / f"prompt_{ep + 1}.md"
        prompt_path.write_text(generated_prompt, encoding="utf-8")

        protocol_path = protocol_generator.generate_protocol(
            prompt=generated_prompt,
            protocol_name=f"protocol_{ep + 1}",
            save_dir=str(PROTOCOL_HISTORY_DIR),
        )
        protocol_text = protocol_path.read_text(encoding="utf-8")

        result = env.run_episode(protocol_text=protocol_text, task=task)
        reward, details = reward_computer.compute(result=result, protocol_text=protocol_text)

        reward_tensor = torch.tensor(reward, dtype=avg_log_prob.dtype, device=avg_log_prob.device)
        loss = -(reward_tensor * avg_log_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_report = (
            f"# Episode {ep + 1}\n\n"
            f"- task_id: {task.task_id}\n"
            f"- success: {result.success}\n"
            f"- final_answer: {result.final_answer}\n"
            f"- expected_answer: {task.expected_answer}\n"
            f"- rounds: {result.rounds}\n"
            f"- reward: {reward:.4f}\n"
            f"- reward_details: {details}\n"
            f"- error: {result.error}\n"
        )
        (EVAL_OUTPUTS_DIR / f"episode_{ep + 1}.md").write_text(eval_report, encoding="utf-8")

        print(
            f"[Episode {ep + 1}/{epochs}] "
            f"task={task.task_id} success={result.success} reward={reward:.4f}"
        )

        if (ep + 1) % save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"prompt_generator_ep{ep + 1}"
            prompt_generator.model.save_pretrained(str(ckpt_path))


if __name__ == "__main__":
    train()

"""Evaluate a trained LoRA adapter checkpoint on the guessing game."""

from pathlib import Path
from typing import Dict, List

import torch

from src.prompt_rl_guess.guess_agent import GuessNumAgent
from src.prompt_rl_guess.pz_guess_env import GuessGamePettingZooEnv
from src.llm.config import API_KEY, API_URL
from src.llm.qwen import QwenProvider
from src.prompt_rl_guess.generator import PromptGenerator
from src.prompt_rl_guess.pz_guess_runner import GuessGameRunner
from src.protocol.generator import ProtocolGenerator
from src.protocol.loader import ProtocolLoader


BASE_MODEL_PATH = "/root/aicloud-data/llms/Qwen3-1.7B"
CHECKPOINT_DIR = Path("src/prompt_rl_guess/checkpoints")
EVAL_OUTPUT_DIR = Path("src/prompt_rl_guess/eval_outputs")
NUM_CHOICES = 10
NUM_AGENTS = 2
MAX_STEPS = 10
ROUNDS = 3

TASK_DESCRIPTION = f"""
Game:
- game_type: guessing number game
- range_size: {NUM_CHOICES} (from 0 to {NUM_CHOICES - 1})
- num_agents: {NUM_AGENTS}

Game description:
Multiple agents take turns to guess a secret target number within the specified range.
After each guess, agents can follow a protocol and send a message to others.
In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

Your task:
**Generate a protocol-generation prompt** that guides a LLM to generate the protocol.
Reply ONLY the content of the prompt.
"""

HARD_CONSTRAINT = """
==================== HARD REQUIREMENTS ====================

1. The protocol MUST be written in Markdown.
2. The protocol MUST explicitly require agents to output ONLY valid JSON.
3. Each agent message MUST contain a field named "next_guess" with an integer value.

==================== OUTPUT CONSTRAINT ====================

- Output ONLY the content of the protocol.
- Do NOT include reasoning, analysis, or <think> blocks.
- Do NOT include any text before or after the protocol.
"""


def load_prompt_generator(adapter_path: Path) -> PromptGenerator:
    """Load LoRA adapter from checkpoint."""
    generator = PromptGenerator(
        model_name=BASE_MODEL_PATH,
        adapter_path=str(adapter_path)
    )
    generator.model.eval()
    return generator


def build_protocol(prompt: str, round_idx: int, provider: QwenProvider, protocol_generator: ProtocolGenerator) -> str:
    """Generate protocol markdown from the prompt and persist it for inspection."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_name = f"eval_protocol_round{round_idx}"
    save_dir = EVAL_OUTPUT_DIR / "protocols"
    save_dir.mkdir(parents=True, exist_ok=True)

    full_prompt = prompt + "\n\n" + HARD_CONSTRAINT
    protocol_generator.generate_protocol(
        prompt=full_prompt,
        protocol_name=protocol_name,
        save_dir=str(save_dir),
    )
    protocol_text = ProtocolLoader.get_protocol_text(protocol_name=protocol_name, protocol_dir=str(save_dir))

    prompt_path = EVAL_OUTPUT_DIR / f"prompt_round{round_idx}.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    return protocol_text


def run_episode(protocol: str, target: int, provider: QwenProvider):
    """Run one episode for a fixed target and return the trajectory."""
    agents = {
        f"agent_{i}": GuessNumAgent(
            agent_id=f"agent_{i}",
            provider=provider,
            protocol=protocol,
            initial_guess=0,
            num_choices=NUM_CHOICES,
        )
        for i in range(NUM_AGENTS)
    }
    env = GuessGamePettingZooEnv(
        agents=list(agents.keys()),
        target_range=NUM_CHOICES,
        max_steps=MAX_STEPS,
    )
    runner = GuessGameRunner(env=env, agents=agents)
    return runner.run_episode(target=target)


def summarize_trajectory(trajectory: List[Dict]) -> Dict[str, object]:
    agent_rewards: Dict[str, float] = {}
    for item in trajectory:
        agent_id = item["agent"]
        agent_rewards[agent_id] = agent_rewards.get(agent_id, 0.0) + float(item.get("reward", 0.0))

    success = bool(agent_rewards) and all(r > 0 for r in agent_rewards.values())
    steps = len(trajectory)
    return {"success": success, "steps": steps, "agent_rewards": agent_rewards}


def main() -> None:
    provider = QwenProvider(api_key=API_KEY, base_url=API_URL, model="Qwen/Qwen3-14B")
    protocol_generator = ProtocolGenerator(provider=provider)
    
    checkpoint_path = CHECKPOINT_DIR / "prompt_generator_ep40"
    # no lora
    # prompt_generator = PromptGenerator(model_name=BASE_MODEL_PATH)
    # with lora
    prompt_generator = load_prompt_generator(checkpoint_path)

    round_summaries: List[Dict[str, object]] = []

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n===== Round {round_idx}/{ROUNDS} =====")

        print("generating prompt...")
        with torch.no_grad():
            prompt = prompt_generator.generate_prompt_without_log_prob(TASK_DESCRIPTION)
        
        print("building protocol...")
        protocol = build_protocol(prompt, round_idx, provider, protocol_generator)

        target_results = []
        for target in range(NUM_CHOICES):
            trajectory = run_episode(protocol, target, provider)
            summary = summarize_trajectory(trajectory)
            target_results.append({"target": target, **summary})
            outcome = "success" if summary["success"] else "fail"
            print(
                f"target={target} -> {outcome}, steps={summary['steps']}, "
                f"agent_rewards={summary['agent_rewards']}"
            )

        successes = sum(1 for r in target_results if r["success"])
        avg_steps = sum(r["steps"] for r in target_results) / len(target_results)
        print(
            f"Round {round_idx} summary: success_rate={successes/NUM_CHOICES:.2f} "
            f"({successes}/{NUM_CHOICES}), avg_steps={avg_steps:.2f}"
        )

        round_summaries.append(
            {
                "round": round_idx,
                "successes": successes,
                "avg_steps": avg_steps,
                "targets": target_results,
            }
        )

    print("\n===== Overall =====")
    overall_successes = sum(rs["successes"] for rs in round_summaries)
    overall_tests = NUM_CHOICES * ROUNDS
    overall_avg_steps = sum(rs["avg_steps"] for rs in round_summaries) / len(round_summaries)
    print(
        f"success_rate={overall_successes/overall_tests:.2f} "
        f"({overall_successes}/{overall_tests}), avg_steps={overall_avg_steps:.2f}"
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from statistics import mean

from src.provider.config import API_KEY, API_URL
from src.provider.qwen import QwenProvider
from src.generator.qwen3_prompt_generator import PromptGenerator
from src.generator.protocol_generator import ProtocolGenerator

from src.prompt_rl_autogen.autogen_env import AutoGenEpisodeEnv
from src.prompt_rl_autogen.config import AutoGenModelConfig, build_default_tasks
from src.prompt_rl_autogen.prompt_template import build_protocol_generation_prompt


def evaluate_checkpoint(adapter_path: str, base_model_path: str = "/root/aicloud-data/llms/Qwen3-1.7B") -> dict:
    cfg = AutoGenModelConfig()
    tasks = build_default_tasks()

    generator = PromptGenerator(
        model_name=base_model_path,
        adapter_path=adapter_path,
    )
    provider = QwenProvider(api_key=API_KEY, base_url=API_URL, model=cfg.model)
    protocol_generator = ProtocolGenerator(provider=provider)
    env = AutoGenEpisodeEnv(
        model=cfg.model,
        api_key=API_KEY,
        base_url=API_URL,
        max_rounds=cfg.max_rounds,
        temperature=cfg.temperature,
    )

    success_flags = []
    round_counts = []

    tmp_dir = Path("src/prompt_rl_autogen/eval_outputs/protocols")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tasks, start=1):
        prompt_task = build_protocol_generation_prompt(task, prior="")
        generated_prompt = generator.generate_prompt_without_log_prob(prompt_task)
        protocol_path = protocol_generator.generate_protocol(
            prompt=generated_prompt,
            protocol_name=f"eval_protocol_{idx}",
            save_dir=str(tmp_dir),
        )
        protocol_text = protocol_path.read_text(encoding="utf-8")
        result = env.run_episode(protocol_text=protocol_text, task=task)

        success_flags.append(1.0 if result.success else 0.0)
        round_counts.append(float(result.rounds))

    return {
        "adapter_path": adapter_path,
        "tasks": len(tasks),
        "success_rate": mean(success_flags) if success_flags else 0.0,
        "avg_rounds": mean(round_counts) if round_counts else 0.0,
    }


def evaluate_all_checkpoints(checkpoint_root: str = "src/prompt_rl_autogen/checkpoints") -> None:
    ckpt_root = Path(checkpoint_root)
    if not ckpt_root.exists():
        print(f"checkpoint root not found: {ckpt_root}")
        return

    checkpoints = sorted([p for p in ckpt_root.iterdir() if p.is_dir()])
    if not checkpoints:
        print("no checkpoints found")
        return

    rows = []
    for ckpt in checkpoints:
        metrics = evaluate_checkpoint(str(ckpt))
        rows.append(metrics)
        print(metrics)

    report_lines = ["# AutoGen Checkpoint Eval\n"]
    for row in rows:
        report_lines.append(
            f"- {row['adapter_path']}: success_rate={row['success_rate']:.3f}, avg_rounds={row['avg_rounds']:.2f}"
        )
    Path("src/prompt_rl_autogen/eval_outputs/checkpoint_eval.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    evaluate_all_checkpoints()

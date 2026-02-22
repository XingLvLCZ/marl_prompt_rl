from src.generator.prompt_generator import PromptGenerator
from src.prompt_rl_autogen.config import build_default_tasks
from src.prompt_rl_autogen.prompt_template import build_protocol_generation_prompt


def load_lora_adapter(adapter_path: str, base_model: str = "/root/aicloud-data/llms/Qwen3-4B-Instruct-2507") -> PromptGenerator:
    return PromptGenerator(
        model_name=base_model,
        adapter_path=adapter_path,
    )


def infer_prompt(generator: PromptGenerator, task_index: int = 0) -> str:
    tasks = build_default_tasks()
    task = tasks[task_index % len(tasks)]
    task_prompt = build_protocol_generation_prompt(task, prior="")
    print(f"=== Inference Task ===\n{task_prompt}\n=== End of Task ===")
    return generator.generate_prompt_without_log_prob(task_prompt)


if __name__ == "__main__":
    adapter = ""
    base_model = "/root/aicloud-data/llms/Qwen3-4B-Instruct-2507"
    generator = load_lora_adapter(adapter_path=adapter, base_model=base_model)
    output = infer_prompt(generator, task_index=1)
    print(output)

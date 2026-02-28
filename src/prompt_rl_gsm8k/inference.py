from src.generator.prompt_generator import PromptGenerator
from src.prompt_rl_gsm8k.config import AutoGenDatasetConfig
from src.prompt_rl_gsm8k.prompt_template import build_protocol_generation_prompt
from src.prompt_rl_gsm8k.task_dataset import load_tasks


INFER_ADAPTER_PATH = "src/prompt_rl_gsm8k/checkpoints_trl/checkpoint-700"
INFER_BASE_MODEL = "/root/models/Qwen3-4B-Instruct-2507"
INFER_TASK_INDEX = 0

INFER_DATASET_CFG = AutoGenDatasetConfig(
    source="hf",  # default/local/hf
    local_path="",
    hf_path="/root/datasets/gsm8k",
    hf_name="main",
    hf_split="train",
    task_id_field="task_id",
    prompt_field="question",
    answer_field="answer",
    task_type="gsm8k",
    limit=0,
    shuffle=False,
    seed=42,
)


def load_lora_adapter(adapter_path: str, base_model: str = "/root/models/Qwen3-4B-Instruct-2507") -> PromptGenerator:
    return PromptGenerator(
        model_name=base_model,
        adapter_path=adapter_path,
    )


def infer_prompt(
    generator: PromptGenerator,
    agent_structure: str = "",
    task_overview: str = "",
) -> str:
    task_prompt = build_protocol_generation_prompt(prior='')
    # print(f"=== Inference Task ===\n{task_prompt}\n=== End of Task ===")
    return generator.generate_prompt_without_log_prob(task_prompt)


if __name__ == "__main__":
    # generator = load_lora_adapter(adapter_path="", base_model=INFER_BASE_MODEL)
    # output = infer_prompt(generator)
    # print(output)
    # with open("inference_base_model.md", "w") as f:
    #     f.write(output)

    generator = load_lora_adapter(adapter_path=INFER_ADAPTER_PATH, base_model=INFER_BASE_MODEL)
    output = infer_prompt(generator)
    print(output)
    with open("inference_rl_model.md", "w") as f:
        f.write(output)
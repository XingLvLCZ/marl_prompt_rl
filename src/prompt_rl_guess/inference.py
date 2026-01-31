"""
LoRA Inference Helper - 使用已训练的LoRA adapter进行推理
"""
import torch
from src.prompt_rl_guess.generator import PromptGenerator


def load_lora_adapter(adapter_path: str, base_model: str = "/root/aicloud-data/llms/Qwen3-1.7B"):
    """
    加载LoRA adapter用于推理
    
    Args:
        adapter_path: LoRA adapter保存路径 (例如: src/prompt_rl_guess/checkpoints/prompt_generator_ep30)
        base_model: 基础模型路径
    
    Returns:
        PromptGenerator 实例
    """
    return PromptGenerator(
        model_name=base_model,
        adapter_path=adapter_path
    )


def infer_prompt(generator: PromptGenerator, task: str, temperature: float = 0.5, max_tokens: int = 1024) -> str:
    """
    使用LoRA模型进行推理
    
    Args:
        generator: PromptGenerator实例
        task: 输入任务描述
        temperature: 采样温度
        max_tokens: 最大生成token数
    
    Returns:
        生成的提示文本
    """
    with torch.no_grad():
        result = generator.generate_prompt_without_log_prob(task)
    return result


if __name__ == "__main__":
    adapter_path = "src/prompt_rl_guess/checkpoints/prompt_generator_ep20"
    generator = load_lora_adapter(adapter_path)
    
    task = "Generate a protocol-generation prompt for a guessing game with multiple agents."
    result = infer_prompt(generator, task, temperature=0.5)
    print("生成结果:")
    print(result)

"""
LoRA Inference Helper - 使用已训练的LoRA adapter进行推理
"""
import os
import time
import torch
from src.generator.qwen3_prompt_generator import PromptGenerator


def load_lora_adapter(adapter_path: str, base_model: str = "/root/models/Qwen3-1.7B"):
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


def measure_inference_time(adapter_path, task, num_runs=10, warmup_runs=2, temperature=0.5, max_tokens=1024):
    """
    加载模型（可选 adapter），多次运行推理并返回平均耗时（秒）
    adapter_path: 如果为 None 或空字符串，则只加载基础模型
    """
    # 处理空字符串情况：视为 None
    if adapter_path == "":
        adapter_path = None

    generator = PromptGenerator(
        model_name="/root/models/Qwen3-1.7B",
        adapter_path=adapter_path
    )

    # 预热
    for _ in range(warmup_runs):
        with torch.no_grad():
            generator.generate_prompt_without_log_prob(task)
    torch.cuda.synchronize()

    # 正式计时
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            generator.generate_prompt_without_log_prob(task)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # 清理
    del generator
    torch.cuda.empty_cache()
    return sum(times) / len(times)


if __name__ == "__main__":
#     adapter_path = "src/prompt_rl_guess/checkpoints/prompt_generator_ep80"
#     generator = load_lora_adapter(adapter_path)
    
#     task = """
# Game:
# - game_type: guessing number game
# - range_size: 10 (from 0 to 9)
# - num_agents: 2

# Game description:
# Multiple agents take turns to guess a secret target number within the specified range.
# After each guess, agents can follow a protocol and send a message to others.
# In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

# Your task:
# Generate a protocol-generation prompt that guides a LLM to generate the protocol.
# Reply ONLY the content of the prompt.
# """
#     result = infer_prompt(generator, task)
#     print("生成结果:")
#     print(result)
#     print(f"GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    checkpoints_dir = "src/prompt_rl_guess/checkpoints"
    task = """
Game:
- game_type: guessing number game
- range_size: 10 (from 0 to 9)
- num_agents: 2

Game description:
Multiple agents take turns to guess a secret target number within the specified range.
After each guess, agents can follow a protocol and send a message to others.
In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

Your task:
Generate a protocol-generation prompt that guides a LLM to generate the protocol.
Reply ONLY the content of the prompt.
"""

    results = {}

    # 1. 测试基础模型（无 adapter）
    print("正在测试 base model (no adapter) ...")
    try:
        avg_time = measure_inference_time(None, task, num_runs=10, warmup_runs=2)
        results["base_model (no adapter)"] = avg_time
        print(f"  平均耗时: {avg_time:.4f} 秒")
    except Exception as e:
        print(f"  测试失败: {e}")

    # 2. 列出所有适配器并测试
    adapters = [
        d for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith("prompt_generator_ep")
    ]
    adapters.sort()

    for adapter in adapters:
        adapter_path = os.path.join(checkpoints_dir, adapter)
        print(f"正在测试 {adapter} ...")
        try:
            avg_time = measure_inference_time(adapter_path, task, num_runs=10, warmup_runs=2)
            results[adapter] = avg_time
            print(f"  平均耗时: {avg_time:.4f} 秒")
        except Exception as e:
            print(f"  测试失败: {e}")

    # 输出汇总
    print("\n===== 推理速度对比 =====")
    for name, avg_time in results.items():
        print(f"{name}: {avg_time:.4f} 秒")
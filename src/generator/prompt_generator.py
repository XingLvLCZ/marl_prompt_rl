from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from typing import Any

class PromptGenerator:
    def __init__(
        self,
        model_name="/root/aicloud-data/llms/Qwen3-4B-Instruct-2507",
        adapter_path=None,
        lora_config=None,
        device_map: Any = "auto",
        torch_dtype: Any = "auto",
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize PromptGenerator with LoRA support for Qwen3-4B-Instruct.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设置 pad_token 为 eos_token（很多模型没有默认 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"torch_dtype": torch_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # 处理 LoRA 适配器
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            if lora_config is None:
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
            self.model = get_peft_model(self.model, lora_config)

        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            # 梯度检查点时禁用 KV-cache
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        self.model.print_trainable_parameters()
        self.model_name = model_name

    def generate_prompt_without_log_prob(self, prompt):
        """Inference-only generation without log probabilities."""
        messages = [{"role": "user", "content": prompt}]
        # Nanbeige 使用标准的 ChatML 格式，移除 Qwen 特有的 enable_thinking 参数
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=16384,
                pad_token_id=self.tokenizer.eos_token_id,  # 防止 pad_token 未定义
            )

        # 直接跳过输入部分，用 skip_special_tokens=True 获取纯文本回复
        prompt_len = model_inputs["input_ids"].size(1)
        generated_prompt = self.tokenizer.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=True
        ).strip()
        return generated_prompt

    def _compute_avg_log_prob(self, model, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Compute mean log-prob over generated tokens only."""
        if generated_ids.size(1) <= prompt_len:
            return torch.tensor(0.0, device=generated_ids.device)

        attention_mask = torch.ones_like(generated_ids, dtype=torch.long)

        outputs = model(
            input_ids=generated_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=False,
        )

        logits = outputs.logits  # [B, T, V]
        target_ids = generated_ids[:, 1:]  # [B, T]

        # 更节省内存的计算方式
        target_logits = logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        log_denom = logits.logsumexp(dim=-1)
        token_log_probs = target_logits - log_denom

        # 只保留生成部分的 log-prob
        gen_token_log_probs = token_log_probs[:, prompt_len - 1:]
        return gen_token_log_probs.mean()

    def compute_avg_log_prob(self, model, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Public helper for mean log-prob."""
        return self._compute_avg_log_prob(model, generated_ids, prompt_len)

    def generate_prompt(self, prompt, temperature=0.8, max_new_tokens=1024):
        """
        Generate a prompt using the LoRA-adapted model.
        Returns (generated_prompt, generated_ids, prompt_len, avg_log_prob).
        """
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt", padding=False).to(self.model.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,  # 安全设置
            )

        prompt_len = attention_mask.size(1)

        self.model.enable_input_require_grads()
        avg_log_prob = self._compute_avg_log_prob(self.model, generated_ids, prompt_len)

        # 使用更通用的解码方式：跳过输入部分和所有特殊 token
        generated_prompt = self.tokenizer.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=True
        ).strip()

        return generated_prompt, generated_ids, prompt_len, avg_log_prob


if __name__ == "__main__":
    generator = PromptGenerator(model_name="/root/aicloud-data/llms/Qwen3-4B-Instruct-2507")  # 如果已下载到本地，替换为本地路径

    task_description = f"""
Task target:
- task_id: arith_1
- collaborative_problem: Solve collaboratively. Compute (17 * 6) + 25. The final responder must output exactly: FINAL_ANSWER: <value>.

Your task:
Generate a protocol-generation prompt that will guide another LLM to produce a collaboration protocol for AutoGen multi-agent problem solving.
DO NOT solve the problem or give the final answer yourself in the prompt. Focus on designing a high-quality prompt.
OUTPUT ONLY the prompt content.
""".strip()

    generated_prompt, generated_ids, prompt_len, log_prob = generator.generate_prompt(
        task_description,
        temperature=0.7,
        max_new_tokens=2048,
    )

    # 获取并打印峰值显存
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory: {peak / 1024**2:.2f} MB")

    print("Prompt for Protocol:\n")
    print(generated_prompt)
    print(f"\nLog probability: {log_prob.item():.4f}")
    print(f"Requires grad: {log_prob.requires_grad}")
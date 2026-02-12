from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch


class PromptGenerator:
    def __init__(self, model_name="/root/aicloud-data/llms/Qwen3-1.7B", adapter_path=None, lora_config=None):
        """
        Initialize PromptGenerator with LoRA support.
        
        Args:
            model_name: Path to base model
            adapter_path: Path to pre-trained LoRA adapter (optional)
            lora_config: LoRA configuration (uses defaults if None)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
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
                    task_type=TaskType.CAUSAL_LM
                )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.print_trainable_parameters()
        self.model_name = model_name
    
    def generate_prompt_without_log_prob(self, prompt):
        """Inference-only generation without computing log probabilities."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
        
        output_ids = generated_ids[0].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        generated_prompt = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return generated_prompt
    
    def _compute_avg_log_prob(self, model, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Compute mean log-prob over generated tokens only."""
        if generated_ids.size(1) <= prompt_len:
            return torch.tensor(0.0, device=generated_ids.device)

        attention_mask = torch.ones_like(generated_ids, dtype=torch.long)

        outputs = model(
            input_ids=generated_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=generated_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Only keep log-probs for generated tokens (exclude prompt tokens)
        gen_token_log_probs = token_log_probs[:, prompt_len - 1 :]
        return gen_token_log_probs.mean()

    def compute_avg_log_prob(self, model, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Public helper to compute mean log-prob over generated tokens."""
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
            enable_thinking=True
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
            )

        prompt_len = attention_mask.size(1)

        self.model.enable_input_require_grads()
        avg_log_prob = self._compute_avg_log_prob(self.model, generated_ids, prompt_len)

        output_ids = generated_ids[0].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        generated_prompt = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True
        ).strip("\n")

        return generated_prompt, generated_ids, prompt_len, avg_log_prob


if __name__ == "__main__":
    generator = PromptGenerator(model_name="/root/aicloud-data/llms/Qwen3-1.7B")
    
    task_description = f"""
Game:
- game_type: guessing number
- range_size: 10 (from 0 to 9)
- num_agents: 2

Game description:
Multiple agents take turns to guess a secret target number within the specified range.
After each guess, agents can follow a protocol and send a message to others.
In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

Your task:
Generate a **protocol-generation prompt** that guides a LLM to generate the protocol.
Reply ONLY the content of the **prompt**.
""".strip()

    generated_prompt, generated_ids, prompt_len, log_prob = generator.generate_prompt(
        task_description,
        temperature=0.3,
        max_new_tokens=3000
    )
    
    print("Prompt for Protocol:\n")
    print(generated_prompt)
    print(f"\nLog probability: {log_prob.item():.4f}")
    print(f"Requires grad: {log_prob.requires_grad}")

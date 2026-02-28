from __future__ import annotations

import inspect
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from src.prompt_rl_gsm8k.config import AutoGenDatasetConfig
from src.prompt_rl_gsm8k.prompt_template import build_protocol_generation_prompt
from src.prompt_rl_gsm8k.task_dataset import load_tasks, split_tasks
from src.prompt_rl_gsm8k.trl_config import TRLGSM8KConfig
from src.prompt_rl_gsm8k.trl_reward_adapter import RewardAdapterRuntime, TRLGSM8KRewardAdapter


def _filter_kwargs_for_callable(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(target)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _build_grpo_args(cfg: TRLGSM8KConfig) -> GRPOConfig:
    raw_args = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "max_completion_length": cfg.max_completion_length,
        "num_generations": cfg.num_generations,
        "generation_batch_size": cfg.generation_batch_size,
        "temperature": cfg.temperature,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "report_to": cfg.report_to,
    }
    kwargs = _filter_kwargs_for_callable(GRPOConfig.__init__, raw_args)
    return GRPOConfig(**kwargs)


def _build_train_dataset(cfg: TRLGSM8KConfig) -> Dataset:
    dataset_cfg = AutoGenDatasetConfig(
        source=cfg.task_source,
        local_path=cfg.task_local_path,
        hf_path=cfg.task_hf_path,
        hf_name=cfg.task_hf_name,
        hf_split=cfg.task_hf_split,
        task_id_field=cfg.task_id_field,
        prompt_field=cfg.task_prompt_field,
        answer_field=cfg.task_answer_field,
        task_type=cfg.task_type,
        limit=cfg.task_limit,
        shuffle=cfg.task_shuffle,
        seed=cfg.seed,
        train_ratio=cfg.task_train_ratio,
    )
    all_tasks = load_tasks(dataset_cfg)
    if dataset_cfg.source == "default":
        train_tasks = all_tasks
    else:
        train_tasks, _ = split_tasks(all_tasks, train_ratio=dataset_cfg.train_ratio, seed=dataset_cfg.seed)

    if cfg.max_train_samples > 0:
        train_tasks = train_tasks[: cfg.max_train_samples]

    # generation_prompt = build_protocol_generation_prompt()  # 第一阶段应用先验知识
    generation_prompt = build_protocol_generation_prompt(prior='')  # 第二阶段不应用先验知识
    rows = []
    for task in train_tasks:
        rows.append(
            {
                "prompt": [{"role": "user", "content": generation_prompt}],
                "task_id": task.task_id,
                "task_prompt": task.prompt,
                "expected_answer": task.expected_answer,
            }
        )
    return Dataset.from_list(rows)

def train_with_trl(cfg: TRLGSM8KConfig | None = None) -> None:
    cfg = cfg or TRLGSM8KConfig()
    if cfg.force_single_thread:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = _build_train_dataset(cfg)

    runtime = RewardAdapterRuntime(
        prompt_history_dir=cfg.prompt_history_dir,
        protocol_history_dir=cfg.protocol_history_dir,
        eval_outputs_dir=cfg.eval_outputs_dir,
        protocol_model=cfg.protocol_model,
        env_model=cfg.env_model,
        env_max_rounds=cfg.env_max_rounds,
        env_temperature=cfg.env_temperature,
        strategist_name=cfg.strategist_name,
        calculator_name=cfg.calculator_name,
        verifier_name=cfg.verifier_name,
    )
    reward_adapter = TRLGSM8KRewardAdapter(runtime=runtime)

    def gsm8k_env_reward(prompts, completions, **kwargs):
        # print(f"\nPrompt (total {len(prompts[0])}): \n{prompts[0][0]['content']}")
        # print(f"\nCompletion (total {len(completions[0])}): \n{completions[0][0]['content']}")
        # return [0 for _ in completions]  # --- IGNORE ---
        return reward_adapter(prompts=prompts, completions=completions, **kwargs)

    grpo_args = _build_grpo_args(cfg)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    trainer = GRPOTrainer(
        model=cfg.base_model_path,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=gsm8k_env_reward,
        peft_config=lora_config,
    )
    
    if cfg.resume_from_checkpoint:
        print(f"[Resume Training] checkpoint: {cfg.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    else:
        trainer.train()

    final_dir = str(Path(cfg.output_dir) / "final")
    trainer.save_model(final_dir)

    print("[TRL Train Done]")
    print(f"- output_dir: {cfg.output_dir}")
    print(f"- final_checkpoint: {final_dir}")
    print(f"- config: {asdict(cfg)}")


if __name__ == "__main__":
    train_with_trl(TRLGSM8KConfig())

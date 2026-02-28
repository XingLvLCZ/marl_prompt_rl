from dataclasses import dataclass


@dataclass
class TRLGSM8KConfig:
    base_model_path: str = "/root/models/Qwen3-4B-Instruct-2507"
    output_dir: str = "src/prompt_rl_gsm8k/checkpoints_trl"
    resume_from_checkpoint: str | bool = "src/prompt_rl_gsm8k/checkpoints_trl/checkpoint-450"  # path/to/checkpoint or True (auto-resume from latest) or False (do not resume)

    epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    logging_steps: int = 10
    save_steps: int = 50

    num_generations: int = 2
    generation_batch_size: int = 4
    max_completion_length: int = 2048
    temperature: float = 0.8

    max_train_samples: int = 0
    seed: int = 42

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False

    task_source: str = "hf"  # default/local/hf
    task_local_path: str = ""
    task_hf_path: str = "/root/datasets/gsm8k"
    task_hf_name: str = "main"
    task_hf_split: str = "train"
    task_id_field: str = "task_id"
    task_prompt_field: str = "question"
    task_answer_field: str = "answer"
    task_type: str = "gsm8k"
    task_limit: int = 0
    task_shuffle: bool = True
    task_train_ratio: float = 0.8

    protocol_model: str = "Qwen/Qwen3-14B"
    env_model: str = "Qwen/Qwen3-14B"
    env_max_rounds: int = 9
    env_temperature: float = 0.2

    strategist_name: str = "strategist"
    calculator_name: str = "calculator"
    verifier_name: str = "verifier"

    prompt_history_dir: str = "src/prompt_rl_gsm8k/trl_prompt_history"
    protocol_history_dir: str = "src/prompt_rl_gsm8k/trl_protocol_history"
    eval_outputs_dir: str = "src/prompt_rl_gsm8k/trl_eval_outputs"

    report_to: str = "none"

    force_single_thread: bool = True
    dataloader_num_workers: int = 0

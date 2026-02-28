from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
from tqdm import tqdm

from src.generator.prompt_generator import PromptGenerator
from src.generator.protocol_generator import ProtocolGenerator
from src.prompt_rl_gsm8k.autogen_env import AutoGenEpisodeEnv
from src.prompt_rl_gsm8k.config import AutoGenDatasetConfig, AutoGenTask
from src.prompt_rl_gsm8k.prompt_template import build_protocol_generation_prompt
from src.prompt_rl_gsm8k.task_dataset import load_tasks
from src.provider.local import LocalProvider


EVAL_ADAPTER_PATH = "src/prompt_rl_gsm8k/checkpoints_trl/checkpoint-700"
EVAL_BASE_MODEL = "/root/models/Qwen3-4B-Instruct-2507"
EVAL_PROTOCOL_MODEL = "Qwen/Qwen3-14B"
EVAL_ENV_MODEL = "Qwen/Qwen3-14B"
EVAL_LOCAL_BASE_URL = "http://localhost:8000/v1"
EVAL_LOCAL_API_KEY = "EMPTY"
EVAL_MAX_ROUNDS = 9
EVAL_ENV_TEMPERATURE = 0.2

EVAL_DATASET_HF_PATH = "/root/datasets/gsm8k"
EVAL_DATASET_HF_NAME = "main"
EVAL_DATASET_SPLIT = "test"
EVAL_TASK_LIMIT = 0
EVAL_TASK_SHUFFLE = False
EVAL_SEED = 42

EVAL_OUTPUT_ROOT = "src/prompt_rl_gsm8k/final_test_outputs"
EVAL_RUN_NAME = ""
EVAL_LOG_EVERY = 20

EVAL_STRATEGIST_NAME = "strategist"
EVAL_CALCULATOR_NAME = "calculator"
EVAL_VERIFIER_NAME = "verifier"


@dataclass
class EvalConfig:
    adapter_path: str = "src/prompt_rl_gsm8k/checkpoints_trl/checkpoint-700"
    base_model: str = "/root/models/Qwen3-4B-Instruct-2507"
    protocol_model: str = "Qwen/Qwen3-14B"
    env_model: str = "Qwen/Qwen3-14B"
    local_base_url: str = "http://localhost:8000/v1"
    local_api_key: str = "EMPTY"
    max_rounds: int = 9
    env_temperature: float = 0.2
    dataset_hf_path: str = "/root/datasets/gsm8k"
    dataset_hf_name: str = "main"
    dataset_split: str = "test"
    task_limit: int = 0
    task_shuffle: bool = False
    seed: int = 42
    output_root: str = "src/prompt_rl_gsm8k/final_test_outputs"
    run_name: str = ""
    log_every: int = 20
    strategist_name: str = "strategist"
    calculator_name: str = "calculator"
    verifier_name: str = "verifier"


@dataclass
class EvalRow:
    index: int
    task_id: str
    success: bool
    rounds: int
    final_answer: str
    expected_answer: str
    latency_sec: float
    error: str | None


def _build_eval_config() -> EvalConfig:
    return EvalConfig(
        adapter_path=EVAL_ADAPTER_PATH,
        base_model=EVAL_BASE_MODEL,
        protocol_model=EVAL_PROTOCOL_MODEL,
        env_model=EVAL_ENV_MODEL,
        local_base_url=EVAL_LOCAL_BASE_URL,
        local_api_key=EVAL_LOCAL_API_KEY,
        max_rounds=EVAL_MAX_ROUNDS,
        env_temperature=EVAL_ENV_TEMPERATURE,
        dataset_hf_path=EVAL_DATASET_HF_PATH,
        dataset_hf_name=EVAL_DATASET_HF_NAME,
        dataset_split=EVAL_DATASET_SPLIT,
        task_limit=EVAL_TASK_LIMIT,
        task_shuffle=EVAL_TASK_SHUFFLE,
        seed=EVAL_SEED,
        output_root=EVAL_OUTPUT_ROOT,
        run_name=EVAL_RUN_NAME,
        log_every=max(1, EVAL_LOG_EVERY),
        strategist_name=EVAL_STRATEGIST_NAME,
        calculator_name=EVAL_CALCULATOR_NAME,
        verifier_name=EVAL_VERIFIER_NAME,
    )


def _load_eval_tasks(cfg: EvalConfig) -> list[AutoGenTask]:
    dataset_cfg = AutoGenDatasetConfig(
        source="hf",
        hf_path=cfg.dataset_hf_path,
        hf_name=cfg.dataset_hf_name,
        hf_split=cfg.dataset_split,
        task_id_field="task_id",
        prompt_field="question",
        answer_field="answer",
        task_type="gsm8k",
        limit=cfg.task_limit,
        shuffle=cfg.task_shuffle,
        seed=cfg.seed,
    )
    return load_tasks(dataset_cfg)


def _prepare_output_dirs(cfg: EvalConfig) -> tuple[Path, Path, Path]:
    root = Path(cfg.output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name.strip() or f"final_eval_{stamp}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifacts_dir, episodes_dir


def _write_episode_file(path: Path, row: EvalRow, transcript: list[str], task_prompt: str) -> None:
    text = [
        f"# Episode {row.index}",
        "",
        f"- task_id: {row.task_id}",
        f"- success: {row.success}",
        f"- rounds: {row.rounds}",
        f"- expected_answer: {row.expected_answer}",
        f"- final_answer: {row.final_answer}",
        f"- latency_sec: {row.latency_sec:.3f}",
        f"- error: {row.error}",
        "",
        "## Task",
        "",
        task_prompt,
        "",
        "## Transcript",
        "",
    ]
    text.extend([f"- {line}" for line in transcript] if transcript else ["- <empty>"])
    path.write_text("\n".join(text), encoding="utf-8")


def _build_summary(cfg: EvalConfig, rows: list[EvalRow], total_sec: float, run_dir: Path) -> dict:
    total = len(rows)
    success_count = sum(1 for row in rows if row.success)
    error_count = sum(1 for row in rows if row.error)

    return {
        "adapter_path": cfg.adapter_path,
        "base_model": cfg.base_model,
        "protocol_model": cfg.protocol_model,
        "env_model": cfg.env_model,
        "dataset": {
            "hf_path": cfg.dataset_hf_path,
            "hf_name": cfg.dataset_hf_name,
            "split": cfg.dataset_split,
        },
        "num_tasks": total,
        "num_success": success_count,
        "num_failed": total - success_count,
        "num_errors": error_count,
        "accuracy": (success_count / total) if total else 0.0,
        "avg_rounds": mean([row.rounds for row in rows]) if rows else 0.0,
        "avg_latency_sec": mean([row.latency_sec for row in rows]) if rows else 0.0,
        "total_time_sec": total_sec,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
    }


def _write_reports(cfg: EvalConfig, rows: list[EvalRow], summary: dict, run_dir: Path) -> None:
    json_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    report_path = run_dir / "report.md"

    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(asdict(rows[0]).keys()) if rows else list(EvalRow.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        "# GSM8K Final Evaluation Report",
        "",
        "## Summary",
        "",
        f"- adapter_path: {summary['adapter_path']}",
        f"- dataset: {summary['dataset']['hf_path']}::{summary['dataset']['hf_name']}::{summary['dataset']['split']}",
        f"- num_tasks: {summary['num_tasks']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- avg_rounds: {summary['avg_rounds']:.3f}",
        f"- avg_latency_sec: {summary['avg_latency_sec']:.3f}",
        f"- num_errors: {summary['num_errors']}",
        f"- total_time_sec: {summary['total_time_sec']:.2f}",
        "",
        "## Files",
        "",
        f"- detailed_json: {json_path}",
        f"- detailed_csv: {csv_path}",
        f"- per_episode_dir: {run_dir / 'episodes'}",
    ]

    worst = sorted(rows, key=lambda x: (x.success, x.rounds, -x.latency_sec), reverse=False)[:10]
    lines.extend(["", "## Hard Cases (Top 10)", ""])
    for row in worst:
        lines.append(
            f"- idx={row.index} task_id={row.task_id} success={row.success} rounds={row.rounds} "
            f"expected={row.expected_answer} pred={row.final_answer} error={row.error}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_final_eval(cfg: EvalConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tasks = _load_eval_tasks(cfg)
    run_dir, artifacts_dir, episodes_dir = _prepare_output_dirs(cfg)

    print(f"[Final Eval] tasks={len(tasks)} split={cfg.dataset_split} adapter={cfg.adapter_path}", flush=True)
    print(f"[Final Eval] output_dir={run_dir}", flush=True)

    generator = PromptGenerator(model_name=cfg.base_model, adapter_path=cfg.adapter_path)
    provider = LocalProvider(api_key=cfg.local_api_key, base_url=cfg.local_base_url, model=cfg.protocol_model)
    protocol_generator = ProtocolGenerator(provider=provider)
    env = AutoGenEpisodeEnv(
        model=cfg.env_model,
        api_key=cfg.local_api_key,
        base_url=cfg.local_base_url,
        max_rounds=cfg.max_rounds,
        temperature=cfg.env_temperature,
        strategist_name=cfg.strategist_name,
        calculator_name=cfg.calculator_name,
        verifier_name=cfg.verifier_name,
    )

    generation_prompt = build_protocol_generation_prompt(prior="")
    generated_prompt = generator.generate_prompt_without_log_prob(generation_prompt)
    prompt_path = artifacts_dir / "generated_prompt.md"
    prompt_path.write_text(generated_prompt, encoding="utf-8")

    protocol_path = protocol_generator.generate_protocol(
        prompt=generated_prompt,
        protocol_name="final_eval_protocol",
        save_dir=str(artifacts_dir),
    )
    protocol_text = protocol_path.read_text(encoding="utf-8")

    rows: list[EvalRow] = []
    eval_start = time.time()
    total = len(tasks)

    for idx, task in enumerate(tqdm(tasks, desc="final_eval", unit="task"), start=1):
        step_start = time.time()
        result = env.run_episode(protocol_text=protocol_text, task=task)
        latency = time.time() - step_start

        row = EvalRow(
            index=idx,
            task_id=task.task_id,
            success=bool(result.success),
            rounds=int(result.rounds),
            final_answer=str(result.final_answer),
            expected_answer=str(task.expected_answer),
            latency_sec=float(latency),
            error=result.error,
        )
        rows.append(row)

        episode_path = episodes_dir / f"episode_{idx:05d}.md"
        _write_episode_file(
            path=episode_path,
            row=row,
            transcript=list(result.transcript),
            task_prompt=task.prompt,
        )

        if idx % cfg.log_every == 0 or idx == total:
            acc = sum(1 for item in rows if item.success) / len(rows)
            avg_rounds = mean([item.rounds for item in rows]) if rows else 0.0
            elapsed = time.time() - eval_start
            eta = (elapsed / len(rows)) * max(0, total - idx)
            tqdm.write(
                f"  - [{idx}/{total}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} "
                f"step_time={latency:.2f}s eta={eta:.1f}s",
            )

    total_sec = time.time() - eval_start
    summary = _build_summary(cfg=cfg, rows=rows, total_sec=total_sec, run_dir=run_dir)
    _write_reports(cfg=cfg, rows=rows, summary=summary, run_dir=run_dir)

    print("[Final Eval Done]", flush=True)
    print(f"- accuracy: {summary['accuracy']:.4f}", flush=True)
    print(f"- avg_rounds: {summary['avg_rounds']:.4f}", flush=True)
    print(f"- report: {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    run_final_eval(_build_eval_config())

from __future__ import annotations

import csv
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generator.prompt_generator import PromptGenerator
from src.generator.protocol_generator import ProtocolGenerator
from src.prompt_rl_gsm8k.autogen_env import AutoGenEpisodeEnv, AutoGenEpisodeResult
from src.prompt_rl_gsm8k.config import AutoGenDatasetConfig, AutoGenTask
from src.prompt_rl_gsm8k.prompt_template import build_protocol_generation_prompt
from src.prompt_rl_gsm8k.task_dataset import load_tasks
from src.provider.local import LocalProvider


COMPARE_BASE_MODEL_PATH = "/root/models/Qwen3-4B-Instruct-2507"
COMPARE_RL_ADAPTER_PATH = "src/prompt_rl_gsm8k/checkpoints_trl/checkpoint-700"

COMPARE_PROTOCOL_MODEL = "Qwen/Qwen3-14B"
COMPARE_ENV_MODEL = "Qwen/Qwen3-14B"
COMPARE_SINGLE_SOLVER_MODEL = "Qwen/Qwen3-14B"
COMPARE_LOCAL_BASE_URL = "http://localhost:8000/v1"
COMPARE_LOCAL_API_KEY = "EMPTY"

COMPARE_MAX_ROUNDS = 9
COMPARE_ENV_TEMPERATURE = 0.2

COMPARE_DATASET_HF_PATH = "/root/datasets/gsm8k"
COMPARE_DATASET_HF_NAME = "main"
COMPARE_DATASET_SPLIT = "test"
COMPARE_TASK_LIMIT = 100
COMPARE_TASK_SHUFFLE = False
COMPARE_SEED = 42

COMPARE_OUTPUT_ROOT = "src/prompt_rl_gsm8k/compare_outputs"
COMPARE_RUN_NAME = ""
COMPARE_LOG_EVERY = 20

COMPARE_STRATEGIST_NAME = "strategist"
COMPARE_CALCULATOR_NAME = "calculator"
COMPARE_VERIFIER_NAME = "verifier"


@dataclass
class CompareConfig:
    base_model_path: str
    rl_adapter_path: str
    protocol_model: str
    env_model: str
    single_solver_model: str
    local_base_url: str
    local_api_key: str
    max_rounds: int
    env_temperature: float
    dataset_hf_path: str
    dataset_hf_name: str
    dataset_split: str
    task_limit: int
    task_shuffle: bool
    seed: int
    output_root: str
    run_name: str
    log_every: int
    strategist_name: str
    calculator_name: str
    verifier_name: str


@dataclass
class TaskEvalRow:
    task_id: str
    success: bool
    rounds: int
    final_answer: str
    expected_answer: str
    latency_sec: float
    error: str | None


def _build_config() -> CompareConfig:
    return CompareConfig(
        base_model_path=COMPARE_BASE_MODEL_PATH,
        rl_adapter_path=COMPARE_RL_ADAPTER_PATH,
        protocol_model=COMPARE_PROTOCOL_MODEL,
        env_model=COMPARE_ENV_MODEL,
        single_solver_model=COMPARE_SINGLE_SOLVER_MODEL,
        local_base_url=COMPARE_LOCAL_BASE_URL,
        local_api_key=COMPARE_LOCAL_API_KEY,
        max_rounds=COMPARE_MAX_ROUNDS,
        env_temperature=COMPARE_ENV_TEMPERATURE,
        dataset_hf_path=COMPARE_DATASET_HF_PATH,
        dataset_hf_name=COMPARE_DATASET_HF_NAME,
        dataset_split=COMPARE_DATASET_SPLIT,
        task_limit=COMPARE_TASK_LIMIT,
        task_shuffle=COMPARE_TASK_SHUFFLE,
        seed=COMPARE_SEED,
        output_root=COMPARE_OUTPUT_ROOT,
        run_name=COMPARE_RUN_NAME,
        log_every=max(1, COMPARE_LOG_EVERY),
        strategist_name=COMPARE_STRATEGIST_NAME,
        calculator_name=COMPARE_CALCULATOR_NAME,
        verifier_name=COMPARE_VERIFIER_NAME,
    )


def _load_tasks(cfg: CompareConfig) -> list[AutoGenTask]:
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


def _prepare_run_dirs(cfg: CompareConfig) -> dict[str, Path]:
    root = Path(cfg.output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name.strip() or f"compare_eval_{stamp}"
    run_dir = root / run_name
    (run_dir / "single_model").mkdir(parents=True, exist_ok=True)
    (run_dir / "multi_agent_base").mkdir(parents=True, exist_ok=True)
    (run_dir / "multi_agent_rl").mkdir(parents=True, exist_ok=True)
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "single_model": run_dir / "single_model",
        "multi_agent_base": run_dir / "multi_agent_base",
        "multi_agent_rl": run_dir / "multi_agent_rl",
        "charts": run_dir / "charts",
    }


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_prompt_base_model(base_model_path: str) -> str:
    prompt = build_protocol_generation_prompt(prior="")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="auto")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    out = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    del model
    del tokenizer
    _cleanup_cuda()
    return out


def _generate_prompt_rl_model(base_model_path: str, adapter_path: str) -> str:
    generator = PromptGenerator(model_name=base_model_path, adapter_path=adapter_path)
    prompt = build_protocol_generation_prompt(prior="")
    out = generator.generate_prompt_without_log_prob(prompt)

    del generator
    _cleanup_cuda()
    return out


def _generate_protocol(
    protocol_model: str,
    generated_prompt: str,
    out_dir: Path,
    local_base_url: str,
    local_api_key: str,
) -> str:
    provider = LocalProvider(api_key=local_api_key, base_url=local_base_url, model=protocol_model)
    protocol_generator = ProtocolGenerator(provider=provider)
    protocol_path = protocol_generator.generate_protocol(
        prompt=generated_prompt,
        protocol_name="generated_protocol",
        save_dir=str(out_dir),
    )
    return protocol_path.read_text(encoding="utf-8")


def _single_model_answer(provider: LocalProvider, question: str) -> str:
    user_prompt = (
        "You are solving a GSM8K math word problem. "
        "Solve it carefully and output your final answer in one line with this exact format: "
        "FINAL_ANSWER: <number>\n\n"
        f"Question:\n{question}"
    )
    return provider.call(user_prompt)


def _compute_row_from_result(
    result: AutoGenEpisodeResult,
    task: AutoGenTask,
    latency_sec: float,
) -> TaskEvalRow:
    return TaskEvalRow(
        task_id=task.task_id,
        success=bool(result.success),
        rounds=int(result.rounds),
        final_answer=str(result.final_answer),
        expected_answer=str(task.expected_answer),
        latency_sec=float(latency_sec),
        error=result.error,
    )


def _evaluate_single_model(cfg: CompareConfig, tasks: list[AutoGenTask], out_dir: Path) -> list[TaskEvalRow]:
    provider = LocalProvider(api_key=cfg.local_api_key, base_url=cfg.local_base_url, model=cfg.single_solver_model)

    rows: list[TaskEvalRow] = []
    total = len(tasks)
    start = time.time()
    for idx, task in enumerate(tqdm(tasks, desc="single_model", unit="task"), start=1):
        t0 = time.time()
        error = None
        response_text = ""
        try:
            response_text = _single_model_answer(provider=provider, question=task.prompt)
            final_answer = AutoGenEpisodeEnv._extract_final_answer([response_text])
            success = AutoGenEpisodeEnv._is_correct(final_answer, task.expected_answer)
            result = AutoGenEpisodeResult(
                task_id=task.task_id,
                success=success,
                score=1.0 if success else 0.0,
                rounds=1,
                final_answer=final_answer,
                transcript=[f"[single_model] {response_text}"],
                early_stop=True,
                error=None,
            )
        except Exception as exc:
            error = str(exc)
            result = AutoGenEpisodeResult(
                task_id=task.task_id,
                success=False,
                score=-1.0,
                rounds=1,
                final_answer="",
                transcript=[f"[single_model_error] {error}"],
                early_stop=False,
                error=error,
            )

        row = _compute_row_from_result(
            result=result,
            task=task,
            latency_sec=time.time() - t0,
        )
        rows.append(row)

        (out_dir / f"episode_{idx:05d}.md").write_text(
            "\n".join(
                [
                    f"# Single Model Episode {idx}",
                    "",
                    f"- task_id: {task.task_id}",
                    f"- success: {row.success}",
                    f"- expected: {task.expected_answer}",
                    f"- pred: {row.final_answer}",
                    f"- rounds: {row.rounds}",
                    f"- error: {error}",
                    "",
                    "## Response",
                    "",
                    response_text if response_text else "<empty>",
                ]
            ),
            encoding="utf-8",
        )

        if idx % cfg.log_every == 0 or idx == total:
            cur_acc = sum(1 for item in rows if item.success) / len(rows)
            cur_rounds = mean([item.rounds for item in rows]) if rows else 0.0
            elapsed = time.time() - start
            eta = (elapsed / len(rows)) * max(0, total - idx)
            tqdm.write(
                f"  - [single] {idx}/{total} acc={cur_acc:.4f} avg_rounds={cur_rounds:.3f} eta={eta:.1f}s",
            )
    return rows


def _evaluate_multi_agent(
    cfg: CompareConfig,
    tasks: list[AutoGenTask],
    out_dir: Path,
    generated_prompt: str,
    protocol_text: str,
    tag: str,
) -> list[TaskEvalRow]:
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
    rows: list[TaskEvalRow] = []
    total = len(tasks)
    start = time.time()
    for idx, task in enumerate(tqdm(tasks, desc=tag, unit="task"), start=1):
        t0 = time.time()
        result = env.run_episode(protocol_text=protocol_text, task=task)
        row = _compute_row_from_result(
            result=result,
            task=task,
            latency_sec=time.time() - t0,
        )
        rows.append(row)

        (out_dir / f"episode_{idx:05d}.md").write_text(
            "\n".join(
                [
                    f"# {tag} Episode {idx}",
                    "",
                    f"- task_id: {task.task_id}",
                    f"- success: {row.success}",
                    f"- rounds: {row.rounds}",
                    f"- expected: {task.expected_answer}",
                    f"- pred: {row.final_answer}",
                    f"- error: {row.error}",
                    "",
                    "## Transcript",
                    "",
                ]
                + ([f"- {line}" for line in result.transcript] if result.transcript else ["- <empty>"])
            ),
            encoding="utf-8",
        )

        if idx % cfg.log_every == 0 or idx == total:
            cur_acc = sum(1 for item in rows if item.success) / len(rows)
            cur_rounds = mean([item.rounds for item in rows]) if rows else 0.0
            elapsed = time.time() - start
            eta = (elapsed / len(rows)) * max(0, total - idx)
            tqdm.write(
                f"  - [{tag}] {idx}/{total} acc={cur_acc:.4f} avg_rounds={cur_rounds:.3f} eta={eta:.1f}s",
            )
    return rows


def _summary(rows: list[TaskEvalRow]) -> dict:
    total = len(rows)
    success = sum(1 for row in rows if row.success)
    errors = sum(1 for row in rows if row.error)
    return {
        "num_tasks": total,
        "num_success": success,
        "num_failed": total - success,
        "num_errors": errors,
        "accuracy": (success / total) if total else 0.0,
        "avg_rounds": mean([row.rounds for row in rows]) if rows else 0.0,
        "avg_rounds_success_only": mean([row.rounds for row in rows if row.success]) if success else 0.0,
        "avg_latency_sec": mean([row.latency_sec for row in rows]) if rows else 0.0,
    }


def _write_method_outputs(out_dir: Path, rows: list[TaskEvalRow], summary: dict, name: str) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(TaskEvalRow.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        f"# {name} Summary",
        "",
        f"- num_tasks: {summary['num_tasks']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- avg_rounds: {summary['avg_rounds']:.4f}",
        f"- avg_rounds_success_only: {summary['avg_rounds_success_only']:.4f}",
        f"- avg_latency_sec: {summary['avg_latency_sec']:.4f}",
        f"- num_errors: {summary['num_errors']}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_task_comparison(run_dir: Path, tasks: list[AutoGenTask], method_rows: dict[str, list[TaskEvalRow]]) -> None:
    name_map = {
        "single_model": method_rows["single_model"],
        "multi_agent_base": method_rows["multi_agent_base"],
        "multi_agent_rl": method_rows["multi_agent_rl"],
    }

    with (run_dir / "task_level_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task_id",
            "expected_answer",
            "single_success",
            "single_pred",
            "ma_base_success",
            "ma_base_rounds",
            "ma_base_pred",
            "ma_rl_success",
            "ma_rl_rounds",
            "ma_rl_pred",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, task in enumerate(tasks):
            s = name_map["single_model"][idx]
            b = name_map["multi_agent_base"][idx]
            r = name_map["multi_agent_rl"][idx]
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "expected_answer": task.expected_answer,
                    "single_success": s.success,
                    "single_pred": s.final_answer,
                    "ma_base_success": b.success,
                    "ma_base_rounds": b.rounds,
                    "ma_base_pred": b.final_answer,
                    "ma_rl_success": r.success,
                    "ma_rl_rounds": r.rounds,
                    "ma_rl_pred": r.final_answer,
                }
            )


def _plot_bar(values: dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    labels = list(values.keys())
    ys = [values[k] for k in labels]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, ys)
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{y:.4f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_charts(charts_dir: Path, summaries: dict[str, dict]) -> list[Path]:
    charts: list[Path] = []

    acc = {k: v["accuracy"] for k, v in summaries.items()}
    p1 = charts_dir / "accuracy_comparison.png"
    _plot_bar(acc, "Accuracy Comparison", "Accuracy", p1)
    charts.append(p1)

    rounds = {k: v["avg_rounds"] for k, v in summaries.items()}
    p2 = charts_dir / "avg_rounds_comparison.png"
    _plot_bar(rounds, "Average Rounds Comparison", "Avg Rounds", p2)
    charts.append(p2)

    latency = {k: v["avg_latency_sec"] for k, v in summaries.items()}
    p3 = charts_dir / "avg_latency_comparison.png"
    _plot_bar(latency, "Average Latency Comparison", "Seconds", p3)
    charts.append(p3)

    return charts


def _write_final_report(
    run_dir: Path,
    cfg: CompareConfig,
    summaries: dict[str, dict],
    charts: list[Path],
    total_time_sec: float,
) -> None:
    rows = [
        ("single_model", summaries["single_model"]),
        ("multi_agent_base", summaries["multi_agent_base"]),
        ("multi_agent_rl", summaries["multi_agent_rl"]),
    ]

    lines = [
        "# GSM8K Three-Setting Comparison Report",
        "",
        "## Experimental Setup",
        "",
        f"- base_model_path: {cfg.base_model_path}",
        f"- rl_adapter_path: {cfg.rl_adapter_path}",
        f"- protocol_model: {cfg.protocol_model}",
        f"- env_model: {cfg.env_model}",
        f"- single_solver_model: {cfg.single_solver_model}",
        f"- dataset: {cfg.dataset_hf_path}::{cfg.dataset_hf_name}::{cfg.dataset_split}",
        f"- max_rounds: {cfg.max_rounds}",
        f"- total_time_sec: {total_time_sec:.2f}",
        "",
        "## Metrics",
        "",
        "| setting | accuracy | avg_rounds | avg_rounds_success_only | avg_latency_sec | num_errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['avg_rounds']:.4f} | {m['avg_rounds_success_only']:.4f} | "
            f"{m['avg_latency_sec']:.4f} | {m['num_errors']} |"
        )

    lines.extend(
        [
            "",
            "## Key Deltas",
            "",
            (
                f"- multi_agent_vs_single_accuracy: "
                f"{summaries['multi_agent_rl']['accuracy'] - summaries['single_model']['accuracy']:+.4f} "
                "(using rl multi-agent)"
            ),
            (
                f"- rl_vs_base_multi_agent_accuracy: "
                f"{summaries['multi_agent_rl']['accuracy'] - summaries['multi_agent_base']['accuracy']:+.4f}"
            ),
            (
                f"- rl_vs_base_multi_agent_rounds: "
                f"{summaries['multi_agent_rl']['avg_rounds'] - summaries['multi_agent_base']['avg_rounds']:+.4f}"
            ),
            "",
            "## Charts",
            "",
        ]
    )

    for chart in charts:
        rel = chart.relative_to(run_dir)
        lines.append(f"- {rel}")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- single_model/summary.md",
            "- multi_agent_base/summary.md",
            "- multi_agent_rl/summary.md",
            "- task_level_comparison.csv",
            "- charts/*.png",
        ]
    )

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_comparison_eval() -> None:
    cfg = _build_config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tasks = _load_tasks(cfg)
    dirs = _prepare_run_dirs(cfg)
    run_dir = dirs["run_dir"]

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Compare Eval] tasks={len(tasks)} split={cfg.dataset_split}", flush=True)
    print(f"[Compare Eval] run_dir={run_dir}", flush=True)

    total_start = time.time()

    print("\n[1/3] Single-model direct solving", flush=True)
    single_rows = _evaluate_single_model(cfg=cfg, tasks=tasks, out_dir=dirs["single_model"])
    single_summary = _summary(single_rows)
    _write_method_outputs(dirs["single_model"], single_rows, single_summary, "single_model")

    print("\n[2/3] Multi-agent with base prompt generator (no RL adapter)", flush=True)
    base_generated_prompt = _generate_prompt_base_model(cfg.base_model_path)
    (dirs["multi_agent_base"] / "generated_prompt.md").write_text(base_generated_prompt, encoding="utf-8")
    base_protocol = _generate_protocol(
        cfg.protocol_model,
        base_generated_prompt,
        dirs["multi_agent_base"],
        cfg.local_base_url,
        cfg.local_api_key,
    )
    (dirs["multi_agent_base"] / "generated_protocol.md").write_text(base_protocol, encoding="utf-8")
    base_rows = _evaluate_multi_agent(
        cfg=cfg,
        tasks=tasks,
        out_dir=dirs["multi_agent_base"],
        generated_prompt=base_generated_prompt,
        protocol_text=base_protocol,
        tag="multi_agent_base",
    )
    base_summary = _summary(base_rows)
    _write_method_outputs(dirs["multi_agent_base"], base_rows, base_summary, "multi_agent_base")

    print("\n[3/3] Multi-agent with RL-optimized prompt generator", flush=True)
    rl_generated_prompt = _generate_prompt_rl_model(cfg.base_model_path, cfg.rl_adapter_path)
    (dirs["multi_agent_rl"] / "generated_prompt.md").write_text(rl_generated_prompt, encoding="utf-8")
    rl_protocol = _generate_protocol(
        cfg.protocol_model,
        rl_generated_prompt,
        dirs["multi_agent_rl"],
        cfg.local_base_url,
        cfg.local_api_key,
    )
    (dirs["multi_agent_rl"] / "generated_protocol.md").write_text(rl_protocol, encoding="utf-8")
    rl_rows = _evaluate_multi_agent(
        cfg=cfg,
        tasks=tasks,
        out_dir=dirs["multi_agent_rl"],
        generated_prompt=rl_generated_prompt,
        protocol_text=rl_protocol,
        tag="multi_agent_rl",
    )
    rl_summary = _summary(rl_rows)
    _write_method_outputs(dirs["multi_agent_rl"], rl_rows, rl_summary, "multi_agent_rl")

    summaries = {
        "single_model": single_summary,
        "multi_agent_base": base_summary,
        "multi_agent_rl": rl_summary,
    }

    _write_task_comparison(
        run_dir=run_dir,
        tasks=tasks,
        method_rows={
            "single_model": single_rows,
            "multi_agent_base": base_rows,
            "multi_agent_rl": rl_rows,
        },
    )
    charts = _write_charts(dirs["charts"], summaries)

    total_time_sec = time.time() - total_start
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_final_report(run_dir, cfg, summaries, charts, total_time_sec)

    print("\n[Compare Eval Done]", flush=True)
    print(f"- single_model accuracy: {single_summary['accuracy']:.4f}", flush=True)
    print(f"- multi_agent_base accuracy: {base_summary['accuracy']:.4f}", flush=True)
    print(f"- multi_agent_rl accuracy: {rl_summary['accuracy']:.4f}", flush=True)
    print(f"- multi_agent_base avg_rounds: {base_summary['avg_rounds']:.4f}", flush=True)
    print(f"- multi_agent_rl avg_rounds: {rl_summary['avg_rounds']:.4f}", flush=True)
    print(f"- report: {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    run_comparison_eval()

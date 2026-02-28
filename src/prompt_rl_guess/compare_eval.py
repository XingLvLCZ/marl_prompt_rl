from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generator.protocol_generator import ProtocolGenerator
from src.generator.qwen3_prompt_generator import PromptGenerator
from src.prompt_rl_guess.guess_agent import GuessNumAgent
from src.prompt_rl_guess.pz_guess_env import GuessGamePettingZooEnv
from src.prompt_rl_guess.pz_guess_runner import GuessGameRunner
from src.provider.local import LocalProvider
from src.provider.qwen import QwenProvider
from src.provider.config import API_KEY, API_URL


COMPARE_BASE_MODEL_PATH = "/root/models/Qwen3-1.7B"
COMPARE_RL_ADAPTER_ROOT = "src/prompt_rl_guess/checkpoints"
# 留空则自动发现该目录下所有 adapter（包含 adapter_model.safetensors 的目录）
COMPARE_RL_ADAPTER_INCLUDE: List[str] = []

# COMPARE_LOCAL_BASE_URL = "http://localhost:8000/v1"
COMPARE_LOCAL_BASE_URL = API_URL
COMPARE_LOCAL_API_KEY = API_KEY
COMPARE_PROVIDER_MODEL = "Qwen/Qwen3-14B"

COMPARE_OUTPUT_ROOT = "src/prompt_rl_guess/eval_outputs"
COMPARE_RUN_NAME = ""

COMPARE_NUM_CHOICES = 10
COMPARE_BASE_NUM_AGENTS = 2
COMPARE_MAX_STEPS = 20
COMPARE_ROUNDS = 3

COMPARE_LOG_EVERY_TARGET = 1

HARD_CONSTRAINT = """
==================== HARD REQUIREMENTS ====================

1. The protocol MUST be written in Markdown.
2. The protocol MUST explicitly require agents to output ONLY valid JSON.
3. Each agent message MUST contain a field named "next_guess" with an integer value.

==================== OUTPUT CONSTRAINT ====================

- Output ONLY the content of the protocol.
- Do NOT include reasoning, analysis, or <think> blocks.
- Do NOT include any text before or after the protocol.
""".strip()


@dataclass
class MethodConfig:
    name: str
    use_lora: bool
    adapter_path: Optional[str] = None


@dataclass
class EvalScenario:
    name: str
    num_choices: int
    num_agents: int
    max_steps: int
    rounds: int


@dataclass
class TargetResult:
    scenario_name: str
    num_agents: int
    round_idx: int
    target: int
    success: bool
    steps: int
    avg_agent_reward: float
    total_agent_reward: float
    latency_sec: float
    agent_rewards: Dict[str, float]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_scenarios() -> List[EvalScenario]:
    return [
        EvalScenario(
            name="eval_checkpoint_like",
            num_choices=COMPARE_NUM_CHOICES,
            num_agents=COMPARE_BASE_NUM_AGENTS,
            max_steps=COMPARE_MAX_STEPS,
            rounds=COMPARE_ROUNDS,
        )
    ]


def _discover_adapter_dirs() -> List[Path]:
    root = Path(COMPARE_RL_ADAPTER_ROOT)
    if not root.exists():
        return []
    if COMPARE_RL_ADAPTER_INCLUDE:
        candidates = [root / name for name in COMPARE_RL_ADAPTER_INCLUDE]
    else:
        candidates = sorted([p for p in root.iterdir() if p.is_dir()])

    adapters: List[Path] = []
    for path in candidates:
        if (path / "adapter_model.safetensors").exists() and (path / "adapter_config.json").exists():
            adapters.append(path)
    return adapters


def _build_methods() -> List[MethodConfig]:
    methods: List[MethodConfig] = [MethodConfig(name="base_model", use_lora=False, adapter_path=None)]
    for adapter_dir in _discover_adapter_dirs():
        methods.append(
            MethodConfig(
                name=adapter_dir.name,
                use_lora=True,
                adapter_path=str(adapter_dir),
            )
        )
    return methods


def _build_task_description(scenario: EvalScenario) -> str:
    return f"""
Game:
- game_type: guessing number game
- range_size: {scenario.num_choices} (from 0 to {scenario.num_choices - 1})
- num_agents: {scenario.num_agents}

Game description:
Multiple agents take turns to guess a secret target number within the specified range.
After each guess, agents can follow a protocol and send a message to others.
In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

Your task:
Generate a protocol-generation prompt that guides a LLM to generate the protocol.
Reply ONLY the content of the prompt.
""".strip()


def _prepare_run_dirs(methods: List[MethodConfig]) -> dict[str, Path]:
    run_name = COMPARE_RUN_NAME.strip() or f"compare_base_multi_adapter_{_now_stamp()}"
    root = Path(COMPARE_OUTPUT_ROOT) / run_name
    methods_root = root / "methods"
    methods_root.mkdir(parents=True, exist_ok=True)
    (root / "charts").mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {
        "run_dir": root,
        "methods_root": methods_root,
        "charts": root / "charts",
    }
    for method in methods:
        path = methods_root / method.name
        path.mkdir(parents=True, exist_ok=True)
        result[f"method:{method.name}"] = path
    return result


def _build_provider():
    return QwenProvider(
        api_key=COMPARE_LOCAL_API_KEY,
        base_url=COMPARE_LOCAL_BASE_URL,
        model=COMPARE_PROVIDER_MODEL,
    )


def _generate_prompt_base_model(task_description: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(COMPARE_BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        COMPARE_BASE_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [{"role": "user", "content": task_description}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

    output_ids = generated_ids[0].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = model_inputs["input_ids"].shape[1]

    prompt = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt


def _generate_prompt_lora_model(task_description: str, adapter_path: str) -> str:
    generator = PromptGenerator(
        model_name=COMPARE_BASE_MODEL_PATH,
        adapter_path=adapter_path,
    )
    generator.model.eval()
    with torch.no_grad():
        prompt = generator.generate_prompt_without_log_prob(task_description)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt


def _build_protocol(prompt_text: str, protocol_generator: ProtocolGenerator, out_dir: Path, round_idx: int) -> str:
    protocol_name = f"protocol_round_{round_idx}"
    full_prompt = prompt_text + "\n\n" + HARD_CONSTRAINT
    protocol_path = protocol_generator.generate_protocol(
        prompt=full_prompt,
        protocol_name=protocol_name,
        save_dir=str(out_dir),
    )
    return protocol_path.read_text(encoding="utf-8")


def _run_episode(protocol: str, target: int, provider: Any, scenario: EvalScenario) -> List[Dict]:
    agents = {
        f"agent_{i}": GuessNumAgent(
            agent_id=f"agent_{i}",
            provider=provider,
            protocol=protocol,
            initial_guess=0,
            num_choices=scenario.num_choices,
        )
        for i in range(scenario.num_agents)
    }
    env = GuessGamePettingZooEnv(
        agents=list(agents.keys()),
        target_range=scenario.num_choices,
        max_steps=scenario.max_steps,
    )
    runner = GuessGameRunner(env=env, agents=agents)
    return runner.run_episode(target=target)


def _summarize_trajectory(trajectory: List[Dict]) -> dict:
    agent_rewards: Dict[str, float] = {}
    for item in trajectory:
        agent_id = item["agent"]
        agent_rewards[agent_id] = agent_rewards.get(agent_id, 0.0) + float(item.get("reward", 0.0))

    success = bool(agent_rewards) and all(r > 0 for r in agent_rewards.values())
    steps = len(trajectory)
    total_reward = sum(agent_rewards.values()) if agent_rewards else 0.0
    avg_reward = total_reward / len(agent_rewards) if agent_rewards else 0.0
    return {
        "success": success,
        "steps": steps,
        "agent_rewards": agent_rewards,
        "total_agent_reward": total_reward,
        "avg_agent_reward": avg_reward,
    }


def _write_trajectory_markdown(path: Path, method: str, round_idx: int, target: int, trajectory: List[Dict], result: TargetResult) -> None:
    lines = [
        f"# {method} | {result.scenario_name} | Round {round_idx} | Target {target}",
        "",
        f"- num_agents: {result.num_agents}",
        f"- success: {result.success}",
        f"- steps: {result.steps}",
        f"- avg_agent_reward: {result.avg_agent_reward:.4f}",
        f"- total_agent_reward: {result.total_agent_reward:.4f}",
        f"- latency_sec: {result.latency_sec:.3f}",
        f"- agent_rewards: {result.agent_rewards}",
        "",
        "## Trajectory",
        "",
    ]
    for i, item in enumerate(trajectory, start=1):
        lines.extend(
            [
                f"### Step {i}",
                f"- agent: {item.get('agent')}",
                f"- observation: {item.get('observation')}",
                f"- action: {item.get('action')}",
                f"- messages: {item.get('messages')}",
                f"- reward: {item.get('reward')}",
                f"- done: {item.get('done')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_method(method_cfg: MethodConfig, method_dir: Path, scenarios: List[EvalScenario]) -> dict:
    provider = _build_provider()
    protocol_generator = ProtocolGenerator(provider=provider)

    all_results: List[TargetResult] = []
    scenario_summaries: List[dict] = []

    scenario_iter = tqdm(scenarios, desc=f"{method_cfg.name}_scenarios", unit="scenario")
    for scenario in scenario_iter:
        scenario_dir = method_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scenario_results: List[TargetResult] = []
        round_summaries: List[dict] = []

        round_iter = tqdm(
            range(1, scenario.rounds + 1),
            desc=f"{method_cfg.name}_{scenario.name}_rounds",
            unit="round",
            leave=False,
        )
        for round_idx in round_iter:
            round_dir = scenario_dir / f"round_{round_idx}"
            protocols_dir = round_dir / "protocols"
            traj_dir = round_dir / "trajectories"
            round_dir.mkdir(parents=True, exist_ok=True)
            protocols_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)

            task_description = _build_task_description(scenario)
            prompt_start = time.time()
            if method_cfg.use_lora:
                if not method_cfg.adapter_path:
                    raise ValueError(f"adapter_path is required for method {method_cfg.name}")
                prompt_text = _generate_prompt_lora_model(task_description, method_cfg.adapter_path)
            else:
                prompt_text = _generate_prompt_base_model(task_description)
            prompt_latency = time.time() - prompt_start
            (round_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")

            protocol_start = time.time()
            protocol_text = _build_protocol(prompt_text, protocol_generator, protocols_dir, round_idx)
            protocol_latency = time.time() - protocol_start
            (round_dir / "protocol.md").write_text(protocol_text, encoding="utf-8")

            target_results: List[TargetResult] = []
            target_iter = tqdm(
                range(scenario.num_choices),
                desc=f"{method_cfg.name}_{scenario.name}_r{round_idx}_targets",
                unit="target",
                leave=False,
            )
            for target in target_iter:
                t0 = time.time()
                trajectory = _run_episode(protocol_text, target, provider, scenario)
                latency = time.time() - t0
                summary = _summarize_trajectory(trajectory)
                result = TargetResult(
                    scenario_name=scenario.name,
                    num_agents=scenario.num_agents,
                    round_idx=round_idx,
                    target=target,
                    success=bool(summary["success"]),
                    steps=int(summary["steps"]),
                    avg_agent_reward=float(summary["avg_agent_reward"]),
                    total_agent_reward=float(summary["total_agent_reward"]),
                    latency_sec=float(latency),
                    agent_rewards=summary["agent_rewards"],
                )
                target_results.append(result)
                scenario_results.append(result)
                all_results.append(result)

                _write_trajectory_markdown(
                    path=traj_dir / f"target_{target:02d}.md",
                    method=method_cfg.name,
                    round_idx=round_idx,
                    target=target,
                    trajectory=trajectory,
                    result=result,
                )

                if (target + 1) % max(1, COMPARE_LOG_EVERY_TARGET) == 0 or (target + 1) == scenario.num_choices:
                    done = target + 1
                    cur_sr = sum(1 for item in target_results if item.success) / len(target_results)
                    cur_steps = mean([item.steps for item in target_results]) if target_results else 0.0
                    tqdm.write(
                        f"[{method_cfg.name}|{scenario.name}] round={round_idx} target={done}/{scenario.num_choices} "
                        f"success_rate={cur_sr:.3f} avg_steps={cur_steps:.3f}"
                    )

            with (round_dir / "target_results.jsonl").open("w", encoding="utf-8") as f:
                for item in target_results:
                    f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

            successes = sum(1 for item in target_results if item.success)
            round_summary = {
                "scenario_name": scenario.name,
                "num_agents": scenario.num_agents,
                "round_idx": round_idx,
                "num_targets": len(target_results),
                "successes": successes,
                "success_rate": successes / len(target_results) if target_results else 0.0,
                "avg_steps": mean([item.steps for item in target_results]) if target_results else 0.0,
                "avg_latency_sec": mean([item.latency_sec for item in target_results]) if target_results else 0.0,
                "prompt_latency_sec": prompt_latency,
                "protocol_latency_sec": protocol_latency,
            }
            round_summaries.append(round_summary)
            (round_dir / "round_summary.json").write_text(
                json.dumps(round_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        scenario_summary = {
            "scenario_name": scenario.name,
            "num_agents": scenario.num_agents,
            "num_rounds": scenario.rounds,
            "num_targets_per_round": scenario.num_choices,
            "max_steps": scenario.max_steps,
            "num_episodes": len(scenario_results),
            "success_rate": mean([rs["success_rate"] for rs in round_summaries]) if round_summaries else 0.0,
            "avg_steps": mean([item.steps for item in scenario_results]) if scenario_results else 0.0,
            "avg_latency_sec": mean([item.latency_sec for item in scenario_results]) if scenario_results else 0.0,
            "avg_agent_reward": mean([item.avg_agent_reward for item in scenario_results]) if scenario_results else 0.0,
            "avg_prompt_latency_sec": mean([rs["prompt_latency_sec"] for rs in round_summaries]) if round_summaries else 0.0,
            "avg_protocol_latency_sec": mean([rs["protocol_latency_sec"] for rs in round_summaries]) if round_summaries else 0.0,
            "round_summaries": round_summaries,
        }
        scenario_summaries.append(scenario_summary)
        (scenario_dir / "scenario_summary.json").write_text(
            json.dumps(scenario_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "method": method_cfg.name,
        "num_scenarios": len(scenario_summaries),
        "num_episodes": len(all_results),
        "success_rate": mean([item["success_rate"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "avg_steps": mean([item["avg_steps"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "avg_latency_sec": mean([item["avg_latency_sec"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "avg_agent_reward": mean([item["avg_agent_reward"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "avg_prompt_latency_sec": mean([item["avg_prompt_latency_sec"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "avg_protocol_latency_sec": mean([item["avg_protocol_latency_sec"] for item in scenario_summaries]) if scenario_summaries else 0.0,
        "scenario_summaries": scenario_summaries,
    }

    (method_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (method_dir / "all_results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(TargetResult.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_results:
            writer.writerow(asdict(item))

    return {"summary": summary, "results": all_results}


def _write_comparison_table(run_dir: Path, method_results: Dict[str, List[TargetResult]], base_method_name: str = "base_model") -> None:
    base_rows = method_results.get(base_method_name, [])
    if not base_rows:
        return

    by_method_index: Dict[str, Dict[tuple, TargetResult]] = {}
    for method_name, rows in method_results.items():
        mapping: Dict[tuple, TargetResult] = {}
        for row in rows:
            key = (row.scenario_name, row.num_agents, row.round_idx, row.target)
            mapping[key] = row
        by_method_index[method_name] = mapping

    others = [name for name in method_results.keys() if name != base_method_name]

    fieldnames = [
        "scenario_name",
        "num_agents",
        "round_idx",
        "target",
        "base_success",
        "base_steps",
        "base_latency_sec",
        "base_avg_agent_reward",
    ]
    for name in others:
        fieldnames.extend(
            [
                f"{name}_success",
                f"{name}_steps",
                f"{name}_latency_sec",
                f"{name}_avg_agent_reward",
                f"{name}_delta_success",
                f"{name}_delta_steps",
                f"{name}_delta_latency_sec",
            ]
        )

    with (run_dir / "episode_level_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for base in base_rows:
            key = (base.scenario_name, base.num_agents, base.round_idx, base.target)
            row = {
                "scenario_name": base.scenario_name,
                "num_agents": base.num_agents,
                "round_idx": base.round_idx,
                "target": base.target,
                "base_success": base.success,
                "base_steps": base.steps,
                "base_latency_sec": f"{base.latency_sec:.6f}",
                "base_avg_agent_reward": f"{base.avg_agent_reward:.6f}",
            }
            for name in others:
                other = by_method_index[name].get(key)
                if other is None:
                    continue
                row[f"{name}_success"] = other.success
                row[f"{name}_steps"] = other.steps
                row[f"{name}_latency_sec"] = f"{other.latency_sec:.6f}"
                row[f"{name}_avg_agent_reward"] = f"{other.avg_agent_reward:.6f}"
                row[f"{name}_delta_success"] = int(other.success) - int(base.success)
                row[f"{name}_delta_steps"] = other.steps - base.steps
                row[f"{name}_delta_latency_sec"] = f"{(other.latency_sec - base.latency_sec):.6f}"
            writer.writerow(row)


def _write_charts(charts_dir: Path, summaries: Dict[str, dict]) -> List[Path]:
    charts: List[Path] = []
    labels = list(summaries.keys())

    sr_values = [summaries[name]["success_rate"] for name in labels]
    plt.figure(figsize=(max(8, len(labels) * 1.6), 5))
    bars = plt.bar(labels, sr_values)
    for bar, val in zip(bars, sr_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    plt.title("Overall Success Rate Comparison")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()
    p1 = charts_dir / "overall_success_rate_comparison.png"
    plt.savefig(p1, dpi=180)
    plt.close()
    charts.append(p1)

    steps_values = [summaries[name]["avg_steps"] for name in labels]
    plt.figure(figsize=(max(8, len(labels) * 1.6), 5))
    bars = plt.bar(labels, steps_values)
    for bar, val in zip(bars, steps_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.title("Overall Average Steps Comparison")
    plt.ylabel("Average Steps")
    plt.xticks(rotation=20)
    plt.tight_layout()
    p2 = charts_dir / "overall_avg_steps_comparison.png"
    plt.savefig(p2, dpi=180)
    plt.close()
    charts.append(p2)

    latency_values = [summaries[name]["avg_latency_sec"] for name in labels]
    plt.figure(figsize=(max(8, len(labels) * 1.6), 5))
    bars = plt.bar(labels, latency_values)
    for bar, val in zip(bars, latency_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.title("Overall Average Episode Latency Comparison")
    plt.ylabel("Latency (sec)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    p3 = charts_dir / "overall_avg_latency_comparison.png"
    plt.savefig(p3, dpi=180)
    plt.close()
    charts.append(p3)

    if not labels:
        return charts

    scenario_names = [
        item["scenario_name"]
        for item in summaries[labels[0]].get("scenario_summaries", [])
    ]
    if scenario_names:
        x = list(range(len(scenario_names)))
        width = max(0.8 / max(1, len(labels)), 0.12)
        plt.figure(figsize=(max(10, len(scenario_names) * 2.2), 5))
        for idx, name in enumerate(labels):
            sc_map = {item["scenario_name"]: item for item in summaries[name].get("scenario_summaries", [])}
            ys = [sc_map.get(s, {"success_rate": 0.0})["success_rate"] for s in scenario_names]
            offsets = [i + (idx - (len(labels) - 1) / 2) * width for i in x]
            plt.bar(offsets, ys, width=width, label=name)
        plt.xticks(x, scenario_names, rotation=20)
        plt.ylabel("Success Rate")
        plt.title("Success Rate by Scenario")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        p4 = charts_dir / "scenario_success_rate_comparison.png"
        plt.savefig(p4, dpi=180)
        plt.close()
        charts.append(p4)

    return charts


def _write_report(run_dir: Path, methods: List[MethodConfig], summaries: Dict[str, dict], charts: List[Path], total_time_sec: float) -> None:
    base_name = "base_model"
    base_summary = summaries.get(base_name)

    lines = [
        "# Guess Task Multi-Adapter Comparison Report",
        "",
        "## Setup",
        "",
        f"- base_model_path: {COMPARE_BASE_MODEL_PATH}",
        f"- adapter_root: {COMPARE_RL_ADAPTER_ROOT}",
        f"- provider_model: {COMPARE_PROVIDER_MODEL}",
        f"- base scenario: choices={COMPARE_NUM_CHOICES}, agents={COMPARE_BASE_NUM_AGENTS}, rounds={COMPARE_ROUNDS}, max_steps={COMPARE_MAX_STEPS}",
        f"- evaluated_methods: {[m.name for m in methods]}",
        f"- total_time_sec: {total_time_sec:.2f}",
        "",
        "## Overall Summary Metrics",
        "",
        "| method | success_rate | avg_steps | avg_latency_sec | avg_agent_reward | avg_prompt_latency_sec | avg_protocol_latency_sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for method in methods:
        s = summaries[method.name]
        lines.append(
            f"| {method.name} | {s['success_rate']:.4f} | {s['avg_steps']:.4f} | {s['avg_latency_sec']:.4f} | "
            f"{s['avg_agent_reward']:.4f} | {s['avg_prompt_latency_sec']:.4f} | {s['avg_protocol_latency_sec']:.4f} |"
        )

    lines.extend([
        "",
        "## Deltas vs Base",
        "",
    ])

    if base_summary is not None:
        for method in methods:
            if method.name == base_name:
                continue
            s = summaries[method.name]
            lines.append(f"- {method.name}: success_rate_delta={s['success_rate'] - base_summary['success_rate']:+.4f}, avg_steps_delta={s['avg_steps'] - base_summary['avg_steps']:+.4f}, avg_latency_sec_delta={s['avg_latency_sec'] - base_summary['avg_latency_sec']:+.4f}")

    lines.extend([
        "",
        "## Scenario Breakdown (Success Rate)",
        "",
        "| scenario | num_agents | " + " | ".join([m.name for m in methods]) + " |",
        "|---|---:|" + "|".join(["---:" for _ in methods]) + "|",
    ])

    if methods:
        first_name = methods[0].name
        scenario_names = [item["scenario_name"] for item in summaries[first_name].get("scenario_summaries", [])]
        for scenario_name in scenario_names:
            num_agents = None
            values = []
            for method in methods:
                sc_map = {item["scenario_name"]: item for item in summaries[method.name].get("scenario_summaries", [])}
                sc = sc_map.get(scenario_name)
                if sc is None:
                    values.append("N/A")
                else:
                    if num_agents is None:
                        num_agents = sc["num_agents"]
                    values.append(f"{sc['success_rate']:.4f}")
            lines.append(f"| {scenario_name} | {num_agents if num_agents is not None else 'N/A'} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Charts",
        "",
    ])
    for chart in charts:
        lines.append(f"- {chart.relative_to(run_dir)}")

    lines.extend([
        "",
        "## Detailed Outputs",
        "",
        "- methods/<method>/<scenario>/round_*/prompt.md",
        "- methods/<method>/<scenario>/round_*/protocol.md",
        "- methods/<method>/<scenario>/round_*/target_results.jsonl",
        "- methods/<method>/<scenario>/round_*/trajectories/target_*.md",
        "- methods/<method>/<scenario>/scenario_summary.json",
        "- episode_level_comparison.csv",
    ])

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_compare_eval() -> None:
    methods = _build_methods()
    if len(methods) <= 1:
        raise RuntimeError(
            "No LoRA adapters found. Please place adapter folders under "
            f"{COMPARE_RL_ADAPTER_ROOT} or set COMPARE_RL_ADAPTER_INCLUDE."
        )

    dirs = _prepare_run_dirs(methods)
    run_dir = dirs["run_dir"]
    scenarios = _build_scenarios()

    config = {
        "base_model_path": COMPARE_BASE_MODEL_PATH,
        "adapter_root": COMPARE_RL_ADAPTER_ROOT,
        "adapter_include": COMPARE_RL_ADAPTER_INCLUDE,
        "detected_methods": [asdict(m) for m in methods],
        "provider_model": COMPARE_PROVIDER_MODEL,
        "base_num_choices": COMPARE_NUM_CHOICES,
        "base_num_agents": COMPARE_BASE_NUM_AGENTS,
        "base_max_steps": COMPARE_MAX_STEPS,
        "base_rounds": COMPARE_ROUNDS,
        "scenarios": [asdict(s) for s in scenarios],
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Guess Compare Eval] run_dir={run_dir}", flush=True)
    print(f"[Guess Compare Eval] methods={[m.name for m in methods]}", flush=True)
    total_start = time.time()

    summaries: Dict[str, dict] = {}
    method_results: Dict[str, List[TargetResult]] = {}

    for method in methods:
        tqdm.write(f"\n=== Evaluating method: {method.name} ===")
        method_dir = dirs[f"method:{method.name}"]
        out = _evaluate_method(method, method_dir, scenarios)
        summaries[method.name] = out["summary"]
        method_results[method.name] = out["results"]

    _write_comparison_table(run_dir, method_results, base_method_name="base_model")
    charts = _write_charts(dirs["charts"], summaries)

    (run_dir / "comparison_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_time_sec = time.time() - total_start
    _write_report(run_dir, methods, summaries, charts, total_time_sec)

    print("[Guess Compare Eval Done]", flush=True)
    for method in methods:
        print(f"- {method.name} success_rate: {summaries[method.name]['success_rate']:.4f}", flush=True)
    print(f"- report: {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    run_compare_eval()

import csv
import importlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.prompt_rl_gsm8k.config import AutoGenDatasetConfig, AutoGenTask, build_default_tasks


def _extract_gsm8k_final_answer(answer: str) -> str:
    text = str(answer).strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    text = text.replace(",", "").strip()
    return text


def _normalize_record(
    record: Dict,
    index: int,
    task_id_field: str,
    prompt_field: str,
    answer_field: str,
    task_type: str,
) -> AutoGenTask | None:
    prompt = str(record.get(prompt_field, "")).strip()
    answer = str(record.get(answer_field, "")).strip()
    if task_type == "gsm8k":
        answer = _extract_gsm8k_final_answer(answer)
    if not prompt or not answer:
        return None

    task_id = str(record.get(task_id_field, "")).strip() or f"task_{index + 1}"
    return AutoGenTask(task_id=task_id, prompt=prompt, expected_answer=answer)


def _load_json(path: Path) -> Iterable[Dict]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        if "data" in content and isinstance(content["data"], list):
            return content["data"]
        raise ValueError("JSON 文件必须是 list，或包含 list 类型的 data 字段")
    raise ValueError("JSON 文件格式不合法")


def _load_jsonl(path: Path) -> Iterable[Dict]:
    records: List[Dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 解析失败，行 {line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL 第 {line_no} 行不是对象")
        records.append(obj)
    return records


def _load_csv(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_local_records(path: Path) -> Iterable[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError("仅支持 .jsonl/.json/.csv")


def _load_hf_records(cfg: AutoGenDatasetConfig) -> Iterable[Dict]:
    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = getattr(datasets_module, "load_dataset")
    except Exception as exc:
        raise ImportError("使用 HuggingFace 数据集需要先安装: pip install datasets") from exc

    if not cfg.hf_path:
        raise ValueError("hf_path 不能为空")

    if cfg.hf_name:
        ds = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.hf_split)
    else:
        ds = load_dataset(cfg.hf_path, split=cfg.hf_split)
    return [dict(x) for x in ds]


def load_tasks(cfg: AutoGenDatasetConfig | None = None) -> List[AutoGenTask]:
    cfg = cfg or AutoGenDatasetConfig()

    if cfg.source == "default":
        tasks = build_default_tasks()
    elif cfg.source == "local":
        if not cfg.local_path:
            raise ValueError("source=local 时 local_path 不能为空")
        path = Path(cfg.local_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")
        records = _load_local_records(path)
        tasks = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            task = _normalize_record(
                record=record,
                index=idx,
                task_id_field=cfg.task_id_field,
                prompt_field=cfg.prompt_field,
                answer_field=cfg.answer_field,
                task_type=cfg.task_type,
            )
            if task is not None:
                tasks.append(task)
    elif cfg.source == "hf":
        records = _load_hf_records(cfg)
        tasks = []
        for idx, record in enumerate(records):
            task = _normalize_record(
                record=record,
                index=idx,
                task_id_field=cfg.task_id_field,
                prompt_field=cfg.prompt_field,
                answer_field=cfg.answer_field,
                task_type=cfg.task_type,
            )
            if task is not None:
                tasks.append(task)
    else:
        raise ValueError("source 仅支持: default/local/hf")

    if cfg.shuffle and len(tasks) > 1:
        rng = random.Random(cfg.seed)
        rng.shuffle(tasks)

    if cfg.limit > 0:
        tasks = tasks[: cfg.limit]

    if not tasks:
        raise ValueError("未加载到有效任务，请检查字段映射和数据内容")

    return tasks


def split_tasks(tasks: List[AutoGenTask], train_ratio: float, seed: int = 42) -> Tuple[List[AutoGenTask], List[AutoGenTask]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1) 区间")
    if len(tasks) < 2:
        return tasks, tasks

    shuffled = list(tasks)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    cut = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return shuffled[:cut], shuffled[cut:]

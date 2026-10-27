"""Shared helpers: paths, config loading, JSONL I/O, stable per-item randomness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LETTERS = "ABCDEFGHIJ"  # MMLU-Pro has at most 10 options


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def item_seed(item_id: str | int, salt: str, base: int) -> int:
    """Deterministic 32-bit seed per item, independent of item order in the file."""
    h = hashlib.sha256(f"{base}|{salt}|{item_id}".encode()).hexdigest()
    return int(h[:8], 16)


def experiment_config(path: str | None = None) -> dict:
    """configs/experiment.yaml (MMLU-Pro) by default; configs/experiment_bfcl.yaml for BFCL."""
    return load_yaml(path or "configs/experiment.yaml")


def task_of(item_or_row: dict) -> str:
    return "bfcl_multiple" if str(item_or_row["item_id"]).startswith("bfcl_") else "mmlu_pro"

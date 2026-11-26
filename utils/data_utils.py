"""
Utility functions for dataset inspection, conversion, and cleaning.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from utils.logger import logger


def load_jsonl(file_path: Path | str) -> List[Dict[str, Any]]:
    """Loads a list of JSON objects from a JSONL file."""
    path = Path(file_path)
    records = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {idx} in {path.name}: {e}")
    return records


def save_jsonl(data: List[Dict[str, Any]], file_path: Path | str) -> Path:
    """Saves a list of dictionaries to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def load_csv_records(file_path: Path | str) -> List[Dict[str, Any]]:
    """Loads CSV as list of dictionaries."""
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


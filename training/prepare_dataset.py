"""Dataset Preparation Pipeline: Loads, validates, splits, and persists datasets."""
import glob
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.config import ROOT_DIR, settings
from datasets.synthetic.generate_synthetic import generate_and_save_synthetic_data
from training.preprocess import (
    clean_query,
    deduplicate_dataset,
    format_chat_prompt,
    validate_sample_against_schemas
)
from utils.logger import logger
from utils.pickle_utils import save_pickle, load_pickle


def load_all_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Loads all JSON schemas from tool_schemas folder."""
    schema_dir = Path(settings.TOOL_SCHEMAS_DIR).resolve()
    schemas = {}
    for p in schema_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                schemas[data["name"]] = data
        except Exception as e:
            logger.error(f"Error reading schema file {p}: {e}")
            
    # Persist schemas into pickle_store
    save_pickle(schemas, settings.PICKLE_STORE_DIR / "tool_schemas.pkl", description="All registered tool schemas")
    return schemas


def prepare_full_dataset() -> Dict[str, Any]:
    """
    Executes full data preparation pipeline:
    1. Loads or generates synthetic tool calls.
    2. Validates against schema definitions.
    3. Formats to standardized ChatML structure.
    4. Splits into Train (80%), Val (10%), Test (10%).
    5. Persists processed splits into JSONL and Pickle store.
    """
    logger.info("Starting Dataset Preparation Pipeline...")
    schemas = load_all_tool_schemas()
    tools_list = list(schemas.values())
    
    # 1. Ensure synthetic data is generated
    synthetic_file = settings.DATASET_DIR / "synthetic" / "synthetic_tool_calls.jsonl"
    if not synthetic_file.exists():
        generate_and_save_synthetic_data(settings.DATASET_DIR / "synthetic")
        
    raw_samples = []
    with open(synthetic_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))
                
    logger.info(f"Loaded {len(raw_samples)} raw samples.")
    
    # 2. Deduplicate
    deduped = deduplicate_dataset(raw_samples)
    
    # 3. Validate against schemas and format
    valid_samples = []
    preference_pairs = []
    label_encoder = {}
    
    for s in deduped:
        is_val, reason = validate_sample_against_schemas(s, schemas)
        if not is_val:
            logger.warning(f"Skipping invalid sample {s.get('id')}: {reason}")
            continue
            
        tool_name = s["tool_call"]["name"]
        if tool_name not in label_encoder:
            label_encoder[tool_name] = len(label_encoder)
            
        chat_messages = format_chat_prompt(s["query"], tools_list)
        assistant_turn = {"role": "assistant", "content": json.dumps(s["tool_call"])}
        
        sample_entry = {
            "id": s.get("id"),
            "messages": chat_messages + [assistant_turn],
            "query": clean_query(s["query"]),
            "tool_call": s["tool_call"],
            "tool_label": label_encoder[tool_name],
            "domain": s.get("domain", "general")
        }
        valid_samples.append(sample_entry)
        
        # Prepare DPO preference pairs if rejected call exists
        if "rejected_tool_call" in s:
            preference_pairs.append({
                "prompt": chat_messages,
                "chosen": json.dumps(s["tool_call"]),
                "rejected": json.dumps(s["rejected_tool_call"])
            })

    logger.info(f"Total valid formatted samples: {len(valid_samples)}")

    # 4. Train / Val / Test Split (Prevent data leakage)
    train_data, temp_data = train_test_split(valid_samples, test_size=0.20, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)
    
    processed_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "tools_included": list(schemas.keys())
    }
    
    # 5. Export JSONL splits
    proc_dir = Path(settings.DATASET_DIR / "processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        split_file = proc_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for item in processed_dict[split_name]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(processed_dict[split_name])} examples to {split_file}")
        
    # 6. Save Mandatory Pickle Files
    pickle_dir = Path(settings.PICKLE_STORE_DIR)
    save_pickle(processed_dict, pickle_dir / "processed_dataset.pkl", description="Train, Val, Test split datasets")
    save_pickle(label_encoder, pickle_dir / "label_encoder.pkl", description="Categorical tool label index mapping")
    save_pickle(preference_pairs, pickle_dir / "preference_pairs.pkl", description="DPO preference chosen/rejected pairs")
    
    dataset_metadata = {
        "version": "1.0.0",
        "sample_counts": {
            "total_raw": len(raw_samples),
            "total_valid": len(valid_samples),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "domains": list(set(s["domain"] for s in valid_samples)),
        "tools": list(schemas.keys())
    }
    save_pickle(dataset_metadata, pickle_dir / "dataset_metadata.pkl", description="Dataset statistics and metadata")
    
    logger.info("Dataset Preparation & Mandatory Pickle Store Serialization complete!")
    return processed_dict


if __name__ == "__main__":
    prepare_full_dataset()


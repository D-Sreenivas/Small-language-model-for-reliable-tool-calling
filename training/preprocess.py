"""Data Preprocessing, Cleaning, Deduplication, and Schema Formatting."""
import json
import hashlib
from typing import Any, Dict, List, Tuple
from utils.logger import logger
from utils.json_validator import validate_against_schema


def clean_query(query: str) -> str:
    """Normalizes whitespace and trims edge punctuation."""
    return " ".join(query.strip().split())


def compute_sample_fingerprint(query: str, tool_name: str, args: Dict[str, Any]) -> str:
    """Generates unique deterministic hash for deduplication."""
    serialized = f"{clean_query(query)}|{tool_name}|{json.dumps(args, sort_keys=True)}"
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def deduplicate_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates samples based on semantic fingerprint."""
    seen = set()
    deduped = []
    for s in samples:
        fp = compute_sample_fingerprint(
            s.get("query", ""),
            s.get("tool_call", {}).get("name", ""),
            s.get("tool_call", {}).get("arguments", {})
        )
        if fp not in seen:
            seen.add(fp)
            deduped.append(s)
    logger.info(f"Deduplication complete: {len(samples)} -> {len(deduped)} samples (removed {len(samples) - len(deduped)} duplicates).")
    return deduped


def format_chat_prompt(query: str, available_tools_schema: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Constructs a Qwen2.5-compatible ChatML conversation structure
    with system tools specification.
    """
    system_prompt = (
        "You are an expert Tool-Calling AI Assistant. Given the available tools and a user query, "
        "your objective is to select the correct tool and output ONLY a valid JSON object matching this schema:\n"
        '{"name": "<tool_name>", "arguments": {<key_value_pairs>}}\n'
        "Never add explanations, commentary, or markdown fences outside the JSON object.\n\n"
        f"Available Tools:\n{json.dumps(available_tools_schema, indent=2)}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clean_query(query)}
    ]
    return messages


def validate_sample_against_schemas(sample: Dict[str, Any], schemas_dict: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    """Validates that a sample's tool call matches known tool schemas."""
    tc = sample.get("tool_call", {})
    name = tc.get("name")
    args = tc.get("arguments", {})
    
    if not name or name not in schemas_dict:
        return False, f"Unknown or missing tool '{name}'"
        
    schema = schemas_dict[name].get("parameters", {})
    is_valid, err = validate_against_schema(args, schema)
    if not is_valid:
        return False, f"Arguments invalid for {name}: {err}"
        
    return True, "Valid"


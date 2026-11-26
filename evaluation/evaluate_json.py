"""JSON Validity and Schema Conformance Evaluator."""
import json
from typing import Any, Dict, List, Tuple
from utils.json_validator import parse_and_validate_json, validate_against_schema
from backend.tools.registry import global_tool_registry


def evaluate_json_and_schema(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes JSON validity rate and schema match percentage across a collection of generations.
    """
    total = len(predictions)
    if total == 0:
        return {"json_validity_rate": 0.0, "schema_validity_rate": 0.0}

    valid_json_count = 0
    valid_schema_count = 0

    for pred in predictions:
        raw = pred.get("raw_output", "")
        is_json, data, _ = parse_and_validate_json(raw)
        if is_json and isinstance(data, dict):
            valid_json_count += 1
            tool_name = data.get("name")
            tool = global_tool_registry.get_tool(tool_name)
            if tool:
                schema = tool.get_schema().get("parameters", {})
                schema_ok, _ = validate_against_schema(data.get("arguments", {}), schema)
                if schema_ok:
                    valid_schema_count += 1

    return {
        "json_validity_rate": valid_json_count / total,
        "schema_validity_rate": valid_schema_count / total,
        "total_evaluated": total
    }


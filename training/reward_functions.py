"""Verifiable Reward Functions for GRPO / Reinforcement Learning."""
import json
from typing import Any, Dict, List
from utils.json_validator import parse_and_validate_json, validate_against_schema
from backend.tools.registry import global_tool_registry
from utils.logger import logger


def reward_json_validity(completion: str, **kwargs) -> float:
    """Reward +1.0 if generated text is structurally sound JSON with tool calling keys."""
    is_valid, data, _ = parse_and_validate_json(completion)
    if is_valid and isinstance(data, dict) and "name" in data and "arguments" in data:
        return 1.0
    return 0.0


def reward_tool_selection(completion: str, target_tool: str, **kwargs) -> float:
    """Reward +1.5 if the chosen tool matches ground truth."""
    is_valid, data, _ = parse_and_validate_json(completion)
    if is_valid and isinstance(data, dict) and data.get("name") == target_tool:
        return 1.5
    return -0.5


def reward_schema_conformance(completion: str, **kwargs) -> float:
    """Reward +2.0 if arguments strictly comply with registered JSON schema."""
    is_valid, data, _ = parse_and_validate_json(completion)
    if not is_valid or not isinstance(data, dict):
        return -1.0
        
    tool_name = data.get("name")
    args = data.get("arguments", {})
    tool = global_tool_registry.get_tool(tool_name)
    if not tool:
        return -1.0
        
    schema = tool.get_schema().get("parameters", {})
    schema_ok, _ = validate_against_schema(args, schema)
    return 2.0 if schema_ok else -1.0


def reward_execution_success(completion: str, **kwargs) -> float:
    """Reward +3.0 if executing tool produces clean result without runtime exceptions."""
    is_valid, data, _ = parse_and_validate_json(completion)
    if not is_valid or not isinstance(data, dict):
        return -1.0
        
    try:
        from backend.schemas.tool_schemas import ToolCall
        tc = ToolCall(name=data.get("name", ""), arguments=data.get("arguments", {}))
        res = global_tool_registry.execute_tool(tc)
        return 3.0 if res.success else -1.0
    except Exception:
        return -2.0


def compute_composite_reward(completion: str, target_tool: str) -> float:
    """Aggregate verifiable rewards for GRPO policy updates."""
    r_json = reward_json_validity(completion)
    r_tool = reward_tool_selection(completion, target_tool)
    r_schema = reward_schema_conformance(completion)
    r_exec = reward_execution_success(completion)
    return r_json + r_tool + r_schema + r_exec


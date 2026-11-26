"""Tool Execution Success and Safety Verification Evaluator."""
from typing import Any, Dict, List
from backend.tools.registry import global_tool_registry
from backend.schemas.tool_schemas import ToolCall
from utils.logger import logger


def evaluate_execution_success(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tests if generated tool calls execute safely without runtime exceptions."""
    total = len(predictions)
    if total == 0:
        return {"execution_success_rate": 0.0}

    successes = 0
    errors = []

    for idx, p in enumerate(predictions):
        tc = p.get("tool_call")
        if not tc or not isinstance(tc, dict):
            continue
        try:
            call_obj = ToolCall(name=tc.get("name", ""), arguments=tc.get("arguments", {}))
            res = global_tool_registry.execute_tool(call_obj, request_id=f"eval_{idx}")
            if res.success:
                successes += 1
            else:
                errors.append({"index": idx, "tool": tc.get("name"), "error": res.error_message})
        except Exception as e:
            errors.append({"index": idx, "tool": tc.get("name"), "error": str(e)})

    return {
        "execution_success_rate": round(successes / total, 4),
        "total_executed": total,
        "error_count": len(errors)
    }


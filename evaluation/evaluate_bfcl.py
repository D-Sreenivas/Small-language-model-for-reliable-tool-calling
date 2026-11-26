"""Berkeley Function-Calling Leaderboard (BFCL) Evaluation Protocol."""
import json
from typing import Any, Dict, List
from utils.logger import logger


def evaluate_bfcl_benchmark(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Implements BFCL evaluation criteria:
    - Tool Selection Exact Match (AST Match)
    - Argument Subtree Exact Match
    """
    total = len(ground_truth)
    if total == 0:
        return {"bfcl_accuracy": 0.0, "ast_match_rate": 0.0}

    tool_matches = 0
    full_ast_matches = 0

    for pred, gt in zip(predictions, ground_truth):
        p_call = pred.get("tool_call") or {}
        gt_call = gt.get("tool_call") or {}

        if p_call.get("name") == gt_call.get("name"):
            tool_matches += 1
            if p_call.get("arguments") == gt_call.get("arguments"):
                full_ast_matches += 1

    return {
        "bfcl_tool_selection_accuracy": round(tool_matches / total, 4),
        "bfcl_full_ast_match_rate": round(full_ast_matches / total, 4),
        "total_bfcl_samples": total
    }


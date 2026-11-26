"""Mock Inference Backend for instant zero-dependency testing and CPU validation."""
import json
import time
import re
from typing import Any, Dict, List, Tuple
from backend.inference.base_model import BaseModelBackend


class MockInferenceBackend(BaseModelBackend):
    """
    Deterministic rule-based mock backend used for tests, CI/CD, and fast verification
    when offline or when an external API key is pending.
    """

    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        start = time.perf_counter()
        q_lower = query.lower()

        # Math matching
        if any(w in q_lower for w in ["multiply", "product", "times", "*", "multiplied"]):
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
            a = nums[0] if len(nums) > 0 else 10.0
            b = nums[1] if len(nums) > 1 else 5.0
            pred = {"name": "calculator", "arguments": {"operation": "multiply", "a": a, "b": b}}
        elif any(w in q_lower for w in ["sqrt", "square root"]):
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
            a = nums[0] if len(nums) > 0 else 16.0
            pred = {"name": "calculator", "arguments": {"operation": "sqrt", "a": a}}
        elif any(w in q_lower for w in ["divide", "divided by", "/"]):
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
            a = nums[0] if len(nums) > 0 else 100.0
            b = nums[1] if len(nums) > 1 else 4.0
            pred = {"name": "calculator", "arguments": {"operation": "divide", "a": a, "b": b}}
        elif any(w in q_lower for w in ["add", "plus", "+", "sum", "subtract", "minus", "-"]):
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
            a = nums[0] if len(nums) > 0 else 50.0
            b = nums[1] if len(nums) > 1 else 25.0
            op = "subtract" if any(w in q_lower for w in ["subtract", "minus", "-"]) else "add"
            pred = {"name": "calculator", "arguments": {"operation": op, "a": a, "b": b}}
        # Conversion matching
        elif any(w in q_lower for w in ["convert", "celsius", "fahrenheit", "feet", "meters", "miles", "pounds", "kilograms"]):
            nums = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
            val = nums[0] if len(nums) > 0 else 100.0
            u1 = "celsius" if "celsius" in q_lower else ("meters" if "meter" in q_lower else "kilograms")
            u2 = "fahrenheit" if "fahrenheit" in q_lower else ("feet" if "feet" in q_lower else "pounds")
            pred = {"name": "unit_converter", "arguments": {"value": val, "from_unit": u1, "to_unit": u2}}
        # Database matching
        elif any(w in q_lower for w in ["order", "inventory", "customer", "database", "table"]):
            pred = {"name": "database_lookup", "arguments": {"table": "customers", "query_field": "email", "query_value": "alex@enterprise.com", "limit": 1}}
        else:
            # Default search
            pred = {"name": "mock_search", "arguments": {"query": query, "category": "general", "max_results": 3}}

        latency = (time.perf_counter() - start) * 1000.0
        return json.dumps(pred), latency

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Mock-SLM-Engine (Ready)",
            "backend": "mock",
            "device": "cpu",
            "vram_allocated_mb": 0.0,
            "status": "ready"
        }

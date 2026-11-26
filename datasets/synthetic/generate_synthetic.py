"""Synthetic Dataset Generation for Tool Calling SFT, DPO, and Multi-turn Traces."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from utils.logger import logger
from utils.pickle_utils import save_pickle

SYNTHETIC_DATA_SAMPLES = [
    # Calculator Samples
    {
        "query": "What is 458 multiplied by 32?",
        "tool_call": {"name": "calculator", "arguments": {"operation": "multiply", "a": 458, "b": 32}},
        "expected_execution": {"status": "success", "result": 14656.0},
        "rejected_tool_call": {"name": "calculator", "arguments": {"operation": "add", "a": 458, "b": 32}},
        "domain": "math"
    },
    {
        "query": "Calculate the square root of 144 please.",
        "tool_call": {"name": "calculator", "arguments": {"operation": "sqrt", "a": 144}},
        "expected_execution": {"status": "success", "result": 12.0},
        "rejected_tool_call": {"name": "calculator", "arguments": {"operation": "sqrt", "a": 144, "b": 2}},
        "domain": "math"
    },
    {
        "query": "Can you compute 2 to the power of 10?",
        "tool_call": {"name": "calculator", "arguments": {"operation": "power", "a": 2, "b": 10}},
        "expected_execution": {"status": "success", "result": 1024.0},
        "rejected_tool_call": {"name": "calculator", "arguments": {"operation": "multiply", "a": 2, "b": 10}},
        "domain": "math"
    },
    {
        "query": "Subtract 950 from 2500.",
        "tool_call": {"name": "calculator", "arguments": {"operation": "subtract", "a": 2500, "b": 950}},
        "expected_execution": {"status": "success", "result": 1550.0},
        "rejected_tool_call": {"name": "calculator", "arguments": {"operation": "subtract", "a": 950, "b": 2500}},
        "domain": "math"
    },
    {
        "query": "Divide 750 by 25.",
        "tool_call": {"name": "calculator", "arguments": {"operation": "divide", "a": 750, "b": 25}},
        "expected_execution": {"status": "success", "result": 30.0},
        "rejected_tool_call": {"name": "calculator", "arguments": {"operation": "divide", "a": 25, "b": 750}},
        "domain": "math"
    },
    # Unit Converter Samples
    {
        "query": "How many feet are in 15 meters?",
        "tool_call": {"name": "unit_converter", "arguments": {"value": 15, "from_unit": "meters", "to_unit": "feet"}},
        "expected_execution": {"status": "success", "result": 49.2126},
        "rejected_tool_call": {"name": "unit_converter", "arguments": {"value": 15, "from_unit": "feet", "to_unit": "meters"}},
        "domain": "conversion"
    },
    {
        "query": "Convert 100 degrees Celsius to Fahrenheit.",
        "tool_call": {"name": "unit_converter", "arguments": {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}},
        "expected_execution": {"status": "success", "result": 212.0},
        "rejected_tool_call": {"name": "unit_converter", "arguments": {"value": 100, "from_unit": "fahrenheit", "to_unit": "celsius"}},
        "domain": "conversion"
    },
    {
        "query": "Convert 150 pounds into kilograms.",
        "tool_call": {"name": "unit_converter", "arguments": {"value": 150, "from_unit": "pounds", "to_unit": "kilograms"}},
        "expected_execution": {"status": "success", "result": 68.0389},
        "rejected_tool_call": {"name": "unit_converter", "arguments": {"value": 150, "from_unit": "kilograms", "to_unit": "pounds"}},
        "domain": "conversion"
    },
    {
        "query": "What is 42 kilometers in miles?",
        "tool_call": {"name": "unit_converter", "arguments": {"value": 42, "from_unit": "kilometers", "to_unit": "miles"}},
        "expected_execution": {"status": "success", "result": 26.0976},
        "rejected_tool_call": {"name": "unit_converter", "arguments": {"value": 42, "from_unit": "miles", "to_unit": "kilometers"}},
        "domain": "conversion"
    },
    # Database Lookup Samples
    {
        "query": "Find the order details for order ID ORD-9821.",
        "tool_call": {"name": "database_lookup", "arguments": {"table": "orders", "query_field": "order_id", "query_value": "ORD-9821", "limit": 1}},
        "expected_execution": {"status": "success", "count": 1},
        "rejected_tool_call": {"name": "database_lookup", "arguments": {"table": "customers", "query_field": "order_id", "query_value": "ORD-9821"}},
        "domain": "database"
    },
    {
        "query": "Look up inventory level for SKU-4001 in our stock.",
        "tool_call": {"name": "database_lookup", "arguments": {"table": "inventory", "query_field": "sku", "query_value": "SKU-4001", "limit": 5}},
        "expected_execution": {"status": "success", "count": 1},
        "rejected_tool_call": {"name": "database_lookup", "arguments": {"table": "orders", "query_field": "sku", "query_value": "SKU-4001"}},
        "domain": "database"
    },
    {
        "query": "Retrieve profile information for customer email alex@enterprise.com.",
        "tool_call": {"name": "database_lookup", "arguments": {"table": "customers", "query_field": "email", "query_value": "alex@enterprise.com", "limit": 1}},
        "expected_execution": {"status": "success", "count": 1},
        "rejected_tool_call": {"name": "database_lookup", "arguments": {"table": "customers", "query_field": "name", "query_value": "alex@enterprise.com"}},
        "domain": "database"
    },
    # Mock Search Samples
    {
        "query": "Search for documentation on python asyncio task group.",
        "tool_call": {"name": "mock_search", "arguments": {"query": "python asyncio task group", "category": "documentation", "max_results": 3}},
        "expected_execution": {"status": "success"},
        "rejected_tool_call": {"name": "mock_search", "arguments": {"query": "python asyncio task group", "category": "news", "max_results": 3}},
        "domain": "search"
    },
    {
        "query": "Find recent news about renewable solar energy breakthroughs.",
        "tool_call": {"name": "mock_search", "arguments": {"query": "renewable solar energy breakthroughs", "category": "news", "max_results": 5}},
        "expected_execution": {"status": "success"},
        "rejected_tool_call": {"name": "mock_search", "arguments": {"query": "renewable solar energy breakthroughs", "category": "general"}},
        "domain": "search"
    },
    {
        "query": "Look up pricing for ultra-lightweight carbon bicycles.",
        "tool_call": {"name": "mock_search", "arguments": {"query": "ultra-lightweight carbon bicycles pricing", "category": "products", "max_results": 4}},
        "expected_execution": {"status": "success"},
        "rejected_tool_call": {"name": "mock_search", "arguments": {"query": "ultra-lightweight carbon bicycles pricing"}},
        "domain": "search"
    }
]


def expand_synthetic_dataset(base_samples: List[Dict[str, Any]], target_count: int = 200) -> List[Dict[str, Any]]:
    """Expands base seed samples into a robust training set with variable phrasing and numbers."""
    expanded = []
    
    # Add base seeds
    for idx, s in enumerate(base_samples):
        item = s.copy()
        item["id"] = f"seed_{idx+1:04d}"
        expanded.append(item)
        
    ops = ["add", "subtract", "multiply", "divide"]
    units_len = ["meters", "feet", "kilometers", "miles"]
    units_temp = ["celsius", "fahrenheit"]
    
    counter = len(expanded) + 1
    while len(expanded) < target_count:
        choice = random.choice(["math", "conversion", "database", "search"])
        
        if choice == "math":
            op = random.choice(ops)
            a = random.randint(10, 500)
            b = random.randint(2, 50)
            query_templates = [
                f"Calculate {op} for {a} and {b}.",
                f"Please {op} {a} with {b}.",
                f"What is {a} {op}ed by {b}?",
                f"Compute {op}({a}, {b}) using calculator."
            ]
            q = random.choice(query_templates)
            good_call = {"name": "calculator", "arguments": {"operation": op, "a": float(a), "b": float(b)}}
            bad_call = {"name": "calculator", "arguments": {"operation": "divide" if op != "divide" else "add", "a": float(a), "b": float(b)}}
            expanded.append({
                "id": f"syn_{counter:04d}",
                "query": q,
                "tool_call": good_call,
                "rejected_tool_call": bad_call,
                "domain": "math"
            })
            
        elif choice == "conversion":
            val = round(random.uniform(5.0, 200.0), 1)
            u1, u2 = random.sample(units_len, 2)
            q = f"Convert {val} {u1} into {u2}."
            good_call = {"name": "unit_converter", "arguments": {"value": val, "from_unit": u1, "to_unit": u2}}
            bad_call = {"name": "unit_converter", "arguments": {"value": val, "from_unit": u2, "to_unit": u1}}
            expanded.append({
                "id": f"syn_{counter:04d}",
                "query": q,
                "tool_call": good_call,
                "rejected_tool_call": bad_call,
                "domain": "conversion"
            })
            
        elif choice == "database":
            table = random.choice(["customers", "inventory", "orders"])
            field_map = {"customers": "customer_id", "inventory": "sku", "orders": "order_id"}
            val = f"VAL-{random.randint(1000, 9999)}"
            q = f"Query {table} records where {field_map[table]} is {val}."
            good_call = {"name": "database_lookup", "arguments": {"table": table, "query_field": field_map[table], "query_value": val, "limit": 5}}
            bad_call = {"name": "database_lookup", "arguments": {"table": table, "query_field": "unknown_col", "query_value": val}}
            expanded.append({
                "id": f"syn_{counter:04d}",
                "query": q,
                "tool_call": good_call,
                "rejected_tool_call": bad_call,
                "domain": "database"
            })
            
        else: # search
            topic = random.choice(["Python 3.12 release notes", "Kubernetes pod scaling", "Microservices architecture patterns", "Vector embeddings"])
            cat = random.choice(["general", "documentation", "news"])
            q = f"Search for {cat} articles on {topic}."
            good_call = {"name": "mock_search", "arguments": {"query": topic, "category": cat, "max_results": 3}}
            bad_call = {"name": "mock_search", "arguments": {"query": topic, "category": "invalid_category"}}
            expanded.append({
                "id": f"syn_{counter:04d}",
                "query": q,
                "tool_call": good_call,
                "rejected_tool_call": bad_call,
                "domain": "search"
            })
            
        counter += 1
        
    return expanded


def generate_and_save_synthetic_data(output_dir: Path | str = "./datasets/synthetic") -> Path:
    """Generates synthetic dataset and writes to JSONL and Pickle."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = expand_synthetic_dataset(SYNTHETIC_DATA_SAMPLES, target_count=200)
    jsonl_path = out_dir / "synthetic_tool_calls.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    logger.info(f"Generated {len(samples)} synthetic examples to {jsonl_path}")
    return jsonl_path


if __name__ == "__main__":
    generate_and_save_synthetic_data()


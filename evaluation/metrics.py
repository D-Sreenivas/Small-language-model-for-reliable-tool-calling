"""Comprehensive Evaluation Suite: Executes BFCL, JSON, Execution, and Latency benchmarks."""
import json
import pandas as pd
from pathlib import Path
from backend.config import settings, ROOT_DIR
from backend.inference.manager import get_inference_backend
from backend.tools.registry import global_tool_registry
from evaluation.evaluate_json import evaluate_json_and_schema
from evaluation.evaluate_bfcl import evaluate_bfcl_benchmark
from evaluation.evaluate_execution import evaluate_execution_success
from evaluation.evaluate_latency import profile_latency
from utils.pickle_utils import load_pickle, save_pickle
from utils.logger import logger


def run_full_evaluation():
    logger.info("Executing Full Tool-Calling Benchmark Evaluation Suite...")
    
    # 1. Load test split from pickle store
    data_file = settings.PICKLE_STORE_DIR / "processed_dataset.pkl"
    if not data_file.exists():
        raise FileNotFoundError("Processed dataset pickle not found.")
        
    ds = load_pickle(data_file)
    test_samples = ds.get("test", [])
    logger.info(f"Loaded {len(test_samples)} test samples.")

    # 2. Run inference across test set
    backend = get_inference_backend()
    all_schemas = global_tool_registry.get_all_schemas()

    predictions = []
    latencies = []
    
    for s in test_samples:
        raw_out, lat = backend.generate_tool_call(query=s["query"], tools_schema=all_schemas)
        latencies.append(lat)
        
        # Parse JSON
        import json
        try:
            parsed = json.loads(raw_out)
        except Exception:
            parsed = None

        predictions.append({
            "id": s.get("id"),
            "query": s["query"],
            "raw_output": raw_out,
            "tool_call": parsed,
            "latency_ms": lat
        })

    # 3. Compute Metrics
    json_metrics = evaluate_json_and_schema(predictions)
    bfcl_metrics = evaluate_bfcl_benchmark(predictions, test_samples)
    exec_metrics = evaluate_execution_success(predictions)
    lat_metrics = profile_latency(latencies)

    # 4. Aggregate Summary
    summary = {
        "json_validity_rate": json_metrics["json_validity_rate"],
        "schema_validity_rate": json_metrics["schema_validity_rate"],
        "tool_selection_accuracy": bfcl_metrics["bfcl_tool_selection_accuracy"],
        "full_ast_match_rate": bfcl_metrics["bfcl_full_ast_match_rate"],
        "execution_success_rate": exec_metrics["execution_success_rate"],
        **lat_metrics
    }

    full_report = {
        "summary": summary,
        "bfcl_metrics": bfcl_metrics,
        "json_metrics": json_metrics,
        "execution_metrics": exec_metrics,
        "latency_records": predictions,
        "test_sample_count": len(test_samples)
    }

    # 5. Persist to Mandatory Pickle Store and Export Reports
    out_pickle = settings.PICKLE_STORE_DIR / "evaluation_results.pkl"
    save_pickle(full_report, out_pickle, description="Full benchmark evaluation results and telemetry")

    # Export CSV & JSON reports
    eval_dir = ROOT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(predictions)
    df.to_csv(eval_dir / "evaluation_report.csv", index=False)
    with open(eval_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation Complete! Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    return full_report


if __name__ == "__main__":
    run_full_evaluation()

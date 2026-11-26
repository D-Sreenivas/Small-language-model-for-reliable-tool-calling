"""Latency Profiler: Measures Time to First Token (TTFT), Mean, and P95 Latencies."""
import time
import numpy as np
from typing import Any, Dict, List


def profile_latency(latency_records: List[float]) -> Dict[str, float]:
    """Calculates latency distribution percentiles."""
    if not latency_records:
        return {"avg_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}

    arr = np.array(latency_records)
    return {
        "avg_latency_ms": round(float(np.mean(arr)), 2),
        "p50_latency_ms": round(float(np.median(arr)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
        "min_latency_ms": round(float(np.min(arr)), 2),
        "max_latency_ms": round(float(np.max(arr)), 2)
    }


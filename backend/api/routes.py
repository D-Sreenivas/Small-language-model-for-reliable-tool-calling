"""FastAPI Routes for Tool Calling, Prediction, Execution, and Monitoring."""
import time
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Depends
from backend.schemas.tool_schemas import (
    PredictionRequest,
    PredictionResponse,
    ToolCall,
    ToolExecutionRequest,
    ToolExecutionResponse
)
from backend.tools.registry import global_tool_registry
from backend.inference.manager import get_inference_backend
from backend.inference.constrained_decoder import ConstrainedJSONEngine
from utils.logger import logger
from utils.pickle_utils import load_pickle
from backend.config import settings

router = APIRouter()
constrained_engine = ConstrainedJSONEngine()


@router.get("/health")
async def health_check():
    """Health and uptime check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/tools")
async def list_available_tools():
    """Returns schemas and documentation of all registered tools."""
    return {
        "count": len(global_tool_registry.list_tool_names()),
        "tools": global_tool_registry.get_all_schemas()
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_tool_call(request: PredictionRequest):
    """
    Predicts tool call from natural language using the SLM inference backend
    with schema constraint verification.
    """
    backend = get_inference_backend()
    all_schemas = global_tool_registry.get_all_schemas()

    # Filter tools if requested
    if request.available_tools:
        all_schemas = [s for s in all_schemas if s["name"] in request.available_tools]

    try:
        raw_output, latency_ms = backend.generate_tool_call(
            query=request.query,
            tools_schema=all_schemas
        )
    except Exception as e:
        logger.error(f"Inference error on query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Constrained Decoding & Schema Validation
    is_valid_json, structured_call, parse_err = constrained_engine.enforce_json_schema(raw_output)
    
    is_schema_valid = False
    validation_err = parse_err
    tool_call_obj = None

    if is_valid_json and structured_call:
        try:
            tool_call_obj = ToolCall(
                name=structured_call.get("name", ""),
                arguments=structured_call.get("arguments", {})
            )
            is_schema_valid, schema_err = global_tool_registry.validate_call(tool_call_obj)
            if not is_schema_valid:
                validation_err = schema_err
        except Exception as ve:
            validation_err = str(ve)

    # Optional immediate execution
    exec_result = None
    if request.execute_immediately and tool_call_obj and is_schema_valid:
        exec_res = global_tool_registry.execute_tool(tool_call_obj, request_id=request.request_id)
        exec_result = exec_res.model_dump()

    model_info = backend.get_info()

    return PredictionResponse(
        request_id=request.request_id,
        query=request.query,
        raw_output=raw_output,
        tool_call=tool_call_obj,
        is_valid_json=is_valid_json,
        is_schema_valid=is_schema_valid,
        validation_error=validation_err,
        execution_result=exec_result,
        latency_ms=round(latency_ms, 2),
        model_name=model_info.get("model_name", "Unknown-SLM")
    )


@router.post("/execute", response_model=ToolExecutionResponse)
async def execute_tool_call(request: ToolExecutionRequest):
    """Executes a validated tool call in a safe Python sandbox."""
    res = global_tool_registry.execute_tool(request.tool_call, request_id=request.request_id)
    return res


@router.get("/model/info")
async def get_model_info():
    """Returns current active model backend, quantization, and memory telemetry."""
    backend = get_inference_backend()
    return backend.get_info()


@router.get("/dataset/info")
async def get_dataset_info():
    """Returns dataset split sizes and metadata from trusted pickle store."""
    meta_path = settings.PICKLE_STORE_DIR / "dataset_metadata.pkl"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset metadata pickle not found. Run prepare_dataset first.")
    
    meta = load_pickle(meta_path)
    return meta


@router.get("/training/status")
async def get_training_status():
    """Checks presence of training artifacts and adapter checkpoints."""
    history_path = settings.PICKLE_STORE_DIR / "training_history.pkl"
    has_history = history_path.exists()
    history = load_pickle(history_path) if has_history else None
    
    return {
        "has_trained_adapter": (settings.MODELS_DIR / "sft_model" / "adapter_model.safetensors").exists(),
        "history_available": has_history,
        "history": history
    }


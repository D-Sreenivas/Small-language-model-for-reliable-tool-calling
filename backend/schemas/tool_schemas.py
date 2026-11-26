"""Pydantic schemas and models for tool calls, validation, and API contracts."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid
import time


class ToolCall(BaseModel):
    """Normalized structured representation of a predicted tool invocation."""
    name: str = Field(..., description="Target tool identifier/function name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema compliant keyword arguments")

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Tool name cannot be empty")
        return cleaned


class ToolDefinition(BaseModel):
    """Specification of an available tool adhering to OpenAI / JSON Schema format."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(
        ...,
        description="JSON Schema object describing required properties, types, defaults, and enums"
    )


class PredictionRequest(BaseModel):
    """Request payload for model tool-call generation."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language user instruction")
    available_tools: Optional[List[str]] = Field(None, description="Subset of tool names to expose to the SLM")
    execute_immediately: bool = Field(False, description="Whether backend executes tool immediately after prediction")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PredictionResponse(BaseModel):
    """Response payload containing predicted tool call and validation status."""
    request_id: str
    query: str
    raw_output: str
    tool_call: Optional[ToolCall] = None
    is_valid_json: bool
    is_schema_valid: bool
    validation_error: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    latency_ms: float
    model_name: str


class ToolExecutionRequest(BaseModel):
    """Explicit request to execute a verified tool call."""
    tool_call: ToolCall
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolExecutionResponse(BaseModel):
    """Result of safe tool execution."""
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    output: Any
    error_message: Optional[str] = None
    execution_time_ms: float


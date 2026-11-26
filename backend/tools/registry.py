"""Tool Registry and Safe Sandboxed Execution Engine."""
import time
from typing import Any, Dict, List, Optional, Tuple
from backend.tools.base import BaseTool
from backend.tools.calculator import CalculatorTool
from backend.tools.unit_converter import UnitConverterTool
from backend.tools.database_lookup import DatabaseLookupTool
from backend.tools.mock_search import MockSearchTool
from backend.schemas.tool_schemas import ToolCall, ToolExecutionResponse
from utils.json_validator import validate_against_schema
from utils.logger import logger


class ToolExecutionError(Exception):
    """Exception raised during tool execution failures."""
    pass


class ToolRegistry:
    """Enterprise Tool Registry with schema validation and access control allowlist."""

    def __init__(self, allowed_tools: Optional[List[str]] = None):
        self._tools: Dict[str, BaseTool] = {}
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self._register_default_tools()

    def _register_default_tools(self):
        default_tool_instances = [
            CalculatorTool(),
            UnitConverterTool(),
            DatabaseLookupTool(),
            MockSearchTool()
        ]
        for tool in default_tool_instances:
            self.register(tool)

    def register(self, tool: BaseTool):
        """Registers a tool instance if permitted by allowlist."""
        if self.allowed_tools is not None and tool.name not in self.allowed_tools:
            logger.warning(f"Tool '{tool.name}' is omitted by ALLOWED_TOOLS config.")
            return
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name.lower().strip())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def validate_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """Validates tool existence and argument schema conformance."""
        tool = self.get_tool(tool_call.name)
        if not tool:
            return False, f"Unknown tool '{tool_call.name}'. Available: {self.list_tool_names()}"
        
        schema = tool.get_schema().get("parameters", {})
        return validate_against_schema(tool_call.arguments, schema)

    def execute_tool(self, tool_call: ToolCall, request_id: str = "") -> ToolExecutionResponse:
        """Safely executes a tool call with strict boundary controls."""
        start_time = time.perf_counter()
        
        # 1. Existence and allowlist check
        tool = self.get_tool(tool_call.name)
        if not tool:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ToolExecutionResponse(
                request_id=request_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                output=None,
                error_message=f"Tool '{tool_call.name}' not found or disallowed.",
                execution_time_ms=round(elapsed, 2)
            )

        # 2. Schema validation
        is_valid, err_msg = self.validate_call(tool_call)
        if not is_valid:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ToolExecutionResponse(
                request_id=request_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                output=None,
                error_message=f"Schema Validation Failed: {err_msg}",
                execution_time_ms=round(elapsed, 2)
            )

        # 3. Safe Execution (sandboxed in python try/except without eval/shell)
        try:
            output = tool.execute(**tool_call.arguments)
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ToolExecutionResponse(
                request_id=request_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=True,
                output=output,
                error_message=None,
                execution_time_ms=round(elapsed, 2)
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Execution error on tool '{tool_call.name}': {e}")
            return ToolExecutionResponse(
                request_id=request_id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                output=None,
                error_message=f"Execution Error: {str(e)}",
                execution_time_ms=round(elapsed, 2)
            )


# Global registry instance
global_tool_registry = ToolRegistry()


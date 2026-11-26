"""Unit tests for tool definitions, schema validation, and safe execution."""
import pytest
from backend.tools.registry import global_tool_registry
from backend.schemas.tool_schemas import ToolCall


def test_calculator_operations():
    calc = global_tool_registry.get_tool("calculator")
    assert calc is not None
    
    # Add
    res = calc.execute(operation="add", a=10, b=5)
    assert res["result"] == 15.0
    
    # Multiply
    res_mul = calc.execute(operation="multiply", a=7, b=8)
    assert res_mul["result"] == 56.0
    
    # Division by zero
    with pytest.raises(ValueError):
        calc.execute(operation="divide", a=10, b=0)


def test_unit_converter():
    conv = global_tool_registry.get_tool("unit_converter")
    assert conv is not None
    
    res = conv.execute(value=0, from_unit="celsius", to_unit="fahrenheit")
    assert res["converted_value"] == 32.0


def test_database_lookup_safe_sql():
    db = global_tool_registry.get_tool("database_lookup")
    assert db is not None
    
    # Valid query
    res = db.execute(table="customers", query_field="email", query_value="alex@enterprise.com")
    assert res["match_count"] >= 1
    
    # SQL injection attempt / unauthorized table
    with pytest.raises(ValueError):
        db.execute(table="sqlite_master", query_field="name", query_value="customers")


def test_tool_registry_validation():
    valid_call = ToolCall(name="calculator", arguments={"operation": "sqrt", "a": 144})
    is_valid, err = global_tool_registry.validate_call(valid_call)
    assert is_valid is True
    assert err is None

    # Missing required argument
    invalid_call = ToolCall(name="calculator", arguments={"operation": "add"})
    is_val, err_msg = global_tool_registry.validate_call(invalid_call)
    assert is_val is False
    assert "Schema Validation Failure" in err_msg


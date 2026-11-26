"""JSON and JSON Schema validation utilities with robust diagnostic messages."""
import json
import re
from typing import Any, Dict, Optional, Tuple
import jsonschema
from jsonschema.exceptions import ValidationError, SchemaError
from utils.logger import logger


def clean_json_markdown(text: str) -> str:
    """Removes ```json markdown blocks or leading/trailing conversational text."""
    text = text.strip()
    # Check for markdown code fences
    json_fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if json_fence:
        return json_fence.group(1).strip()
    
    # Try finding the first '{' and last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()
        
    return text


def parse_and_validate_json(raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses a string into a JSON dictionary safely.
    Returns: (is_valid, parsed_dict, error_message)
    """
    cleaned = clean_json_markdown(raw_text)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return False, None, f"Expected top-level JSON object, got {type(data).__name__}"
        return True, data, None
    except json.JSONDecodeError as jde:
        return False, None, f"JSONDecodeError at line {jde.lineno}, col {jde.colno}: {jde.msg}"
    except Exception as e:
        return False, None, f"Parsing error: {str(e)}"


def validate_against_schema(
    instance: Dict[str, Any],
    schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validates arguments dictionary against standard JSON Schema.
    Returns: (is_valid, error_message)
    """
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True, None
    except ValidationError as ve:
        err_msg = f"Schema Validation Failure: {ve.message} (Path: {'/'.join(map(str, ve.path))})"
        return False, err_msg
    except SchemaError as se:
        return False, f"Invalid JSON Schema definition: {se.message}"


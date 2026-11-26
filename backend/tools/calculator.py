"""Safe Calculator Tool implementation without eval() or arbitrary execution."""
import math
from typing import Any, Dict, Literal
from backend.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Performs basic arithmetic and mathematical operations including add, subtract, multiply, divide, power, and sqrt."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                        "description": "The mathematical operation to perform."
                    },
                    "a": {"type": "number", "description": "The first operand."},
                    "b": {"type": "number", "description": "The second operand (optional for sqrt)."}
                },
                "required": ["operation", "a"]
            }
        }

    def execute(self, operation: str, a: float, b: float = 0.0, **kwargs) -> Dict[str, Any]:
        op = operation.lower().strip()
        a_val = float(a)
        b_val = float(b)

        if op == "add":
            res = a_val + b_val
        elif op == "subtract":
            res = a_val - b_val
        elif op == "multiply":
            res = a_val * b_val
        elif op == "divide":
            if b_val == 0:
                raise ValueError("Division by zero error")
            res = a_val / b_val
        elif op == "power":
            if abs(a_val) > 1e4 or abs(b_val) > 100:
                raise OverflowError("Operands too large for safe exponentiation")
            res = math.pow(a_val, b_val)
        elif op == "sqrt":
            if a_val < 0:
                raise ValueError("Cannot calculate square root of a negative number")
            res = math.sqrt(a_val)
        else:
            raise ValueError(f"Unsupported operation '{op}'")

        return {"operation": op, "result": round(res, 6)}


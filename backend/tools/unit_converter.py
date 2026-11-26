"""Unit Converter Tool supporting length, temperature, and mass."""
from typing import Any, Dict
from backend.tools.base import BaseTool


class UnitConverterTool(BaseTool):
    name = "unit_converter"
    description = "Converts quantities between compatible measurement units (length, weight, temperature)."

    CONVERSIONS = {
        ("celsius", "fahrenheit"): lambda c: (c * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda f: (f - 32) * 5/9,
        ("meters", "feet"): lambda m: m * 3.28084,
        ("feet", "meters"): lambda ft: ft / 3.28084,
        ("kilometers", "miles"): lambda km: km * 0.621371,
        ("miles", "kilometers"): lambda mi: mi / 0.621371,
        ("kilograms", "pounds"): lambda kg: kg * 2.20462,
        ("pounds", "kilograms"): lambda lb: lb / 2.20462,
    }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Magnitude to convert."},
                    "from_unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "meters", "feet", "kilograms", "pounds", "kilometers", "miles"],
                        "description": "Source unit."
                    },
                    "to_unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "meters", "feet", "kilograms", "pounds", "kilometers", "miles"],
                        "description": "Destination unit."
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }

    def execute(self, value: float, from_unit: str, to_unit: str, **kwargs) -> Dict[str, Any]:
        u1 = from_unit.lower().strip()
        u2 = to_unit.lower().strip()
        val = float(value)

        if u1 == u2:
            return {"converted_value": val, "from_unit": u1, "to_unit": u2}

        pair = (u1, u2)
        if pair not in self.CONVERSIONS:
            raise ValueError(f"Incompatible or unsupported unit conversion from '{u1}' to '{u2}'")

        res = self.CONVERSIONS[pair](val)
        return {"converted_value": round(res, 4), "from_unit": u1, "to_unit": u2}


"""Base Tool Interface for safe executable tools."""
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class BaseTool(ABC):
    """Abstract base class that all production tools must inherit from."""
    
    name: str
    description: str
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Returns the JSON Schema specification for this tool."""
        pass
        
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Executes the tool with validated kwargs and returns structured output."""
        pass


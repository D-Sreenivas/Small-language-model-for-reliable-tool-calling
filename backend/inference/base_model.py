"""Base Model Inference Interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseModelBackend(ABC):
    """Abstract interface for SLM inference engines (HF, llama.cpp, vLLM, Mock)."""

    @abstractmethod
    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        """
        Executes model inference on user query and tool schemas.
        Returns: (raw_generated_text, latency_ms)
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns model identifier, backend type, device, and memory footprint."""
        pass


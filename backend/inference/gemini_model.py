"""Google Gemini backend for tool-calling using Gemini API."""
import os
import json
import time
from typing import Any, Dict, List, Tuple, Optional
from backend.inference.base_model import BaseModelBackend
from backend.config import settings
from utils.logger import logger


class GeminiBackend(BaseModelBackend):
    """
    Inference backend leveraging Google's Gemini models (e.g. Gemini 3.8 Flash / Gemini 2.5/2.0 Flash)
    via google-genai or google-generativeai SDK, with graceful fallback.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini client if API key is present."""
        if not self.api_key or self.api_key in ["your_gemini_api_key_here", "mock_key"]:
            logger.warning("GEMINI_API_KEY is not set or using placeholder. Gemini calls will fall back to simulation.")
            return

        try:
            # Try new google-genai SDK first
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
            logger.info(f"Initialized Gemini backend with google-genai SDK using {self.model_name}.")
            return
        except ImportError:
            pass

        try:
            # Try google-generativeai SDK
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self.client = genai_legacy.GenerativeModel(self.model_name)
            self._sdk_type = "generativeai"
            logger.info(f"Initialized Gemini backend with google-generativeai SDK using {self.model_name}.")
            return
        except ImportError:
            logger.warning("Neither 'google-genai' nor 'google-generativeai' is installed.")

    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        start = time.perf_counter()

        system_instruction = (
            "You are an expert Tool-Calling AI Assistant. Given the available tools and a user query, "
            "select the correct tool and output ONLY a valid JSON object matching this schema:\n"
            '{"name": "<tool_name>", "arguments": {<key_value_pairs>}}\n'
            "Never add explanations, conversational text, or markdown codeblocks outside the JSON object.\n\n"
            f"Available Tools:\n{json.dumps(tools_schema, indent=2)}"
        )

        if not self.client:
            # Fallback simulated response if API key is not configured or offline
            from backend.inference.mock_model import MockInferenceBackend
            raw, _ = MockInferenceBackend().generate_tool_call(query, tools_schema)
            latency = (time.perf_counter() - start) * 1000.0
            return raw, latency

        try:
            if getattr(self, "_sdk_type", "") == "genai":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_instruction}\n\nUser Query: {query}",
                    config={
                        "temperature": temperature,
                        "response_mime_type": "application/json"
                    }
                )
                raw_text = response.text.strip()
            else:
                prompt = f"{system_instruction}\n\nUser Query: {query}"
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                raw_text = response.text.strip()

            latency_ms = (time.perf_counter() - start) * 1000.0
            return raw_text, latency_ms

        except Exception as e:
            logger.error(f"Gemini API generation error: {e}. Falling back to rule-based engine.")
            from backend.inference.mock_model import MockInferenceBackend
            raw, _ = MockInferenceBackend().generate_tool_call(query, tools_schema)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return raw, latency_ms

    def get_info(self) -> Dict[str, Any]:
        has_key = bool(self.api_key and self.api_key not in ["your_gemini_api_key_here", "mock_key"])
        return {
            "model_name": self.model_name,
            "backend": "gemini",
            "device": "cloud_api",
            "api_key_configured": has_key,
            "status": "ready" if self.client else "fallback_ready"
        }

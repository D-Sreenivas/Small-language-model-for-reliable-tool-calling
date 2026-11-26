"""llama.cpp and vLLM optional backends for high-speed edge serving."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from backend.inference.base_model import BaseModelBackend
from training.preprocess import format_chat_prompt
from utils.logger import logger


class LlamaCppBackend(BaseModelBackend):
    """GGUF-based inference backend powered by llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 33):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self._load_gguf()

    def _load_gguf(self):
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from {self.model_path}...")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("Successfully loaded llama.cpp engine.")
        except ImportError:
            logger.warning("llama-cpp-python is not installed. llama.cpp backend is unavailable.")
        except Exception as e:
            logger.error(f"Error loading llama.cpp GGUF model: {e}")

    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        if not self.llm:
            raise RuntimeError("llama.cpp backend is not initialized.")
        
        start = time.perf_counter()
        messages = format_chat_prompt(query, tools_schema)
        
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object"}
        )
        
        raw_text = response["choices"][0]["message"]["content"]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return raw_text, latency_ms

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "backend": "llama_cpp",
            "device": "gpu" if self.n_gpu_layers > 0 else "cpu",
            "n_ctx": self.n_ctx,
            "status": "ready" if self.llm else "not_loaded"
        }


class VLLMBackend(BaseModelBackend):
    """High throughput continuous batching backend using vLLM."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.tp_size = tensor_parallel_size
        self.llm = None
        self._load_vllm()

    def _load_vllm(self):
        try:
            from vllm import LLM
            logger.info(f"Initializing vLLM engine with {self.model_name}...")
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tp_size,
                trust_remote_code=True
            )
            logger.info("Successfully loaded vLLM engine.")
        except ImportError:
            logger.warning("vLLM is not installed (typical on Windows natively). Backend skipped.")
        except Exception as e:
            logger.error(f"Error loading vLLM engine: {e}")

    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        if not self.llm:
            raise RuntimeError("vLLM backend is not initialized.")
            
        from vllm import SamplingParams
        start = time.perf_counter()
        messages = format_chat_prompt(query, tools_schema)
        
        # Format with system template
        prompt_str = json.dumps(messages)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        
        outputs = self.llm.generate([prompt_str], sampling_params)
        raw_text = outputs[0].outputs[0].text
        latency_ms = (time.perf_counter() - start) * 1000.0
        return raw_text, latency_ms

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": "vllm",
            "tensor_parallel_size": self.tp_size,
            "status": "ready" if self.llm else "not_loaded"
        }


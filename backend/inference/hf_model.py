"""Hugging Face Transformers Backend for Qwen2.5-1.5B-Instruct."""
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import torch
from backend.inference.base_model import BaseModelBackend
from training.preprocess import format_chat_prompt
from utils.logger import logger


class HuggingFaceBackend(BaseModelBackend):
    """Production Hugging Face Backend supporting QLoRA adapters and 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads model and tokenizer with optional 4-bit BitsAndBytes quantization."""
        logger.info(f"Loading Hugging Face model: {self.model_name} on {self.device}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            bnb_config = None
            if self.load_in_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )

            # Apply adapter if provided
            if self.adapter_path and Path(self.adapter_path).exists():
                from peft import PeftModel
                logger.info(f"Attaching LoRA adapter from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            if self.device != "cuda":
                self.model.to("cpu")

            self.model.eval()
            logger.info("Successfully loaded Hugging Face model weights.")
        except Exception as e:
            logger.error(f"Failed to load HF model {self.model_name}: {e}. Fallback to mock may be required.")
            raise e

    def generate_tool_call(
        self,
        query: str,
        tools_schema: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        start = time.perf_counter()
        messages = format_chat_prompt(query, tools_schema)
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return raw_output, latency_ms

    def get_info(self) -> Dict[str, Any]:
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return {
            "model_name": self.model_name,
            "backend": "huggingface",
            "device": str(self.device),
            "vram_allocated_mb": vram_mb,
            "adapter": self.adapter_path if self.adapter_path else "None",
            "status": "ready"
        }


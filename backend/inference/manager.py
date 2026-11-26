"""Factory and Manager for Active Inference Backend."""
from backend.config import settings
from backend.inference.base_model import BaseModelBackend
from backend.inference.mock_model import MockInferenceBackend
from utils.logger import logger

_active_backend: BaseModelBackend = None


def get_inference_backend() -> BaseModelBackend:
    """Singleton getter for the selected inference engine with automatic fallback."""
    global _active_backend
    if _active_backend is not None:
        return _active_backend

    mode = settings.MODEL_BACKEND.lower()
    logger.info(f"Configuring inference backend: '{mode}'...")

    if mode == "gemini":
        try:
            from backend.inference.gemini_model import GeminiBackend
            _active_backend = GeminiBackend(
                model_name=settings.BASE_MODEL_NAME,
                api_key=settings.GEMINI_API_KEY
            )
            return _active_backend
        except Exception as e:
            logger.warning(f"Could not load Gemini backend ({e}). Activating MockInferenceBackend fallback.")
            _active_backend = MockInferenceBackend()
            return _active_backend

    elif mode == "huggingface":
        try:
            from backend.inference.hf_model import HuggingFaceBackend
            _active_backend = HuggingFaceBackend(
                model_name=settings.BASE_MODEL_NAME,
                load_in_4bit=settings.LOAD_IN_4BIT,
                device=settings.DEVICE
            )
            return _active_backend
        except Exception as e:
            logger.warning(f"Could not load HuggingFace backend ({e}). Activating MockInferenceBackend fallback.")
            _active_backend = MockInferenceBackend()
            return _active_backend

    elif mode == "llama_cpp":
        try:
            from backend.inference.vllm_llama_cpp import LlamaCppBackend
            gguf_path = settings.MODELS_DIR / "quantized" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            _active_backend = LlamaCppBackend(model_path=str(gguf_path))
            return _active_backend
        except Exception as e:
            logger.warning(f"llama.cpp initialization failed ({e}); falling back to Mock.")
            _active_backend = MockInferenceBackend()
            return _active_backend

    # Default mock backend for instant responsiveness
    _active_backend = MockInferenceBackend()
    return _active_backend

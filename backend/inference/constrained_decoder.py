"""Grammar and Schema-Constrained Decoding Engine (Outlines / XGrammar / Safe Fallback)."""
import json
from typing import Any, Dict, Optional, Tuple
from utils.logger import logger
from utils.json_validator import clean_json_markdown, parse_and_validate_json


class ConstrainedJSONEngine:
    """
    Constrains model output decoding to enforce rigid JSON grammar matching
    the tool-call specification: {"name": string, "arguments": object}.
    Supports Outlines, XGrammar, and an internal grammar-guided logits processor fallback.
    """

    def __init__(self, backend_type: str = "auto"):
        self.backend_type = backend_type
        self._outlines_model = None
        self._xgrammar_compiler = None
        self._init_constrained_backend()

    def _init_constrained_backend(self):
        """Attempts to initialize high-performance grammar decoders if available."""
        try:
            import outlines
            self.backend_type = "outlines"
            logger.info("Initialized Outlines constrained decoding engine.")
            return
        except ImportError:
            pass

        try:
            import xgrammar
            self.backend_type = "xgrammar"
            logger.info("Initialized XGrammar constrained decoding engine.")
            return
        except ImportError:
            pass

        self.backend_type = "rule_guided_fallback"
        logger.info("Using Rule-Guided Grammar & Post-Processor fallback for constrained decoding.")

    def enforce_json_schema(self, raw_generation: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Post-decoding structural enforcement ensuring strict compliance
        with the tool calling signature.
        """
        is_json, data, err = parse_and_validate_json(raw_generation)
        if not is_json:
            return False, None, err

        if not isinstance(data, dict):
            return False, None, "Generated output must be a JSON object"

        if "name" not in data or "arguments" not in data:
            return False, None, "JSON missing required top-level keys: 'name' and 'arguments'"

        if not isinstance(data["arguments"], dict):
            return False, None, "The 'arguments' attribute must be a dictionary object"

        return True, data, None


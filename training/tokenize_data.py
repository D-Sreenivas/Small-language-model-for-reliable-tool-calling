"""Tokenization and ChatML formatting utilities."""
from typing import Any, Dict, List
from transformers import PreTrainedTokenizer
from utils.pickle_utils import save_pickle
from backend.config import settings


def tokenize_chatml_example(example: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int = 1024) -> Dict[str, Any]:
    """Applies official ChatML template to dialogue messages and tokenizes."""
    messages = example.get("messages", [])
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def save_tokenizer_metadata(tokenizer: PreTrainedTokenizer, output_path: str = "./pickle_store/tokenizer_metadata.pkl"):
    """Saves tokenizer metadata and vocabulary details into pickle store."""
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "chat_template": tokenizer.chat_template
    }
    save_pickle(meta, output_path, description="Tokenizer vocabulary and chat template metadata")


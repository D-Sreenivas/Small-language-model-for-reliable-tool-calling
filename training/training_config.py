"""Training configuration and hyperparameters for SFT, DPO, and GRPO."""
from pydantic import BaseModel, Field
from pathlib import Path


class TrainingConfig(BaseModel):
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./models/sft_model"
    
    # QLoRA / PEFT
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    logging_steps: int = 5
    save_steps: int = 20
    eval_steps: int = 20
    fp16: bool = False
    bf16: bool = False  # Set true if modern Ampere/Ada GPU detected
    seed: int = 42
    
    # DPO Specific
    dpo_output_dir: str = "./models/dpo_model"
    dpo_beta: float = 0.1
    dpo_epochs: int = 2
    
    # GRPO Specific
    grpo_output_dir: str = "./models/grpo_model"
    num_generations: int = 4


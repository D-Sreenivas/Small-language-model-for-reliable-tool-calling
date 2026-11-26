"""Global Application Configuration using Pydantic Settings."""
import os
from pathlib import Path
from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Production application settings with environment variable overrides."""
    
    # Environment & Server
    APP_ENV: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    FRONTEND_PORT: int = 8501
    API_BASE_URL: str = "http://127.0.0.1:8000"
    
    # Model Configuration
    BASE_MODEL_NAME: str = "gemini-3.8-flash"
    MODEL_BACKEND: Literal["huggingface", "llama_cpp", "vllm", "gemini", "mock"] = "gemini"
    GEMINI_API_KEY: str = ""
    DEVICE: str = "auto"
    LOAD_IN_4BIT: bool = True
    MAX_SEQ_LENGTH: int = 2048
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    MAX_NEW_TOKENS: int = 512
    
    # Paths
    DATASET_DIR: Path = Field(default_factory=lambda: ROOT_DIR / "datasets")
    MODELS_DIR: Path = Field(default_factory=lambda: ROOT_DIR / "models")
    PICKLE_STORE_DIR: Path = Field(default_factory=lambda: ROOT_DIR / "pickle_store")
    TOOL_SCHEMAS_DIR: Path = Field(default_factory=lambda: ROOT_DIR / "tool_schemas")
    
    # Security & Tool Engine
    ENFORCE_SCHEMA_CONSTRAINTS: bool = True
    MAX_EXECUTION_TIMEOUT_SECONDS: int = 10
    ALLOWED_TOOLS: List[str] = [
        "calculator",
        "unit_converter",
        "database_lookup",
        "mock_search"
    ]
    
    WEATHER_API_KEY: str = "mock_key"
    SEARCH_API_KEY: str = "mock_key"
    DATABASE_URL: str = "sqlite:///./lookup.db"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

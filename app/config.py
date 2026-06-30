"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Database configuration
    DATABASE_PATH: str = "./dataset_catalog.db"

    # Upload configuration
    MAX_UPLOAD_SIZE_MB: int = 100  # Maximum upload size in MB
    SAMPLE_ROWS: int = 5  # Number of sample rows to extract

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()

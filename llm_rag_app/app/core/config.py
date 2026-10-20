"""Configuration module for the RAG application.

Loads environment variables and defines constants for API keys, model settings and storage paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (the folder containing pyproject.toml), so paths don't depend on the working directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore"))

# OPENAI_API_KEY is preferred; API_KEY is kept for backwards compatibility with existing .env files
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3")
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "60"))

N_RESULTS = int(os.getenv("N_RESULTS", "4"))
# Minimum relevance (cosine similarity, -1 to 1) for a chunk to be used as context
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.1"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

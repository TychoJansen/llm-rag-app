"""Configuration module for the RAG application.

Loads environment variables and defines constants for API keys and model settings.
"""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY")
OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "llama3"

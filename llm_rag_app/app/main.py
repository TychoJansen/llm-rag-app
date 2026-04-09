"""
1. ollama run llama3
2. poetry run python -m uvicorn llm_rag_app.app.main:app --reload
3. poetry run python -m streamlit run frontend/frontend.py
"""

from fastapi import FastAPI
from llm_rag_app.app.api.routes import router

app = FastAPI(title="RAG Document QA API")

app.include_router(router)
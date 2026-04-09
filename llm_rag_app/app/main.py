"""Main FastAPI application module for the RAG Document QA API.

This module sets up the FastAPI app and includes the API routes.
"""

from fastapi import FastAPI

from llm_rag_app.app.api.routes import router

app = FastAPI(title="RAG Document QA API")

app.include_router(router)

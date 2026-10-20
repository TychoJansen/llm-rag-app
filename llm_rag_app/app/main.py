"""Main FastAPI application module for the RAG Document QA API.

This module sets up the FastAPI app and includes the API routes.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from llm_rag_app.app.api.routes import router
from llm_rag_app.app.services.embeddings import get_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model at startup, so the first upload or question doesn't wait for the download.

    Args:
        app (FastAPI): The FastAPI application.

    Yields:
        None: Control back to FastAPI while the app is running.
    """
    get_model()
    yield


app = FastAPI(title="RAG Document QA API", lifespan=lifespan)

app.include_router(router)

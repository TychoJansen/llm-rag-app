from fastapi import FastAPI
from llm_rag_app.app.api import router

app = FastAPI(title="RAG Document QA API")

app.include_router(router)
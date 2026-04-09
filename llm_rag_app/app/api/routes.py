"""API routes for the RAG Document QA application.

This module defines the FastAPI routes for uploading PDFs and querying the RAG system.
"""

import os
import shutil
from typing import Any, Dict

from app.services.pdf_utilities import load_pdf, split_text
from app.services.rag import add_documents, query
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()


# ------------------------
# Upload PDF endpoint
# ------------------------
@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint to upload a PDF document.

    Processes the uploaded PDF by extracting text, splitting into chunks,
    and adding them to the vector database.

    Args:
        file (UploadFile): The uploaded PDF file.

    Returns:
        Dict[str, Any]: Response containing status, message, and number of chunks.

    Raises:
        HTTPException: If processing fails.
    """
    try:
        os.makedirs("data", exist_ok=True)

        file_path = f"data/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        text = load_pdf(file_path)
        chunks = split_text(text)

        add_documents(chunks)

        return {"status": "success", "message": "File processed and stored", "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Ask endpoint (UPDATED)
# ------------------------
@router.get("/ask")
def ask(question: str) -> Dict[str, Any]:
    """Endpoint to ask a question about the uploaded document.

    Queries the RAG system for an answer based on the provided question.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Response containing status, answer, source, and warning.

    Raises:
        HTTPException: If the question is empty or processing fails.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = query(question)

        return {
            "status": "success",
            "answer": result["answer"],
            "source": result["source"],
            "warning": result["warning"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

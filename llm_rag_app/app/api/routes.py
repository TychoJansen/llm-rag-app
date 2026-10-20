"""API routes for the RAG Document QA application.

This module defines the FastAPI routes for uploading PDFs and querying the RAG system.
"""

import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from llm_rag_app.app.core.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from llm_rag_app.app.services.pdf_utilities import chunk_pages, load_pdf
from llm_rag_app.app.services.rag import add_documents, delete_documents, list_documents, query

router = APIRouter()


# ------------------------
# Upload PDF endpoint
# ------------------------
@router.post("/upload")
def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint to upload a PDF document.

    Processes the uploaded PDF by extracting text, splitting into chunks,
    and adding them to the vector database. Re-uploading a file with the same
    name replaces its earlier chunks.

    Args:
        file (UploadFile): The uploaded PDF file.

    Returns:
        Dict[str, Any]: Response containing status, message, and number of chunks.

    Raises:
        HTTPException: If the file is not a PDF, contains no text, or processing fails.
    """
    # Strip any directory components to prevent path traversal
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = DATA_DIR / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        pages = load_pdf(str(file_path))
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process PDF: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF (scanned documents are not supported)")

    try:
        add_documents(chunks, source=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "message": "File processed and stored", "file": filename, "chunks": len(chunks)}


# ------------------------
# Document management endpoints
# ------------------------
@router.get("/documents")
def documents() -> Dict[str, Any]:
    """Endpoint to list the uploaded documents.

    Returns:
        Dict[str, Any]: Response containing the document names.
    """
    return {"status": "success", "documents": list_documents()}


@router.delete("/documents")
def clear_documents() -> Dict[str, Any]:
    """Endpoint to delete all documents from the vector database.

    Returns:
        Dict[str, Any]: Response containing status and message.
    """
    delete_documents()
    return {"status": "success", "message": "All documents deleted"}


# ------------------------
# Ask endpoint
# ------------------------
@router.get("/ask")
def ask(question: str) -> Dict[str, Any]:
    """Endpoint to ask a question about the uploaded document.

    Queries the RAG system for an answer based on the provided question.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Response containing status, answer, source, warning, and sources.

    Raises:
        HTTPException: If the question is empty or processing fails.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = query(question)

        return {
            "status": "success",
            "answer": result["answer"],
            "source": result["source"],
            "warning": result["warning"],
            "sources": result["sources"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

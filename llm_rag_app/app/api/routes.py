from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

from app.services.pdf_utilities import load_pdf, split_text
from app.services.rag import add_documents, query

router = APIRouter()


# ------------------------
# Upload PDF endpoint
# ------------------------
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Endpoint to upload a PDF document."""
    try:
        os.makedirs("data", exist_ok=True)

        file_path = f"data/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        text = load_pdf(file_path)
        chunks = split_text(text)

        add_documents(chunks)

        return {
            "status": "success",
            "message": "File processed and stored",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Ask endpoint (UPDATED)
# ------------------------
@router.get("/ask")
def ask(question: str):
    """Endpoint to ask a question about the uploaded document."""
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = query(question)

        return {
            "status": "success",
            "answer": result["answer"],
            "source": result["source"],
            "warning": result["warning"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
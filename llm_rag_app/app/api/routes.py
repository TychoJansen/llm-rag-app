from fastapi import APIRouter, UploadFile, File
import shutil
import os

from llm_rag_app.app.services.pdf_utilities import load_pdf, split_text
from llm_rag_app.app.services.rag import add_documents, query

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = load_pdf(file_path)
    chunks = split_text(text)

    add_documents(chunks)

    return {"status": "processed", "chunks": len(chunks)}


@router.get("/ask")
def ask(question: str):
    answer = query(question)
    return {"answer": answer}
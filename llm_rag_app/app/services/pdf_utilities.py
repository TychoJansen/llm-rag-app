"""PDF processing utilities for the RAG application.

Provides functions to load text from PDF files and split text into chunks.
"""

from typing import List

from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Load and extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from all pages of the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): Size of each chunk. Defaults to 500.
        overlap (int, optional): Number of characters to overlap between chunks. Defaults to 50.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

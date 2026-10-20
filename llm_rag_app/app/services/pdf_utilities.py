"""PDF processing utilities for the RAG application.

Provides functions to load text from PDF files and split text into sentence-aware chunks.
"""

import re
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

# A sentence ends with . ! or ? (optionally followed by a closing quote/bracket) and whitespace.
# Both checks are lookbehinds, so only the whitespace is removed and the punctuation stays in the sentence.
SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+")


def load_pdf(file_path: str) -> List[Tuple[int, str]]:
    """Load and extract text from a PDF file, page by page.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Tuple[int, str]]: List of (page number starting at 1, page text) tuples.
    """
    reader = PdfReader(file_path)
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, treating paragraph breaks as sentence boundaries.

    Line breaks inside a paragraph are joined, and words hyphenated across lines are merged.

    Args:
        text (str): The text to split.

    Returns:
        List[str]: List of sentences.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        sentences.extend(s.strip() for s in SENTENCE_END.split(paragraph) if s.strip())
    return sentences


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into chunks that don't cut sentences in half.

    Sentences are added to a chunk until it would exceed chunk_size. Each new chunk starts
    with the last sentences of the previous chunk (up to overlap characters) for context.
    Sentences longer than chunk_size are split on whitespace.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): Maximum number of characters per chunk. Defaults to 800.
        overlap (int, optional): Maximum number of characters repeated from the previous chunk. Defaults to 150.

    Returns:
        List[str]: List of text chunks.
    """
    pieces = []
    for sentence in split_sentences(text):
        while len(sentence) > chunk_size:
            cut = sentence.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)

    chunks: List[str] = []
    current: List[str] = []
    length = 0

    for piece in pieces:
        if current and length + len(piece) > chunk_size:
            chunks.append(" ".join(current))

            # Carry over trailing sentences as overlap
            tail: List[str] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length + len(previous) + 1 > overlap:
                    break
                tail.insert(0, previous)
                tail_length += len(previous) + 1
            current, length = tail, tail_length

        current.append(piece)
        length += len(piece) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_pages(pages: List[Tuple[int, str]], chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    """Split the pages of a document into chunks, keeping track of the page number.

    Chunks never cross page boundaries, so every chunk maps to exactly one page.

    Args:
        pages (List[Tuple[int, str]]): List of (page number, page text) tuples.
        chunk_size (int, optional): Maximum number of characters per chunk. Defaults to 800.
        overlap (int, optional): Maximum number of characters repeated from the previous chunk. Defaults to 150.

    Returns:
        List[Dict[str, Any]]: List of chunks as dicts with "text" and "page" keys.
    """
    return [
        {"text": chunk, "page": number}
        for number, text in pages
        for chunk in split_text(text, chunk_size=chunk_size, overlap=overlap)
    ]

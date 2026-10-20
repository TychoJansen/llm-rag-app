"""Embedding service for the RAG application.

Turns text into vectors with a local, multilingual sentence-transformers model, so
questions and documents in different languages (e.g. Dutch and English) can be matched.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from llm_rag_app.app.core.config import EMBEDDING_MODEL

# E5 models are trained with these prefixes and give worse results without them
USE_E5_PREFIXES = "e5" in EMBEDDING_MODEL.lower()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once. The model is downloaded on first use.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document chunks.

    Args:
        texts (List[str]): The chunk texts.

    Returns:
        List[List[float]]: One normalized vector per chunk.
    """
    if USE_E5_PREFIXES:
        texts = [f"passage: {text}" for text in texts]
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> List[float]:
    """Embed a question.

    Args:
        text (str): The question.

    Returns:
        List[float]: Normalized vector for the question.
    """
    if USE_E5_PREFIXES:
        text = f"query: {text}"
    return get_model().encode(text, normalize_embeddings=True).tolist()

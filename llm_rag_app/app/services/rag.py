"""RAG (Retrieval-Augmented Generation) service module.

Provides functions for document ingestion, querying local and remote LLMs,
and managing the RAG pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI, RateLimitError

from llm_rag_app.app.core.config import (
    LOCAL_LLM_TIMEOUT,
    LOCAL_MODEL,
    MIN_RELEVANCE,
    N_RESULTS,
    OLLAMA_BASE_URL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from llm_rag_app.app.db.chroma import collection
from llm_rag_app.app.services.embeddings import embed_documents, embed_query

# Only create the OpenAI client when a key is configured, so the app also runs with just Ollama
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ------------------------
# Document ingestion
# ------------------------
def add_documents(chunks: List[Dict[str, Any]], source: str) -> None:
    """Add document chunks to ChromaDB, replacing earlier chunks of the same source.

    Args:
        chunks (List[Dict[str, Any]]): List of chunks as dicts with "text" and "page" keys.
        source (str): Name of the document the chunks belong to.
    """
    texts = [chunk["text"] for chunk in chunks]

    delete_documents(source)
    collection.add(
        documents=texts,
        embeddings=embed_documents(texts),
        ids=[f"{source}-{i}" for i in range(len(chunks))],
        metadatas=[{"source": source, "page": chunk["page"], "chunk": i} for i, chunk in enumerate(chunks)],
    )


def delete_documents(source: Optional[str] = None) -> None:
    """Delete chunks from ChromaDB.

    Args:
        source (Optional[str]): Only delete chunks of this document. Deletes everything when None.
    """
    if source is None:
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
    else:
        collection.delete(where={"source": source})


def list_documents() -> List[str]:
    """List the names of all documents stored in ChromaDB.

    Returns:
        List[str]: Sorted list of document names.
    """
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    return sorted({m["source"] for m in metadatas if m and "source" in m})


# ------------------------
# Local LLM (Ollama)
# ------------------------
def is_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_local_llm(prompt: str) -> str:
    """Call a local LLM via Ollama.

    Assumes Ollama is already running.
    """
    response = requests.post(
        OLLAMA_URL, json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False}, timeout=LOCAL_LLM_TIMEOUT
    )

    response.raise_for_status()
    return response.json()["response"]


def try_local_llm(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Safely try the local LLM.

    Returns: (answer, error)
    """
    if not is_ollama_running():
        return None, "Ollama is not running. Start it with: `ollama serve`"

    try:
        answer = call_local_llm(prompt)
        return answer, None
    except requests.exceptions.Timeout:
        return None, "Local LLM timeout"
    except Exception as e:
        return None, str(e)


# ------------------------
# Main query function
# ------------------------
def relevance(distance: float) -> float:
    """Convert a ChromaDB distance to a relevance score.

    The collection uses cosine distance (1 - cosine similarity), so the score is the
    cosine similarity between the question and the chunk: higher means more related.

    Args:
        distance (float): Cosine distance returned by ChromaDB.

    Returns:
        float: Cosine similarity between the question and the chunk.
    """
    return 1 - distance


def query(question: str) -> Dict[str, Any]:
    """Query the RAG system for an answer.

    Retrieves relevant context from the vector database and queries LLMs
    (OpenAI first, then local fallback) to generate an answer.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Response containing answer, source (the LLM used), warning, and
            sources (the retrieved chunks with document name and page number).
    """
    # ------------------------
    # Retrieve context safely
    # ------------------------
    if collection.count() == 0:
        return {
            "answer": "❌ No documents uploaded yet. Upload a PDF first.",
            "source": "error",
            "warning": None,
            "sources": [],
        }

    results = collection.query(query_embeddings=[embed_query(question)], n_results=min(N_RESULTS, collection.count()))

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    sources = [
        {
            "document": (meta or {}).get("source"),
            "page": (meta or {}).get("page"),
            "text": text,
            "relevance": round(relevance(distance), 3),
        }
        for text, meta, distance in zip(documents, metadatas, distances)
    ]

    # Drop chunks that are not related to the question, instead of letting the LLM guess
    sources = [s for s in sources if s["relevance"] >= MIN_RELEVANCE]
    if not sources:
        return {
            "answer": "🔍 I couldn't find relevant information about this in the uploaded documents.",
            "source": None,
            "warning": None,
            "sources": [],
        }

    # Label each chunk with its origin so the LLM can tell the fragments apart
    context = "\n\n".join(f"[{s['document']}, page {s['page']}]\n{s['text']}" for s in sources)

    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    # ------------------------
    # Try OpenAI first
    # ------------------------
    if client is None:
        warning = "⚠️ No OpenAI API key configured. Using local LLM..."
    else:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}]
            )

            return {
                "answer": response.choices[0].message.content,
                "source": "openai",
                "warning": None,
                "sources": sources,
            }

        # ------------------------
        # OpenAI failure → fallback
        # ------------------------
        except RateLimitError:
            warning = (
                "⚠️ OpenAI quota exceeded.\n\n"
                "👉 Trying local LLM...\n\n"
                "Fix it here:\n"
                "https://platform.openai.com/account/billing"
            )
        except Exception:
            warning = "⚠️ OpenAI failed. Trying local LLM..."

    # Try local LLM
    answer, error = try_local_llm(prompt)

    if answer:
        return {"answer": answer, "source": "local", "warning": warning, "sources": sources}

    return {"answer": f"❌ All LLMs failed: {error}", "source": "error", "warning": warning, "sources": []}

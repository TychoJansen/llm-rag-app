"""RAG (Retrieval-Augmented Generation) service module.

Provides functions for document ingestion, querying local and remote LLMs,
and managing the RAG pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI, RateLimitError

from llm_rag_app.app.db.chroma import collection

try:
    from llm_rag_app.app.core.config import LOCAL_MODEL, OLLAMA_URL, OPENAI_API_KEY
except ImportError as e:
    print(
        f"Error importing config: {e}, make sure you have a config.py with the loaded api key, ollama url, and local model."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------
# Document ingestion
# ------------------------
def add_documents(chunks: List[str]) -> None:
    """Add document chunks to ChromaDB.

    Args:
        chunks (List[str]): List of text chunks to add to the database.
    """
    collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])


# ------------------------
# Local LLM (Ollama)
# ------------------------
def is_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_local_llm(prompt: str) -> str:
    """Call a local LLM via Ollama.

    Assumes Ollama is already running.
    """
    response = requests.post(OLLAMA_URL, json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False}, timeout=60)

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
def query(question: str) -> Dict[str, Any]:
    """Query the RAG system for an answer.

    Retrieves relevant context from the vector database and queries LLMs
    (OpenAI first, then local fallback) to generate an answer.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Response containing answer, source, and warning.
    """
    # ------------------------
    # Retrieve context safely
    # ------------------------
    results = collection.query(query_texts=[question], n_results=3)

    documents = results.get("documents", [[]])[0]
    context = "\n".join(documents)

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
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])

        return {"answer": response.choices[0].message.content, "source": "openai", "warning": None}

    # ------------------------
    # OpenAI failure → fallback
    # ------------------------
    except Exception as e:

        warning = None

        # Detect quota / rate limit specifically
        if isinstance(e, RateLimitError):
            warning = (
                "⚠️ OpenAI quota exceeded.\n\n"
                "👉 Trying local LLM...\n\n"
                "Fix it here:\n"
                "https://platform.openai.com/account/billing"
            )
        else:
            warning = "⚠️ OpenAI failed. Trying local LLM..."

        # Try local LLM
        answer, error = try_local_llm(prompt)

        if answer:
            return {"answer": answer, "source": "local", "warning": warning or "⚠️ Using local LLM fallback"}

        return {"answer": f"❌ All LLMs failed: {error}", "source": "error", "warning": warning}

    # ------------------------
    # Retrieve context safely
    # ------------------------
    results = collection.query(query_texts=[question], n_results=3)

    documents = results.get("documents", [[]])[0]
    context = "\n".join(documents)

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
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])

        return {"answer": response.choices[0].message.content, "source": "openai", "warning": None}

    # ------------------------
    # OpenAI failure → fallback
    # ------------------------
    except Exception as e:

        warning = None

        # Detect quota / rate limit specifically
        if isinstance(e, RateLimitError):
            warning = (
                "⚠️ OpenAI quota exceeded.\n\n"
                "👉 Trying local LLM...\n\n"
                "Fix it here:\n"
                "https://platform.openai.com/account/billing"
            )
        else:
            warning = "⚠️ OpenAI failed. Trying local LLM..."

        # Try local LLM
        answer, error = try_local_llm(prompt)

        if answer:
            return {"answer": answer, "source": "local", "warning": warning or "⚠️ Using local LLM fallback"}

        return {"answer": f"❌ All LLMs failed: {error}", "source": "error", "warning": warning}

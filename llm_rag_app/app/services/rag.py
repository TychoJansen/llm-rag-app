import requests
from openai import OpenAI, RateLimitError

from llm_rag_app.app.db.chroma import collection
from llm_rag_app.app.core.config import OPENAI_API_KEY
from llm_rag_app.app.core.config import OLLAMA_URL
from llm_rag_app.app.core.config import LOCAL_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def add_documents(chunks: list[str]):
    """Add document chunks to ChromaDB."""
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

# ------------------------
# Local LLM (Ollama)
# ------------------------
def call_local_llm(prompt: str) -> str:
    """
    Call a local LLM via Ollama.
    Make sure Ollama is running:
    ollama run llama3
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    return response.json()["response"]

# ------------------------
# Try OpenAI first, then fallback to local LLM
# ------------------------
def query(question: str):
    """Query the RAG system for an answer."""

    # ------------------------
    # Retrieve context
    # ------------------------
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    # ------------------------
    # Try OpenAI
    # ------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.choices[0].message.content,
            "source": "openai",
            "warning": None
        }

    # ------------------------
    # Quota error → warn + fallback
    # ------------------------
    except RateLimitError:
        warning = (
            "⚠️ OpenAI quota exceeded.\n\n"
            "👉 Using local LLM instead.\n\n"
            "Fix it here:\n"
            "1. https://platform.openai.com/account/billing\n"
            "2. Add a payment method\n"
        )

        try:
            answer = call_local_llm(prompt)

            return {
                "answer": answer,
                "source": "local",
                "warning": warning
            }

        except Exception:
            return {
                "answer": "❌ Local LLM also failed.",
                "source": "error",
                "warning": warning
            }

    # ------------------------
    # Other errors → fallback
    # ------------------------
    except Exception:
        try:
            answer = call_local_llm(prompt)

            return {
                "answer": answer,
                "source": "local",
                "warning": None
            }

        except Exception:
            return {
                "answer": "❌ Both OpenAI and local LLM failed.",
                "source": "error",
                "warning": None
            }
"""Streamlit frontend for the LLM RAG Application.

Provides a ChatGPT-style UI for uploading PDFs and chatting with documents.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
UPLOAD_TIMEOUT = 120
# The backend may first wait for OpenAI and then for the local LLM, so allow plenty of time
ASK_TIMEOUT = 120

EXAMPLE_QUESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "Which names, dates or numbers are mentioned?",
]


def error_detail(response: requests.Response) -> str:
    """Extract the error message from a FastAPI error response.

    Args:
        response (requests.Response): The failed response.

    Returns:
        str: The error detail, or the status code when no detail is available.
    """
    try:
        return response.json().get("detail", f"HTTP {response.status_code}")
    except ValueError:
        return f"HTTP {response.status_code}"


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> Optional[Dict[str, Any]]:
    """Fetch which LLMs are available. Cached briefly, because Streamlit reruns on every interaction.

    Returns:
        Optional[Dict[str, Any]]: Dict with "openai" and "ollama" flags, or None when the backend is unreachable.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def fetch_documents() -> List[Dict[str, Any]]:
    """Fetch the uploaded documents from the backend.

    Returns:
        List[Dict[str, Any]]: Documents with name, pages and chunks, or an empty list when the backend is unreachable.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException:
        return []


def upload_file(name: str, content: bytes) -> None:
    """Upload a PDF to the backend and show the result.

    Args:
        name (str): File name.
        content (bytes): File content.
    """
    files = {"file": (name, content, "application/pdf")}

    try:
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT)
    except requests.exceptions.RequestException:
        st.error(f"❌ Upload of {name} failed. Is the backend running?")
        return

    if response.ok:
        st.session_state.processed_files[hashlib.sha256(content).hexdigest()] = name
        st.toast(f"✅ {name} processed ({response.json().get('chunks')} chunks)")
    else:
        st.error(f"❌ Upload of {name} failed: {error_detail(response)}")


def delete_document(name: Optional[str] = None) -> None:
    """Delete one document, or all documents when name is None, and reset the uploader.

    Args:
        name (Optional[str]): Name of the document to delete. Deletes all documents when None.
    """
    url = f"{BACKEND_URL}/documents" + (f"/{quote(name, safe='')}" if name else "")

    try:
        requests.delete(url, timeout=10).raise_for_status()
    except requests.exceptions.RequestException:
        st.error("❌ Could not delete document(s).")
        return

    # Forget the deleted files, so uploading them again works. Changing the uploader key clears
    # the widget, otherwise the files still in it would be re-uploaded on the next rerun.
    st.session_state.processed_files = {
        file_hash: file_name
        for file_hash, file_name in st.session_state.processed_files.items()
        if name is not None and file_name != name
    }
    st.session_state.uploader_key += 1
    st.rerun()


def ask_backend(question: str) -> Dict[str, Any]:
    """Send a question to the backend.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Dict with answer, warning, source (the LLM used) and sources (the retrieved chunks).
    """
    try:
        response = requests.get(f"{BACKEND_URL}/ask", params={"question": question}, timeout=ASK_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"answer": "❌ The backend took too long to answer.", "warning": None, "source": None, "sources": []}
    except requests.exceptions.RequestException:
        return {
            "answer": "❌ Backend error. Please check if the API is running.",
            "warning": None,
            "source": None,
            "sources": [],
        }

    if response.status_code != 200:
        return {
            "answer": f"❌ Backend error: {error_detail(response)}",
            "warning": None,
            "source": None,
            "sources": [],
        }

    data = response.json()
    return {
        "answer": data.get("answer", ""),
        "warning": data.get("warning"),
        "source": data.get("source"),
        "sources": data.get("sources") or [],
    }


def show_sources(sources: List[Dict[str, Any]]) -> None:
    """Show the document fragments an answer was based on in a collapsible block.

    Args:
        sources (List[Dict[str, Any]]): Retrieved chunks with document, page and text.
    """
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            page = f", page {source['page']}" if source.get("page") else ""
            score = f" · relevance {source['relevance']:.2f}" if source.get("relevance") is not None else ""
            st.markdown(f"**{source.get('document') or 'Unknown document'}{page}**{score}")
            st.caption(source.get("text", ""))


def show_status(health: Optional[Dict[str, Any]]) -> None:
    """Show whether the backend and the LLMs are available.

    Args:
        health (Optional[Dict[str, Any]]): Result of fetch_health().
    """
    if health is None:
        st.markdown("🔴 Backend")
        st.caption("Start it with `poetry run python llm_rag_app/run_app.py`")
        return

    st.markdown(
        f"🟢 Backend &nbsp; {'🟢' if health.get('openai') else '⚪'} OpenAI &nbsp; "
        f"{'🟢' if health.get('ollama') else '🔴'} Ollama"
    )
    if not health.get("openai") and not health.get("ollama"):
        st.caption("⚠️ No LLM available: add an OpenAI key or start Ollama with `ollama serve`")


def show_welcome(has_documents: bool) -> None:
    """Show an introduction and example questions when the chat is empty.

    Args:
        has_documents (bool): Whether any documents are uploaded.
    """
    st.markdown(
        """
Ask questions about your PDFs and get answers based on their content,
with the fragments and page numbers they came from.

1. **Upload** one or more PDFs in the sidebar
2. **Ask** a question below, in any language
3. **Check** the sources under each answer
"""
    )

    if not has_documents:
        st.info("👈 Upload a PDF in the sidebar to get started.")
        return

    st.markdown("**Try asking:**")
    for column, question in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if column.button(question, width="stretch"):
            st.session_state.pending_prompt = question
            st.rerun()


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="centered")

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quota_warning_shown" not in st.session_state:
    st.session_state.quota_warning_shown = False

# Hash -> name of files already sent to the backend, so reruns don't upload them again
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

# Changing the uploader key resets the widget, so a deleted file isn't re-uploaded on the next rerun
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📄 RAG App")

    show_status(fetch_health())

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}"
    )

    new_files = [
        f
        for f in uploaded_files or []
        if hashlib.sha256(f.getvalue()).hexdigest() not in st.session_state.processed_files
    ]
    for uploaded_file in new_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            upload_file(uploaded_file.name, uploaded_file.getvalue())

    documents = fetch_documents()

    if documents:
        st.markdown(f"### Documents ({len(documents)})")

        for document in documents:
            name_column, delete_column = st.columns([5, 1], vertical_alignment="center")
            pages = (
                f"{document['pages']} page{'s' if document['pages'] != 1 else ''} · " if document.get("pages") else ""
            )
            name_column.markdown(f"**{document['name']}**  \n:gray[{pages}{document['chunks']} chunks]")
            if delete_column.button("🗑️", key=f"delete_{document['name']}", help=f"Delete {document['name']}"):
                delete_document(document["name"])

        with st.popover("🗑️ Delete all documents", width="stretch"):
            st.markdown("This removes **all** documents from the database.")
            if st.button("Yes, delete all", type="primary", width="stretch"):
                delete_document()

    if st.session_state.messages and st.button("🧹 Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Chat with your Documents")

# -----------------------------
# CHAT INPUT
# -----------------------------
# The chat input is always shown at the bottom; an example question clicked on the welcome screen is used instead
prompt = st.chat_input("Ask something about your documents...") or st.session_state.pop("pending_prompt", None)

# -----------------------------
# WELCOME / CHAT HISTORY
# -----------------------------
if not st.session_state.messages and not prompt:
    show_welcome(has_documents=bool(documents))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_sources(msg.get("sources", []))
        if msg.get("source"):
            st.caption(f"Answered by: {msg['source']}")

if prompt:
    # --- User message ---
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Assistant response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_backend(prompt)

        answer, warning, source, sources = result["answer"], result["warning"], result["source"], result["sources"]

        # --- Show warning ---
        if warning and not st.session_state.quota_warning_shown:
            st.warning(warning)
            st.session_state.quota_warning_shown = True

        # --- Show answer ---
        if answer:
            if "Ollama is not running" in answer:
                st.error("⚠️ Local LLM Ollama is not running.\n\nStart it in terminal with:\n\n`ollama serve`.")
            elif answer.startswith("❌"):
                st.error(answer)
            else:
                st.write(answer)

        # --- Show sources and model ---
        show_sources(sources)
        if source and source != "error":
            st.caption(f"Answered by: {source}")

    # --- Save assistant message ---
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "source": source if source != "error" else None,
            "sources": sources,
        }
    )

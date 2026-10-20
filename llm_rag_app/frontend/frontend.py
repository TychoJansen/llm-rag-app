"""Streamlit frontend for the LLM RAG Application.

Provides a ChatGPT-style UI for uploading PDFs and chatting with documents.
"""

import hashlib
import os
from typing import Any, Dict

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
UPLOAD_TIMEOUT = 120
# The backend may first wait for OpenAI and then for the local LLM, so allow plenty of time
ASK_TIMEOUT = 120


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


def fetch_documents() -> list:
    """Fetch the names of the uploaded documents from the backend.

    Returns:
        list: Document names, or an empty list when the backend is unreachable.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException:
        return []


def ask_backend(question: str) -> Dict[str, Any]:
    """Send a question to the backend.

    Args:
        question (str): The question to ask.

    Returns:
        Dict[str, Any]: Dict with answer, warning and source.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/ask", params={"question": question}, timeout=ASK_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"answer": "❌ The backend took too long to answer.", "warning": None, "source": None}
    except requests.exceptions.RequestException:
        return {"answer": "❌ Backend error. Please check if the API is running.", "warning": None, "source": None}

    if response.status_code != 200:
        return {"answer": f"❌ Backend error: {error_detail(response)}", "warning": None, "source": None}

    data = response.json()
    return {"answer": data.get("answer", ""), "warning": data.get("warning"), "source": data.get("source")}


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

# Hashes of files already sent to the backend, so reruns don't upload them again
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Changing the uploader key resets the widget, so a deleted file isn't re-uploaded on the next rerun
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📄 RAG App")

    st.markdown(
        """
    ### How it works
    1. Upload a PDF
    2. Ask questions
    3. Get answers from your document

    ### Tech
    - FastAPI
    - ChromaDB
    - OpenAI / Ollama
    """
    )

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key=f"uploader_{st.session_state.uploader_key}")

    if uploaded_file:
        content = uploaded_file.getvalue()
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, content, "application/pdf")}

                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT)
                    if response.ok:
                        st.session_state.processed_files.add(file_hash)
                        st.success(f"✅ Document processed! ({response.json().get('chunks')} chunks)")
                    else:
                        st.error(f"❌ Upload failed: {error_detail(response)}")

                except requests.exceptions.RequestException:
                    st.error("❌ Upload failed. Is the backend running?")

    documents = fetch_documents()
    if documents:
        st.markdown("### Documents")
        for name in documents:
            st.markdown(f"- {name}")

        if st.button("🗑️ Delete all documents"):
            try:
                requests.delete(f"{BACKEND_URL}/documents", timeout=10).raise_for_status()
                st.session_state.processed_files = set()
                st.session_state.uploader_key += 1
                st.rerun()
            except requests.exceptions.RequestException:
                st.error("❌ Could not delete documents.")

    if st.session_state.messages and st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Chat with your Documents")

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            st.caption(f"Source: {msg['source']}")

# -----------------------------
# CHAT INPUT
# -----------------------------
prompt = st.chat_input("Ask something about your document...")

if prompt:
    # --- User message ---
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Assistant response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_backend(prompt)

        answer, warning, source = result["answer"], result["warning"], result["source"]

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

        # --- Show source ---
        if source and source != "error":
            st.caption(f"Source: {source}")

    # --- Save assistant message ---
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "source": source if source != "error" else None}
    )

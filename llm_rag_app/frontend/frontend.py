"""Streamlit frontend for the LLM RAG Application.

Provides a ChatGPT-style UI for uploading PDFs and chatting with documents.
"""

import requests
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="centered")

# -----------------------------
# CUSTOM STYLING (ChatGPT-like)
# -----------------------------
st.markdown(
    """
<style>

/* Main container spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] > div {
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 75%;
}

/* User message */
div[data-testid="stChatMessage"]:has(div:contains("user")) > div {
    background-color: #2b2b2b;
    color: white;
    margin-left: auto;
}

/* Assistant message */
div[data-testid="stChatMessage"]:has(div:contains("assistant")) > div {
    background-color: #f1f1f1;
    color: black;
}

/* Input styling */
.stChatFloatingInputContainer {
    padding-bottom: 1rem;
}

</style>
""",
    unsafe_allow_html=True,
)

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

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing document..."):
            files = {"file": uploaded_file.getvalue()}

            try:
                response = requests.post("http://127.0.0.1:8000/upload", files=files, timeout=60)
                response.raise_for_status()
                st.success("✅ Document processed!")

            except requests.exceptions.RequestException:
                st.error("❌ Upload failed. Is the backend running?")

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Chat with your Documents")

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if "quota_warning_shown" not in st.session_state:
    st.session_state.quota_warning_shown = False

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
            try:
                response = requests.get("http://127.0.0.1:8000/ask", params={"question": prompt}, timeout=30)

                try:
                    data = response.json()
                except Exception:
                    data = {}

                if response.status_code != 200:
                    answer = data.get("answer", "❌ Backend error (no details)")
                    warning = data.get("warning")
                    source = data.get("source")
                else:
                    answer = data.get("answer", "")
                    warning = data.get("warning")
                    source = data.get("source")

            except requests.exceptions.RequestException:
                answer = "⚠️ Backend error. Please check if the API is running."
                warning = None
                source = None

        # --- Show warning ---
        if warning and not st.session_state.quota_warning_shown:
            st.warning(warning)
            st.session_state.quota_warning_shown = True

        # --- Show answer ---
        if answer:
            if "Ollama is not running" in answer:
                st.error("⚠️ Local LLM Ollama is not running.\n\nStart it  in terminal with:\n\n`ollama run llama3`.")
            elif answer.startswith("❌"):
                st.error(answer)
            else:
                st.write(answer)

        # --- Show source ---
        if source:
            st.caption(f"Source: {source}")

    # --- Save assistant message ---
    st.session_state.messages.append({"role": "assistant", "content": answer})

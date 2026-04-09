"""
LLM RAG Application - Streamlit Frontend

This application provides a user-friendly interface for interacting with a 
Retrieval-Augmented Generation (RAG) system.

Features:
- Upload PDF documents for processing
- Ask questions about uploaded documents
- Chat-style interface with conversation memory
- Real-time responses from a local LLM (via Ollama)

Architecture:
- Frontend: Streamlit (this file)
- Backend: FastAPI (see app/main.py)
- Vector Database: ChromaDB (stores document embeddings)

Flow:
1. User uploads a PDF document
2. Backend processes and stores embeddings in ChromaDB
3. User asks a question
4. Relevant document chunks are retrieved

How to run:
- Start backend:
    poetry run python -m uvicorn llm_rag_app.app.main:app --reload

- Start frontend:
    streamlit run frontend.py

Author:
Tycho Jansen
"""

import streamlit as st
import requests

st.sidebar.title("📄 RAG App")

st.sidebar.markdown("""
### How it works
1. Upload a PDF  
2. Ask questions  
3. Get answers from your document  

### Tech
- FastAPI  
- ChromaDB    
""")

st.set_page_config(page_title="RAG App", layout="wide")
st.title("🧠 Ask Your Documents")

# Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            "http://127.0.0.1:8000/upload",
            files=files
        )

    st.success("✅ Document processed! You can now ask questions.")

# Ask question
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
prompt = st.chat_input("Ask something about your document")

if prompt:
    # Show user message
    st.chat_message("user").write(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Call API
    with st.spinner("Thinking..."):
        try:
            response = requests.get(
                "http://127.0.0.1:8000/ask",
                params={"question": prompt},
                timeout=30
            )

            response.raise_for_status()
            answer = response.json()["answer"]

        except requests.exceptions.RequestException:
            answer = "⚠️ Backend error. Please check if the API is running."

    # Show assistant message with proper handling
    if "quota exceeded" in answer.lower() or "insufficient_quota" in answer.lower():
        st.warning(answer)
    else:
        st.chat_message("assistant").write(answer)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
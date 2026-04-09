"""Script to run both the backend FastAPI server and the frontend Streamlit app simultaneously."""

import subprocess
import sys


def start_backend() -> subprocess.Popen:
    """Start the FastAPI backend server using Uvicorn.

    Returns:
        subprocess.Popen: The process object for the backend server.
    """
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "llm_rag_app.app.main:app", "--reload"])


def start_frontend() -> subprocess.Popen:
    """Start the Streamlit frontend application.

    Returns:
        subprocess.Popen: The process object for the frontend app.
    """
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/frontend.py"])


if __name__ == "__main__":
    backend = start_backend()
    frontend = start_frontend()

    backend.wait()
    frontend.wait()

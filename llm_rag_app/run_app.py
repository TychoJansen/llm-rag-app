"""Script to run both the backend FastAPI server and the frontend Streamlit app simultaneously."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PATH = PROJECT_ROOT / "llm_rag_app" / "frontend" / "frontend.py"


def start_backend() -> subprocess.Popen:
    """Start the FastAPI backend server using Uvicorn.

    Returns:
        subprocess.Popen: The process object for the backend server.
    """
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "llm_rag_app.app.main:app", "--reload", "--reload-dir", "llm_rag_app"],
        cwd=PROJECT_ROOT,
    )


def start_frontend() -> subprocess.Popen:
    """Start the Streamlit frontend application.

    Returns:
        subprocess.Popen: The process object for the frontend app.
    """
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(FRONTEND_PATH)], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    backend = start_backend()
    frontend = start_frontend()

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure neither process keeps running when one stops or on Ctrl+C
        for process in (backend, frontend):
            if process.poll() is None:
                process.terminate()

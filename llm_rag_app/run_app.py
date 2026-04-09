import subprocess
import sys

def start_backend():
    return subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "llm_rag_app.app.main:app",
        "--reload"
    ])

def start_frontend():
    return subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/frontend.py"
    ])

if __name__ == "__main__":
    backend = start_backend()
    frontend = start_frontend()

    backend.wait()
    frontend.wait()
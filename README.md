# LLM RAG App

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with them using Large Language Models (LLMs). The app provides answers based on the content of uploaded documents, with fallback support for local LLMs when cloud services are unavailable.

## Features

- **PDF Upload & Processing**: Upload PDF documents and automatically extract text for indexing
- **Vector Search**: Uses ChromaDB for efficient document chunk retrieval
- **Multi-LLM Support**:
  - Primary: OpenAI GPT-4o-mini
  - Fallback: Local Ollama models (e.g., Llama 3)
- **Chat Interface**: Streamlit-based ChatGPT-style UI for natural conversation
- **API Backend**: FastAPI REST API for document upload and querying
- **Error Handling**: Graceful fallbacks and user-friendly error messages

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Database**: ChromaDB (vector database)
- **LLMs**: OpenAI API, Ollama (local)
- **PDF Processing**: PyPDF
- **Dependency Management**: Poetry

## Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- Ollama (for local LLM support)
- OpenAI API key (optional, for cloud LLM)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-rag-app
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   API_KEY=your_openai_api_key_here
   ```

## Setup

### Ollama Setup (for local LLM)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama 3 model:
   ```bash
   ollama pull llama3
   ```
3. Start Ollama server:
   ```bash
   ollama serve
   ```

### OpenAI Setup (optional)

1. Get an API key from [OpenAI Platform](https://platform.openai.com)
2. Add it to your `.env` file as `API_KEY`

## Running the Application

### Option 1: Run both backend and frontend together
```bash
poetry run python llm_rag_app/run_app.py
```

This will start:
- FastAPI backend on `http://127.0.0.1:8000`
- Streamlit frontend on `http://localhost:8501`

### Option 2: Run separately

**Backend only**:
```bash
poetry run uvicorn llm_rag_app.app.main:app --reload
```

**Frontend only**:
```bash
poetry run streamlit run llm_rag_app/frontend/frontend.py
```

## Usage

1. **Upload a PDF**: Use the sidebar in the Streamlit interface to upload a PDF document
2. **Wait for processing**: The app will extract text, split it into chunks, and index them
3. **Ask questions**: Type questions in the chat interface about your document
4. **Get answers**: The app will retrieve relevant context and generate answers using LLMs

## API Endpoints

### POST `/upload`
Upload a PDF document for processing.

**Request**: Multipart form with `file` field containing PDF
**Response**: JSON with processing status and chunk count

### GET `/ask`
Query the RAG system with a question.

**Parameters**:
- `question` (string): The question to ask

**Response**: JSON with answer, source, and warning information

## Project Structure

```
llm-rag-app/
├── llm_rag_app/
│   ├── run_app.py              # Main runner script
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app setup
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py       # API endpoints
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py       # Configuration
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── chroma.py       # ChromaDB setup
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── pdf_utilities.py # PDF processing
│   │       └── rag.py          # RAG logic
│   ├── frontend/
│   │   └── frontend.py         # Streamlit UI
│   └── data/                   # Data directory
├── pyproject.toml              # Poetry configuration
├── README.md                   # This file
└── .env                        # Environment variables
```

## Troubleshooting

### Backend not starting
- Ensure all dependencies are installed: `poetry install`
- Check Python version: `python --version` (should be >=3.10)

### Upload fails
- Verify backend is running on port 8000
- Check file size limits and PDF format

### LLM not responding
- For OpenAI: Verify API key in `.env`
- For Ollama: Ensure `ollama serve` is running and model is pulled

### ChromaDB issues
- Vector store persists in `./vectorstore/` directory
- Delete this directory to reset the database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the UI
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [OpenAI](https://openai.com/) for LLM API
- [Ollama](https://ollama.ai/) for local LLM support

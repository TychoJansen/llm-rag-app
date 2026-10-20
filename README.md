# LLM RAG App

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with them using Large Language Models (LLMs). Answers are based on the content of the uploaded documents, show the fragments and page numbers they came from, and fall back to a local LLM when the cloud service is unavailable.

## Features

- **PDF Upload & Processing**: Upload PDF documents; text is extracted per page and split into sentence-aware chunks
- **Multilingual Search**: Local multilingual embeddings, so you can ask Dutch questions about English documents and vice versa
- **Source Citations**: Every answer shows the document fragments, page numbers and relevance scores it was based on
- **Relevance Threshold**: Questions unrelated to your documents are recognized instead of answered with made-up information
- **Multiple Documents**: Upload several PDFs at once, see which are indexed, and delete them one by one or all at once
- **Status Overview**: See at a glance whether the backend, OpenAI and Ollama are available
- **Multi-LLM Support**:
  - Primary: OpenAI GPT-4o-mini
  - Fallback: Local Ollama models (e.g., Llama 3)
- **Chat Interface**: Streamlit-based ChatGPT-style UI
- **API Backend**: FastAPI REST API for document upload, management and querying
- **Persistent Storage**: Indexed documents survive restarts

## How It Works

### 1. Uploading a document
When you upload a PDF, the backend stores the file in `data/` and extracts the text **page by page** with PyPDF. Each page is then split into chunks:

- The text is split into **sentences**, where the end of a paragraph also counts as a sentence boundary. Words hyphenated across lines are joined again.
- Sentences are combined into chunks of at most `CHUNK_SIZE` characters (800), so no sentence is ever cut in half.
- Each chunk starts with the last sentence(s) of the previous chunk (up to `CHUNK_OVERLAP` characters, 150), so context at the boundary isn't lost.
- Chunks never cross a page boundary, so every chunk belongs to exactly one page.

Uploading a file with the same name again replaces its earlier chunks.

### 2. Embeddings and storage
Every chunk is turned into a vector (an *embedding*) by a local multilingual model (`paraphrase-multilingual-MiniLM-L12-v2` via sentence-transformers). Texts with a similar meaning get similar vectors, even when they use different words or languages. The vectors are stored in **ChromaDB** together with the document name and page number. The model runs on your own machine, so this costs nothing and works offline. It is downloaded once (~470 MB) when the backend starts for the first time.

### 3. Finding relevant chunks
When you ask a question, it is embedded with the same model. ChromaDB returns the `N_RESULTS` (4) chunks whose vectors are closest to the question. Each gets a **relevance score** (cosine similarity): around 0 means unrelated, 0.2–0.5 means related.

Chunks scoring below `MIN_RELEVANCE` (0.18) are dropped. If no chunk is left, the app answers that it couldn't find relevant information, without calling an LLM. This prevents made-up answers to questions your documents don't cover, and saves API calls.

### 4. Generating the answer
The remaining chunks, each labelled with its document and page, are put into a prompt that instructs the LLM to answer **only** from that context. The app first tries **OpenAI**. If no API key is configured, the quota is exceeded or the call fails, it falls back to a local model via **Ollama**. The answer shows which model was used ("Answered by").

### 5. The chat interface
The Streamlit frontend provides:
- **Status indicator** in the sidebar showing whether the backend, OpenAI (API key configured) and Ollama are available.
- **Upload** of one or more PDFs in the sidebar. Each file is sent to the backend only once, even though Streamlit reruns the page on every interaction.
- **Document list** with the number of pages and chunks per document, a delete button per document, and "Delete all documents" with a confirmation.
- **Welcome screen** with a short explanation and clickable example questions while the chat is empty.
- **Chat** with history and a "Clear chat" button.
- **Sources** below each answer: a collapsible block with the fragments used, their document, page and relevance score.
- **Warnings and errors**, for example when the OpenAI quota is exceeded or Ollama isn't running.

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (local, multilingual)
- **LLMs**: OpenAI API, Ollama (local)
- **PDF Processing**: PyPDF
- **Dependency Management**: Poetry

## Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- At least one LLM:
  - an OpenAI API key, and/or
  - Ollama for a local LLM

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TychoJansen/llm-rag-app.git
   cd llm-rag-app
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   The OpenAI key is optional: without it the app uses the local Ollama model. The old `API_KEY` name still works.

## Configuration

All settings are optional and can be set in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | – | OpenAI API key; without it only Ollama is used |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for answers |
| `LOCAL_MODEL` | `llama3` | Ollama model used as fallback |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Address of the Ollama server |
| `LOCAL_LLM_TIMEOUT` | `60` | Seconds to wait for the local LLM |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (see note below) |
| `N_RESULTS` | `4` | Number of chunks retrieved per question |
| `MIN_RELEVANCE` | `0.18` | Minimum relevance score for a chunk to be used |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Maximum characters repeated from the previous chunk |
| `DATA_DIR` | `./data` | Where uploaded PDFs are stored |
| `VECTORSTORE_DIR` | `./vectorstore` | Where ChromaDB stores its data |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Backend address used by the frontend |

**Changing the embedding model**: each model gets its own collection, so you start with an empty database and have to upload your documents again. Different models also produce different score ranges (E5 models score everything between ~0.65 and 0.85), so re-tune `MIN_RELEVANCE`.

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
2. Add it to your `.env` file as `OPENAI_API_KEY`

## Running the Application

All commands are run from the repository root.

### Option 1: Run both backend and frontend together
```bash
poetry run python llm_rag_app/run_app.py
```

This will start:
- FastAPI backend on `http://127.0.0.1:8000`
- Streamlit frontend on `http://localhost:8501`

Press `Ctrl+C` to stop both.

### Option 2: Run separately

**Backend only**:
```bash
poetry run uvicorn llm_rag_app.app.main:app --reload
```

**Frontend only**:
```bash
poetry run streamlit run llm_rag_app/frontend/frontend.py
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## Usage

1. **Upload PDFs**: Use the sidebar in the Streamlit interface to upload one or more PDF documents
2. **Wait for processing**: The app extracts the text per page, splits it into sentence-aware chunks, and indexes them
3. **Ask questions**: Type questions about your documents in the chat, in any language
4. **Get answers**: Open "Sources" under an answer to see the document fragments, page numbers and relevance scores it was based on

## API Endpoints

### POST `/upload`
Upload a PDF document for processing.

**Request**: Multipart form with `file` field containing a PDF
**Response**: JSON with processing status, file name and chunk count

Uploading a file with the same name again replaces its earlier chunks. Returns `400` for non-PDF files and for PDFs without extractable text (e.g. scanned documents).

### GET `/documents`
List the uploaded documents, each with `name`, `pages` and `chunks`.

### DELETE `/documents/{name}`
Delete a single document from the vector database. Returns `404` if it doesn't exist.

### DELETE `/documents`
Delete all documents from the vector database.

### GET `/health`
Check which LLMs are available: `openai` (an API key is configured) and `ollama` (the Ollama server is reachable).

### GET `/ask`
Query the RAG system with a question.

**Parameters**:
- `question` (string): The question to ask

**Response**: JSON with:
- `answer`: the generated answer, or a message that no relevant information was found
- `source`: the LLM that answered (`openai` or `local`)
- `warning`: e.g. when OpenAI failed and the local LLM was used
- `sources`: the chunks used, each with `document`, `page`, `text` and `relevance`

## Project Structure

```
llm-rag-app/
├── llm_rag_app/
│   ├── run_app.py               # Starts backend and frontend together
│   ├── app/
│   │   ├── main.py              # FastAPI app setup, loads the embedding model at startup
│   │   ├── api/
│   │   │   └── routes.py        # API endpoints
│   │   ├── core/
│   │   │   └── config.py        # Configuration from environment variables
│   │   ├── db/
│   │   │   └── chroma.py        # ChromaDB setup
│   │   └── services/
│   │       ├── embeddings.py    # Multilingual embedding model
│   │       ├── pdf_utilities.py # PDF text extraction and chunking
│   │       └── rag.py           # Retrieval, relevance filtering and LLM calls
│   └── frontend/
│       └── frontend.py          # Streamlit UI
├── data/                        # Uploaded PDFs (not in git)
├── vectorstore/                 # ChromaDB data (not in git)
├── .pre-commit-config.yaml      # Code quality hooks
├── pyproject.toml               # Poetry configuration
├── README.md                    # This file
└── .env                         # Environment variables (not in git)
```

## Development

Code style is enforced with pre-commit hooks (black, isort, flake8, pydocstyle and gitleaks). Install them once:
```bash
pre-commit install
```
They then run automatically on every commit. To run them on all files:
```bash
pre-commit run --all-files
```

## Troubleshooting

### Backend not starting
- Ensure all dependencies are installed: `poetry install`
- Check Python version: `python --version` (should be >=3.10)
- The first start downloads the embedding model (~470 MB) and takes a while

### Upload fails
- Verify the backend is running on port 8000
- Scanned PDFs (images without a text layer) are not supported

### "I couldn't find relevant information"
- The question may really not be covered by your documents
- Check the relevance scores under "Sources" for similar questions; if relevant fragments score just below the threshold, lower `MIN_RELEVANCE` (e.g. to `0.15`)
- Documents uploaded before switching the embedding model must be uploaded again

### LLM not responding
- For OpenAI: Verify `OPENAI_API_KEY` in `.env`
- For Ollama: Ensure `ollama serve` is running and the model is pulled

### ChromaDB issues
- The vector store persists in the `./vectorstore/` directory
- Use the "Delete all documents" button, or delete this directory to reset the database

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
- [sentence-transformers](https://www.sbert.net/) for the embedding model
- [OpenAI](https://openai.com/) for LLM API
- [Ollama](https://ollama.ai/) for local LLM support

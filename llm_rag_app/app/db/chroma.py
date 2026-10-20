"""ChromaDB database setup module.

Initializes a persistent ChromaDB client and creates a collection for storing document vectors.
"""

import re

import chromadb

from llm_rag_app.app.core.config import EMBEDDING_MODEL, VECTORSTORE_DIR

client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

# One collection per embedding model: vectors of different models can't be compared,
# so switching models starts with an empty collection instead of mixing them
collection = client.get_or_create_collection(
    name="documents-" + re.sub(r"[^a-zA-Z0-9._-]", "-", EMBEDDING_MODEL),
    configuration={"hnsw": {"space": "cosine"}},
)

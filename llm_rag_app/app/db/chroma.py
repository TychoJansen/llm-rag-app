"""ChromaDB database setup module.

Initializes a persistent ChromaDB client and creates a collection for storing document vectors.
"""

import chromadb

from llm_rag_app.app.core.config import VECTORSTORE_DIR

client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

collection = client.get_or_create_collection(name="documents")

"""ChromaDB database setup module.

Initializes the ChromaDB client and creates a collection for storing document vectors.
"""

import chromadb

client = chromadb.Client(settings=chromadb.config.Settings(persist_directory="./vectorstore"))

collection = client.get_or_create_collection(name="documents")

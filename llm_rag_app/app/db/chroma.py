import chromadb

client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory="./vectorstore"
    )
)

collection = client.get_or_create_collection(name="documents")
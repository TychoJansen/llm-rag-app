from openai import OpenAI
from llm_rag_app.app.db.chroma import collection
from llm_rag_app.app.core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def add_documents(chunks):
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )


def query(question: str):
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
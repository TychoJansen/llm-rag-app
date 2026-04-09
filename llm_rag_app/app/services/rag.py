from openai import OpenAI
from openai import RateLimitError
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or your model
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except RateLimitError:
        return (
            "⚠️ OpenAI quota exceeded.\n\n"
            "Your API key has no remaining credits.\n\n"
            "👉 To fix this:\n"
            "1. Go to https://platform.openai.com/account/billing\n"
            "2. Add a payment method or credits\n"
            "3. Check your usage at https://platform.openai.com/usage\n\n"
        )
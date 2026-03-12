"""
Project 2 - RAG QA Chatbot
Step 4: Retrieval + Claude answer generation

Retrieves top-k chunks from ChromaDB and passes them to Claude
to generate a grounded, citation-aware answer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import anthropic
from embed_and_store import create_vector_store, SAMPLE_TEXT

# Load API key from ../.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
api_key = os.getenv("ANTHROPIC_API_KEY")

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are a precise QA assistant. Answer the question using ONLY the provided "
    "source chunks. If the answer is not in the chunks, say "
    "'I don't have enough information to answer this.' "
    "Always reference which source chunk your answer comes from."
)


def query_rag(question: str, collection, chunks, top_k: int = 3) -> dict:
    """
    Retrieve top_k chunks for the question and ask Claude to answer.

    Returns:
        dict with keys: answer, chunks_used, distances, model
    """
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]

    # Build labeled context string
    context_parts = [
        f"Source Chunk {i + 1}:\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    ]
    context = "\n\n".join(context_parts)

    user_prompt = f"Source chunks:\n{context}\n\nQuestion: {question}"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return {
        "answer": message.content[0].text,
        "chunks_used": retrieved_docs,
        "distances": distances,
        "model": MODEL,
    }


# ── Test block ────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What was Dollar General's total revenue in fiscal 2021?",
    "How many new stores were opened?",
    "Who is the CEO of Dollar General?",
]

if __name__ == "__main__":
    print("=" * 60)
    print("RAG QUERY — Project 2")
    print("=" * 60)

    print("\nBuilding vector store...")
    collection, chunks = create_vector_store(SAMPLE_TEXT, source_name="dg_annual_report")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'-' * 60}")
        print(f"Q{i}: {question}")
        result = query_rag(question, collection, chunks, top_k=3)
        print(f"Answer: {result['answer']}")
        print(f"Chunks used: {len(result['chunks_used'])}")
        print(f"Distances:   {[round(d, 4) for d in result['distances']]}")

    print(f"\n{'=' * 60}")
    print("Done.")

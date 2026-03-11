"""
Project 2 - RAG QA Chatbot
Step 3: Embed and Store chunks in ChromaDB

ChromaDB uses its built-in embedding model (all-MiniLM-L6-v2) by default —
no external API key needed for this step.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from chunk_text import chunk_text


def create_vector_store(
    text: str,
    source_name: str = "document",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Chunk text, embed each chunk, and store in a local ChromaDB collection.

    Args:
        text:         Full document text to process.
        source_name:  Label stored in chunk metadata (e.g. filename).
        chunk_size:   Target character length per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        (collection, chunks)
    """
    # 1. Chunk the text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    print(f"Chunked into {len(chunks)} pieces.")

    # 2. Create persistent ChromaDB client (stored in ./chroma_db)
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    # 3. Drop existing collection to avoid duplicates, then recreate
    collection_name = source_name
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Add chunks (ChromaDB embeds automatically using its default model)
    ids = [f"chunk_{c['index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"chunk_index": c["index"], "source": source_name, "char_count": c["char_count"]}
        for c in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Stored {collection.count()} chunks in collection '{collection_name}'.")

    return collection, chunks


# ── Test block ────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Dollar General Corporation reported total revenue of $34.2 billion for fiscal year "
    "2021, representing an increase from the prior year. The company operated 18,130 stores "
    "across 47 states as of the end of fiscal 2021. Net income for the year was approximately "
    "$2.4 billion. The company opened 1,050 new stores during the fiscal year while remodeling "
    "1,750 existing locations. Dollar General's selling square footage totaled approximately "
    "156 million square feet. The company employed roughly 163,000 full-time and part-time "
    "employees. Same-store sales decreased 2.8% compared to fiscal 2020, which had benefited "
    "from pandemic-driven demand. Gross profit margin was 31.6% of net sales. The company "
    "returned $2.7 billion to shareholders through share repurchases and dividends."
)

TEST_QUESTIONS = [
    "How many stores does Dollar General have?",
    "How many employees work there?",
    "What is the CEO's name?",
]

if __name__ == "__main__":
    print("=" * 60)
    print("EMBED AND STORE — Project 2 RAG")
    print("=" * 60)

    collection, chunks = create_vector_store(
        SAMPLE_TEXT, source_name="dg_annual_report"
    )

    print("\n" + "=" * 60)
    print("QUERYING THE VECTOR STORE")
    print("=" * 60)

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        results = collection.query(query_texts=[question], n_results=2)
        docs = results["documents"][0]
        distances = results["distances"][0]
        for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
            print(f"  Result {rank} (distance: {dist:.4f}):")
            print(f"    {doc[:200].strip()}{'...' if len(doc) > 200 else ''}")

    print("\n" + "=" * 60)
    print("Done.")

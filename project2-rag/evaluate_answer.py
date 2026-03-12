"""
Project 2 - RAG QA Chatbot
Step 5: Evaluate RAG answers using word overlap + LLM-as-judge

Combines Project 1's text_compare utility with a Claude judge
to score each RAG answer for accuracy, completeness, and hallucination.
"""

import os
import re
import sys

# Add repo root so we can import from project1-eval-toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project1-eval-toolkit"))
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import anthropic
from text_compare import compare_texts

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
api_key = os.getenv("ANTHROPIC_API_KEY")

MODEL = "claude-sonnet-4-20250514"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA evaluator. Evaluate the answer against the provided source chunks. "
    "Score each criterion from 1-5. Respond in this exact format:\n"
    "ACCURACY: [score] - [one line explanation]\n"
    "COMPLETENESS: [score] - [one line explanation]\n"
    "HALLUCINATION: [score] - [one line explanation, where 5 = no hallucination, 1 = major hallucination]\n"
    "OVERALL: [score] - [one line summary]"
)


def _parse_score(label: str, text: str) -> int:
    """Extract the integer score after a label like 'ACCURACY: 4 -'."""
    match = re.search(rf"{label}:\s*(\d)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def evaluate_rag_answer(question: str, answer: str, source_chunks: list, api_key: str = api_key) -> dict:
    """
    Evaluate a RAG-generated answer against its source chunks.

    Args:
        question:      The original user question.
        answer:        The answer produced by the RAG pipeline.
        source_chunks: List of retrieved chunk strings used to generate the answer.
        api_key:       Anthropic API key.

    Returns:
        dict with word_overlap_percent, accuracy_score, completeness_score,
        hallucination_score, overall_score, llm_judge_raw, verdict.
    """
    # ── a. Word overlap check ─────────────────────────────────────────────────
    combined_sources = " ".join(source_chunks)
    overlap_result = compare_texts(combined_sources, answer)
    word_overlap_pct = overlap_result["overlap_percentage"]

    # ── b. LLM-as-judge check ─────────────────────────────────────────────────
    chunks_text = "\n\n".join(
        f"Source Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(source_chunks)
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Source chunks:\n{chunks_text}\n\n"
        f"Answer being evaluated:\n{answer}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    judge_raw = message.content[0].text

    accuracy = _parse_score("ACCURACY", judge_raw)
    completeness = _parse_score("COMPLETENESS", judge_raw)
    hallucination = _parse_score("HALLUCINATION", judge_raw)
    overall = _parse_score("OVERALL", judge_raw)

    verdict = "PASS" if overall >= 3 and hallucination >= 3 else "FAIL"

    return {
        "word_overlap_percent": word_overlap_pct,
        "accuracy_score": accuracy,
        "completeness_score": completeness,
        "hallucination_score": hallucination,
        "overall_score": overall,
        "llm_judge_raw": judge_raw,
        "verdict": verdict,
    }


# ── Test block ────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What was Dollar General's total revenue?",
    "How many stores were opened and remodeled?",
    "Who is the CEO?",
]

if __name__ == "__main__":
    from embed_and_store import create_vector_store, SAMPLE_TEXT
    from rag_query import query_rag

    print("=" * 60)
    print("EVALUATE RAG ANSWERS — Project 2")
    print("=" * 60)

    print("\nBuilding vector store...")
    collection, chunks = create_vector_store(SAMPLE_TEXT, source_name="dg_annual_report")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'-' * 60}")
        print(f"Q{i}: {question}")

        rag_result = query_rag(question, collection, chunks, top_k=3)
        answer = rag_result["answer"]
        print(f"Answer: {answer}")

        eval_result = evaluate_rag_answer(question, answer, rag_result["chunks_used"])
        print(f"Word Overlap:   {eval_result['word_overlap_percent']}%")
        print(f"Accuracy:       {eval_result['accuracy_score']} / 5")
        print(f"Completeness:   {eval_result['completeness_score']} / 5")
        print(f"Hallucination:  {eval_result['hallucination_score']} / 5  (5 = none)")
        print(f"Overall:        {eval_result['overall_score']} / 5")
        print(f"Verdict:        {eval_result['verdict']}")

    print(f"\n{'=' * 60}")
    print("Done.")

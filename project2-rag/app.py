import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project1-eval-toolkit"))

import streamlit as st
import pdfplumber
import io
from dotenv import load_dotenv

# ── API key: st.secrets first, then .env ─────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
except Exception:
    api_key = os.getenv("ANTHROPIC_API_KEY")

# Patch module-level api_key BEFORE importing modules that use it at the top level
os.environ["ANTHROPIC_API_KEY"] = api_key or ""

from embed_and_store import create_vector_store, SAMPLE_TEXT
import rag_query as rq
import evaluate_answer as ea

# Patch module-level keys so functions pick up the resolved key
rq.api_key = api_key
ea.api_key = api_key

from rag_query import query_rag
from evaluate_answer import evaluate_rag_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG QA Chatbot with Auto-Evaluation",
    page_icon="🧠",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Document Upload")

    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            doc_text = "\n".join(pages)
            st.success(f"Extracted {len(pages)} pages from {uploaded_file.name}")
        else:
            doc_text = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded {len(doc_text):,} characters from {uploaded_file.name}")
    else:
        doc_text = st.text_area(
            "Or paste document text below:",
            value=SAMPLE_TEXT,
            height=280,
        )

    if st.button("Process Document", type="primary"):
        if not doc_text.strip():
            st.warning("Please provide some document text first.")
        else:
            with st.spinner("Chunking and embedding..."):
                try:
                    collection, chunks = create_vector_store(
                        doc_text, source_name="rag_doc"
                    )
                    st.session_state.collection = collection
                    st.session_state.chunks = chunks
                    st.success(f"Ready! Created {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("RAG QA Chatbot with Auto-Evaluation")
st.caption("Ask questions about your document. Answers are automatically evaluated for accuracy.")

if not api_key:
    st.error(
        "ANTHROPIC_API_KEY not found. "
        "Add it to your .env file (local) or Streamlit secrets (cloud)."
    )
    st.stop()

question = st.text_input("Your question:", placeholder="e.g. What was the total revenue?")

if st.button("Ask", type="primary"):
    if not st.session_state.collection:
        st.warning("Process a document first using the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer and evaluating..."):
            try:
                rag_result = query_rag(
                    question,
                    st.session_state.collection,
                    st.session_state.chunks,
                    top_k=3,
                )
                eval_result = evaluate_rag_answer(
                    question,
                    rag_result["answer"],
                    rag_result["chunks_used"],
                    api_key=api_key,
                )
                st.session_state.history.insert(0, {
                    "question": question,
                    "rag": rag_result,
                    "eval": eval_result,
                })
            except Exception as e:
                st.error(f"Error: {e}")

# ── Results display ───────────────────────────────────────────────────────────
def render_result(item):
    q = item["question"]
    rag = item["rag"]
    ev = item["eval"]

    st.markdown(f"**Q: {q}**")

    if ev["verdict"] == "PASS":
        st.success(rag["answer"])
    else:
        st.info(rag["answer"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Accuracy**")
        st.progress(ev["accuracy_score"] / 5)
        st.caption(f"{ev['accuracy_score']} / 5")
    with col2:
        st.markdown("**Completeness**")
        st.progress(ev["completeness_score"] / 5)
        st.caption(f"{ev['completeness_score']} / 5")
    with col3:
        st.markdown("**Hallucination**")
        st.progress(ev["hallucination_score"] / 5)
        st.caption(f"{ev['hallucination_score']} / 5  (5 = none)")

    vcol, wcol = st.columns([1, 2])
    with vcol:
        if ev["verdict"] == "PASS":
            st.success(f"Verdict: PASS")
        else:
            st.error(f"Verdict: FAIL")
    with wcol:
        st.metric("Word Overlap", f"{ev['word_overlap_percent']}%")

    with st.expander("Source Chunks Used"):
        for i, (chunk, dist) in enumerate(zip(rag["chunks_used"], rag["distances"]), 1):
            st.markdown(f"**Chunk {i}** — distance: `{dist:.4f}`")
            st.text(chunk)

    with st.expander("Full Evaluation Details"):
        st.text(ev["llm_judge_raw"])

    st.divider()


if st.session_state.history:
    for item in st.session_state.history:
        render_result(item)

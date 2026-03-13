import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project1-eval-toolkit"))

import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── API key: st.secrets → .env → error ───────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

api_key = None
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:
    pass
if not api_key:
    api_key = os.getenv("ANTHROPIC_API_KEY")

os.environ["ANTHROPIC_API_KEY"] = api_key or ""

from embed_and_store import create_vector_store, SAMPLE_TEXT
from parse_files import parse_file
import rag_query as rq
import evaluate_answer as ea

rq.api_key = api_key
ea.api_key = api_key

from rag_query import query_rag
from evaluate_answer import evaluate_rag_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifyAI — Document QA with Auto-Evaluation",
    page_icon="✅",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar branding */
.verify-logo {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a7f5a;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.verify-tagline {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0;
    margin-bottom: 1.2rem;
}
.step-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aaa;
    margin-bottom: 0.2rem;
    margin-top: 1rem;
}

/* Main header */
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1a7f5a;
    margin-bottom: 0;
}
.main-subtitle {
    font-size: 1rem;
    color: #888;
    margin-top: 0.1rem;
    margin-bottom: 1.5rem;
}

/* Answer box */
.answer-box {
    background: #f8fffe;
    border-left: 4px solid #1a7f5a;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 1rem;
    color: #222;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.answer-box-flagged {
    background: #fff8f8;
    border-left: 4px solid #c0392b;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 1rem;
    color: #222;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    margin-top: 1.2rem;
}

/* Verdict badge */
.badge-verified {
    display: inline-block;
    background: #1a7f5a;
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
}
.badge-flagged {
    display: inline-block;
    background: #c0392b;
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
}

/* History card */
.history-card {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.history-q {
    font-weight: 600;
    color: #333;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.history-preview {
    color: #666;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}

/* Sidebar footer */
.sidebar-footer {
    font-size: 0.72rem;
    color: #aaa;
    margin-top: 2rem;
    padding-top: 0.8rem;
    border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "history" not in st.session_state:
    st.session_state.history = []
if "chroma_tmpdir" not in st.session_state:
    st.session_state.chroma_tmpdir = tempfile.mkdtemp()
if "doc_source" not in st.session_state:
    st.session_state.doc_source = None
if "doc_chunks_count" not in st.session_state:
    st.session_state.doc_chunks_count = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="verify-logo">✅ VerifyAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="verify-tagline">Upload documents. Ask questions. Get verified answers.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="step-label">Step 1 — Upload your document</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF, DOCX, Excel, or TXT",
        type=["txt", "pdf", "docx", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            doc_text = parse_file(uploaded_file)
            st.success(f"Extracted {len(doc_text):,} characters from **{uploaded_file.name}**")
        except ValueError as e:
            st.error(str(e))
            doc_text = ""
    else:
        st.markdown('<p class="step-label">Or paste text directly</p>', unsafe_allow_html=True)
        doc_text = st.text_area(
            "Paste document text",
            value=SAMPLE_TEXT,
            height=220,
            label_visibility="collapsed",
        )

    st.markdown('<p class="step-label">Step 2 — Process</p>', unsafe_allow_html=True)
    if st.button("Process Document", type="primary", use_container_width=True):
        if not doc_text.strip():
            st.warning("Please provide some document text first.")
        else:
            with st.spinner("Chunking and embedding…"):
                try:
                    collection, chunks = create_vector_store(
                        doc_text,
                        source_name="rag_doc",
                        db_path=st.session_state.chroma_tmpdir,
                    )
                    st.session_state.collection = collection
                    st.session_state.chunks = chunks
                    st.session_state.doc_source = (
                        uploaded_file.name if uploaded_file else "Pasted text"
                    )
                    st.session_state.doc_chunks_count = len(chunks)
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    if st.session_state.collection:
        st.success(
            f"✅ Ready\n\n"
            f"**Source:** {st.session_state.doc_source}\n\n"
            f"**Chunks:** {st.session_state.doc_chunks_count}"
        )

    st.markdown(
        '<p class="sidebar-footer">Built by Shrayan | '
        '<a href="https://github.com/shrayan-bhattacharya/ai-eval-toolkit" '
        'style="color:#aaa;">AI Evaluation Toolkit</a></p>',
        unsafe_allow_html=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">VerifyAI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Ask questions about your document. '
    "Every answer is automatically verified for accuracy.</p>",
    unsafe_allow_html=True,
)

if not api_key:
    st.error("API key not found. Add it in Streamlit Cloud secrets or .env file.")
    st.stop()

if not st.session_state.collection:
    st.info("Upload and process a document in the sidebar to get started.")

question = st.text_input(
    "Your question",
    placeholder="e.g. What was the total revenue in fiscal 2021?",
    label_visibility="collapsed",
)

if st.button("Get Answer", type="primary"):
    if not st.session_state.collection:
        st.warning("Process a document first using the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and verifying answer…"):
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
def render_result(item, is_latest=False):
    q = item["question"]
    rag = item["rag"]
    ev = item["eval"]
    passed = ev["verdict"] == "PASS"

    if is_latest:
        st.markdown('<p class="section-header">Answer</p>', unsafe_allow_html=True)

    st.markdown(f"**{q}**")

    box_cls = "answer-box" if passed else "answer-box-flagged"
    answer_escaped = rag["answer"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="{box_cls}">{answer_escaped}</div>', unsafe_allow_html=True)

    if is_latest:
        st.markdown('<p class="section-header">Verification Report</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, score in [
        (c1, "Accuracy", ev["accuracy_score"]),
        (c2, "Completeness", ev["completeness_score"]),
        (c3, "Hallucination", ev["hallucination_score"]),
        (c4, "Overall", ev["overall_score"]),
    ]:
        with col:
            st.metric(label, f"{score}/5")
            st.progress(score / 5)

    badge = (
        '<span class="badge-verified">VERIFIED</span>'
        if passed
        else '<span class="badge-flagged">FLAGGED</span>'
    )
    overlap_val = ev["word_overlap_percent"]
    st.markdown(
        f"{badge} &nbsp;&nbsp; Word Overlap: **{overlap_val}%**",
        unsafe_allow_html=True,
    )

    st.markdown("")
    with st.expander("View Source Chunks"):
        for i, (chunk, dist) in enumerate(zip(rag["chunks_used"], rag["distances"]), 1):
            st.markdown(f"**Chunk {i}** — similarity distance: `{dist:.4f}`")
            st.text(chunk)
            if i < len(rag["chunks_used"]):
                st.markdown("---")

    with st.expander("Detailed Evaluation"):
        st.text(ev["llm_judge_raw"])

    st.divider()


if st.session_state.history:
    render_result(st.session_state.history[0], is_latest=True)

    if len(st.session_state.history) > 1:
        st.markdown('<p class="section-header">Previous Questions</p>', unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            ev = item["eval"]
            passed = ev["verdict"] == "PASS"
            badge_html = (
                '<span class="badge-verified" style="font-size:0.75rem;padding:0.2rem 0.7rem">VERIFIED</span>'
                if passed
                else '<span class="badge-flagged" style="font-size:0.75rem;padding:0.2rem 0.7rem">FLAGGED</span>'
            )
            preview = item["rag"]["answer"][:100] + ("…" if len(item["rag"]["answer"]) > 100 else "")
            preview_escaped = preview.replace("<", "&lt;").replace(">", "&gt;")
            scores = (
                f"Accuracy {ev['accuracy_score']}/5 · "
                f"Completeness {ev['completeness_score']}/5 · "
                f"Hallucination {ev['hallucination_score']}/5"
            )
            st.markdown(
                f'<div class="history-card">'
                f'<div class="history-q">{item["question"]}</div>'
                f'<div class="history-preview">{preview_escaped}</div>'
                f'{badge_html} &nbsp; <span style="font-size:0.78rem;color:#999">{scores}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

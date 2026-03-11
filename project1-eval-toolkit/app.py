import json
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_judge import evaluate_with_llm
from text_compare import compare_texts

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Evaluation Toolkit",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AI Eval Toolkit")
    st.markdown(
        """
        Evaluate LLM-generated text against source documents.
        Detects hallucinations, scores accuracy and completeness,
        and measures word-level overlap.

        **How it works**
        - The LLM judge (Claude Opus) compares the output to the
          source and returns structured scores.
        - The text comparator measures word overlap independently.

        ---
        """
    )
    st.markdown("**Built by Shrayan Bhattacharya**")
    st.markdown("[GitHub Repo](https://github.com/Shrayan-ai/ai-eval-toolkit)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score <= 2:
        return "🔴"
    if score == 3:
        return "🟡"
    return "🟢"


def hallucination_badge(has_hallucination: bool) -> str:
    if has_hallucination:
        return "🚨 **Hallucination Detected**"
    return "✅ **No Hallucination**"


def run_single(source: str, llm_output: str):
    with st.spinner("Running LLM judge and text comparison…"):
        llm_eval = evaluate_with_llm(source, llm_output)
        text_eval = compare_texts(source, llm_output)
    return llm_eval, text_eval


def display_single_results(llm_eval: dict, text_eval: dict):
    st.markdown("---")
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Accuracy Score",
        f"{llm_eval['accuracy_score']} / 5",
        delta=None,
    )
    col2.metric(
        "Completeness Score",
        f"{llm_eval['completeness_score']} / 5",
    )
    col3.metric(
        "Word Overlap",
        f"{text_eval['overlap_percentage']}%",
    )
    col4.metric(
        "Source Words",
        text_eval["source_words"],
    )

    st.markdown("#### Hallucination")
    if llm_eval["has_hallucination"]:
        st.error(hallucination_badge(True))
        st.write("Details:", llm_eval["hallucination_details"])
    else:
        st.success(hallucination_badge(False))

    st.markdown("#### Judge Reasoning")
    st.text(llm_eval["reasoning"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("AI Evaluation Toolkit")
st.caption("Detect hallucinations and score LLM outputs against source data")

tab_single, tab_batch = st.tabs(["Single Evaluation", "Batch Evaluation"])

# ── Tab 1: Single Evaluation ──────────────────────────────────────────────────
with tab_single:
    st.markdown("Paste a source document and an LLM-generated output to evaluate them head-to-head.")

    col_left, col_right = st.columns(2)
    with col_left:
        source_input = st.text_area(
            "Source Text (Ground Truth)",
            height=200,
            placeholder="Paste the verified source document or reference text here…",
        )
    with col_right:
        llm_input = st.text_area(
            "LLM Output to Evaluate",
            height=200,
            placeholder="Paste the AI-generated text you want to evaluate…",
        )

    if st.button("Evaluate", type="primary", use_container_width=True):
        if not source_input.strip() or not llm_input.strip():
            st.warning("Please fill in both fields before evaluating.")
        else:
            try:
                llm_eval, text_eval = run_single(source_input, llm_input)
                display_single_results(llm_eval, text_eval)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# ── Tab 2: Batch Evaluation ───────────────────────────────────────────────────
with tab_batch:
    st.markdown(
        "Run evaluation across multiple cases at once. "
        "Load the built-in demo cases or upload your own CSV."
    )

    col_demo, col_upload = st.columns([1, 2])

    with col_demo:
        load_demo = st.button("Load Demo Test Cases", use_container_width=True)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Or upload a CSV with columns: `source_text`, `llm_output`",
            type="csv",
            label_visibility="collapsed",
        )

    # Build cases list from whichever source was chosen
    cases = []

    if load_demo:
        try:
            with open("test_cases.json") as f:
                raw = json.load(f)
            cases = [
                {
                    "question": c.get("question", "—"),
                    "source_text": c["expected_answer"],
                    "llm_output": c["llm_answer"],
                }
                for c in raw
            ]
            st.session_state["batch_cases"] = cases
            st.success(f"Loaded {len(cases)} demo test cases.")
        except FileNotFoundError:
            st.error("test_cases.json not found in the project directory.")

    elif uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            required = {"source_text", "llm_output"}
            if not required.issubset(df_upload.columns):
                st.error(f"CSV must contain columns: {required}")
            else:
                cases = df_upload[["source_text", "llm_output"]].to_dict("records")
                for i, c in enumerate(cases):
                    c["question"] = f"Case {i + 1}"
                st.session_state["batch_cases"] = cases
                st.success(f"Loaded {len(cases)} cases from CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # Use persisted cases if available
    if not cases and "batch_cases" in st.session_state:
        cases = st.session_state["batch_cases"]

    if cases:
        st.markdown(f"**{len(cases)} cases ready.** Click below to run batch evaluation.")

        if st.button("Run Batch Evaluation", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Starting…")

            for i, case in enumerate(cases):
                progress.progress(
                    (i) / len(cases),
                    text=f"Evaluating case {i + 1} of {len(cases)}…",
                )
                try:
                    with st.spinner(f"Case {i + 1}: {case['question'][:60]}…"):
                        llm_eval = evaluate_with_llm(case["source_text"], case["llm_output"])
                        text_eval = compare_texts(case["source_text"], case["llm_output"])

                    results.append(
                        {
                            "Case": i + 1,
                            "Question": case["question"],
                            "Accuracy": llm_eval["accuracy_score"],
                            "Completeness": llm_eval["completeness_score"],
                            "Hallucination": llm_eval["has_hallucination"],
                            "Word Overlap %": text_eval["overlap_percentage"],
                            "Hallucination Details": llm_eval["hallucination_details"],
                            "Reasoning": llm_eval["reasoning"],
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "Case": i + 1,
                            "Question": case["question"],
                            "Accuracy": None,
                            "Completeness": None,
                            "Hallucination": None,
                            "Word Overlap %": None,
                            "Hallucination Details": f"ERROR: {e}",
                            "Reasoning": "",
                        }
                    )

            progress.progress(1.0, text="Complete!")
            st.session_state["batch_results"] = results

        # Display results if they exist
        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
            df = pd.DataFrame(results)

            # ── Summary stats ──────────────────────────────────────────────
            valid = df[df["Accuracy"].notna()]
            if not valid.empty:
                avg_acc = valid["Accuracy"].mean()
                total_hallucinations = valid["Hallucination"].sum()
                pass_rate = (valid["Accuracy"] >= 4).mean() * 100

                st.markdown("---")
                st.subheader("Summary")
                s1, s2, s3 = st.columns(3)
                s1.metric("Average Accuracy", f"{avg_acc:.1f} / 5")
                s2.metric("Hallucinations Found", int(total_hallucinations))
                s3.metric("Pass Rate (score ≥ 4)", f"{pass_rate:.0f}%")

            # ── Color-coded table ──────────────────────────────────────────
            st.markdown("---")
            st.subheader("Detailed Results")

            display_df = df[["Case", "Question", "Accuracy", "Completeness", "Hallucination", "Word Overlap %"]].copy()
            display_df["Accuracy"] = display_df["Accuracy"].apply(
                lambda v: f"{int(v)} / 5" if pd.notna(v) else "—"
            )
            display_df["Completeness"] = display_df["Completeness"].apply(
                lambda v: f"{int(v)} / 5" if pd.notna(v) else "—"
            )
            display_df["Hallucination"] = display_df["Hallucination"].apply(
                lambda v: "Yes" if v is True else ("No" if v is False else "—")
            )
            display_df["Word Overlap %"] = display_df["Word Overlap %"].apply(
                lambda v: f"{float(v):.1f}%" if pd.notna(v) else "—"
            )

            def color_accuracy_str(val):
                if val == "—":
                    return "background-color: #555555"
                score = int(val.split(" /")[0])
                if score <= 2:
                    return "background-color: #ffcccc; color: #000"
                if score == 3:
                    return "background-color: #fff3cc; color: #000"
                return "background-color: #ccffcc; color: #000"

            styled = (
                display_df.style
                .applymap(color_accuracy_str, subset=["Accuracy"])
                .set_properties(subset=["Question"], **{"min-width": "280px"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ── Expandable reasoning ───────────────────────────────────────
            with st.expander("View full reasoning for each case"):
                for _, row in df.iterrows():
                    st.markdown(f"**Case {int(row['Case'])}: {row['Question']}**")
                    if row["Reasoning"]:
                        st.write("Reasoning:")
                        st.text(row["Reasoning"])
                    if row["Hallucination Details"] and row["Hallucination Details"] != "None":
                        st.write("Hallucination details:")
                        st.text(row["Hallucination Details"])
                    st.markdown("---")

            # ── Download ───────────────────────────────────────────────────
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

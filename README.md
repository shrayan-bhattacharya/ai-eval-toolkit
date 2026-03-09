# AI Evaluation Toolkit

An automated tool for evaluating LLM outputs — scoring accuracy,
completeness, and detecting hallucinations against source documents.
Built as a practical project to explore LLM-as-judge evaluation patterns.

---

## Features

- **Single Evaluation** — Paste any source text and LLM output, get instant scores and hallucination analysis
- **Batch Evaluation** — Run evaluations across multiple cases at once using the built-in demo dataset or your own CSV
- **Hallucination Detection** — Claude Opus flags fabricated numbers, invented statistics, and claims that contradict the source
- **Color-coded Results Table** — Red (1–2), yellow (3), green (4–5) accuracy scoring at a glance
- **CSV Upload & Download** — Bring your own test cases and export all results
- **Word Overlap Scoring** — Independent text-similarity metric alongside the LLM judge

---

## Screenshots

> _Screenshots coming soon_

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM Judge | Anthropic Claude Opus (`claude-opus-4-6`) |
| Text Comparison | Custom word-overlap scorer |
| Language | Python 3.10+ |
| Config | python-dotenv |
| Data | pandas |

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/shrayan-bhattacharya/ai-eval-toolkit.git
cd ai-eval-toolkit
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

**4. Launch the app**
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
ai-eval-toolkit/
├── app.py                  # Streamlit UI
├── llm_judge.py            # Claude-powered LLM evaluator
├── text_compare.py         # Word overlap scorer
├── run_evaluation.py       # CLI batch runner
├── test_cases.json         # 8 business report test cases
├── config.json             # Evaluation config
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed)
```

---

## Built by

**Shrayan Bhattacharya** | AI Product Analyst

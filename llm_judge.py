import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


def evaluate_with_llm(source_text, llm_output):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are an AI output evaluator. You compare LLM-generated text against "
            "source documents and evaluate accuracy, completeness, and hallucination.\n\n"
            "Hallucination rules:\n"
            "- Mark has_hallucination as true ONLY when the LLM output contains a specific "
            "fact, number, or claim that directly contradicts the source OR is entirely "
            "absent from the source and cannot be inferred from it.\n"
            "- Rephrasing, paraphrasing, or reasonable rounding (e.g. 'roughly 20,000' "
            "when the source says '19,986') is NOT hallucination.\n"
            "- Minor wording differences, synonyms, or restating the same fact differently "
            "are NOT hallucination.\n"
            "- Adding well-known context that does not contradict the source is NOT "
            "hallucination.\n"
            "- Only fabricated numbers, invented statistics, or claims that contradict "
            "the source should be flagged.\n\n"
            "Always respond with valid JSON only, no markdown or extra text."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Source document:\n{source_text}\n\n"
                    f"LLM output:\n{llm_output}\n\n"
                    "Evaluate the LLM output against the source document and respond "
                    "with a JSON object containing exactly these keys:\n"
                    "- accuracy_score (integer 1-5)\n"
                    "- completeness_score (integer 1-5)\n"
                    "- has_hallucination (boolean true/false)\n"
                    "- hallucination_details (string explaining what was hallucinated, or \"None\")\n"
                    "- reasoning (string explaining the scores)"
                )
            }
        ]
    )
    text_block = next(b for b in response.content if b.type == "text")
    text = text_block.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


source = (
    "Dollar General reported revenue of $38.7 billion in FY2023, "
    "operating 19,986 stores across 47 states"
)
llm_output = (
    "Dollar General's revenue reached approximately $42 billion in FY2023. "
    "The company operates around 20,000 stores across all 50 states, "
    "making it the largest discount retailer in America."
)

print("Running LLM evaluation...\n")
result = evaluate_with_llm(source, llm_output)

print(f"Accuracy Score:       {result['accuracy_score']} / 5")
print(f"Completeness Score:   {result['completeness_score']} / 5")
print(f"Has Hallucination:    {result['has_hallucination']}")
print(f"Hallucination Details: {result['hallucination_details']}")
print(f"Reasoning:            {result['reasoning']}")

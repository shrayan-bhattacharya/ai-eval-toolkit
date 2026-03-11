import json
from llm_judge import evaluate_with_llm
from text_compare import compare_texts

with open("config.json") as f:
    config = json.load(f)

with open("test_cases.json") as f:
    test_cases = json.load(f)

print(f"Loaded config: model={config['model_name']}, criteria={config['eval_criteria']}")
print(f"Loaded {len(test_cases)} test cases\n")

results = []

for i, case in enumerate(test_cases, 1):
    print(f"Evaluating case {i}/{len(test_cases)}: {case['question'][:60]}...")
    try:
        llm_eval = evaluate_with_llm(case["expected_answer"], case["llm_answer"])
        text_eval = compare_texts(case["expected_answer"], case["llm_answer"])

        result = {
            "case_id": i,
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "llm_answer": case["llm_answer"],
            "llm_judge": llm_eval,
            "text_similarity": text_eval,
        }
        results.append(result)
        print(f"  accuracy={llm_eval['accuracy_score']}/5  overlap={text_eval['overlap_percentage']}%  hallucination={llm_eval['has_hallucination']}")
    except Exception as e:
        print(f"  WARNING: case {i} failed — {e}")

print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")

total = len(results)
if total == 0:
    print("No cases evaluated successfully.")
else:
    avg_accuracy = sum(r["llm_judge"]["accuracy_score"] for r in results) / total
    hallucinations_found = sum(1 for r in results if r["llm_judge"]["has_hallucination"])
    worst = min(results, key=lambda r: r["llm_judge"]["accuracy_score"])

    print(f"Total cases evaluated : {total}")
    print(f"Average accuracy score: {avg_accuracy:.1f} / 5")
    print(f"Hallucinations found  : {hallucinations_found}")
    print(f"Worst performing case : Case {worst['case_id']} — \"{worst['question']}\"")
    print(f"  (accuracy={worst['llm_judge']['accuracy_score']}/5)")

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to evaluation_results.json")

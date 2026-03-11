import json

test_cases = [
    {
        "question": "What year was the Eiffel Tower built?",
        "expected_answer": "The Eiffel Tower was completed in 1889.",
        "llm_answer": "The Eiffel Tower was completed in 1887.",
        "accuracy_score": 2,
        "has_hallucination": True
    },
    {
        "question": "What is the capital of Australia?",
        "expected_answer": "The capital of Australia is Canberra.",
        "llm_answer": "The capital of Australia is Sydney.",
        "accuracy_score": 1,
        "has_hallucination": True
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "expected_answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "llm_answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "accuracy_score": 5,
        "has_hallucination": False
    },
    {
        "question": "Who wrote the novel '1984'?",
        "expected_answer": "1984 was written by George Orwell.",
        "llm_answer": "1984 was written by George Orwell, published in 1949.",
        "accuracy_score": 5,
        "has_hallucination": False
    },
    {
        "question": "How many bones are in the adult human body?",
        "expected_answer": "An adult human body has 206 bones.",
        "llm_answer": "An adult human body has 206 bones.",
        "accuracy_score": 5,
        "has_hallucination": False
    }
]

with open("test_cases.json", "w") as f:
    json.dump(test_cases, f, indent=2)

with open("test_cases.json", "r") as f:
    loaded_cases = json.load(f)

total = len(loaded_cases)
hallucinations = sum(1 for tc in loaded_cases if tc["has_hallucination"])
clean = total - hallucinations

print(f"Loaded {total} test cases - {hallucinations} with hallucinations - {clean} clean")

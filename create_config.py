import json

eval_config = {
    "model_name": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1000,
    "eval_criteria": ["accuracy", "completeness", "hallucination"]
}

with open("config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

with open("config.json", "r") as f:
    loaded_config = json.load(f)

print(loaded_config)
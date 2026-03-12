def compare_texts(source, llm_output):
    source_words = set(source.lower().split())
    llm_words = set(llm_output.lower().split())
    common_words = source_words & llm_words
    overlap_percentage = round(len(common_words) / len(source_words) * 100, 1) if source_words else 0.0

    return {
        "source_words": len(source_words),
        "llm_words": len(llm_words),
        "common_words": len(common_words),
        "overlap_percentage": overlap_percentage
    }


if __name__ == "__main__":
    examples = [
        (
            "Dollar General revenue was 38.7 billion in FY2023",
            "Dollar General's revenue reached approximately 42 billion in FY2023"
        ),
        (
            "Todd Vasos is the CEO",
            "Todd Vasos serves as CEO"
        ),
        (
            "19,986 stores across 47 states",
            "The company operates roughly 20,000 retail locations nationwide"
        )
    ]

    for i, (source, llm_output) in enumerate(examples, 1):
        result = compare_texts(source, llm_output)
        print(f"Example {i}:")
        print(f"  Source:  {source}")
        print(f"  LLM:     {llm_output}")
        print(f"  Result:  {result}")
        print()

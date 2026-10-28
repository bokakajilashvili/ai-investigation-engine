import json
from pathlib import Path

from rag_engine import answer_question
from vector_store import client

DATASET_FILE = Path("evaluation_dataset.json")
RESULTS_FILE = Path("evaluation_results.json")


def load_questions():
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset["questions"]


def evaluate_questions(questions):
    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"Evaluating {index}/{len(questions)}: " f"{question}", flush=True)

        try:
            response = answer_question(question=question, limit=5)

            retrieved_sources = list(
                dict.fromkeys(source["filename"] for source in response["sources"])
            )

            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": question,
                    "expected_answer": item["expected_answer"],
                    "generated_answer": response["answer"],
                    "expected_sources": item["expected_sources"],
                    "retrieved_sources": retrieved_sources,
                    "answerable": item["answerable"],
                    "citations": response["citations"],
                    "status": "completed",
                }
            )

        except Exception as exc:
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": question,
                    "expected_answer": item["expected_answer"],
                    "error": str(exc),
                    "status": "failed",
                }
            )

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return results


if __name__ == "__main__":
    try:
        questions = load_questions()

        print(f"Loaded {len(questions)} questions")

        results = evaluate_questions(questions)

        completed = sum(result["status"] == "completed" for result in results)

        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total questions: {len(results)}")
        print(f"Completed: {completed}")
        print(f"Failed: {len(results) - completed}")
        print(f"Results saved to: {RESULTS_FILE}")

    finally:
        client.close()

import json
from pathlib import Path

RESULTS_FILE = Path("evaluation_results.json")
METRICS_FILE = Path("evaluation_metrics.json")


def recall_at_k(expected, retrieved, k=5):
    expected_set = set(expected)

    if not expected_set:
        return None

    retrieved_set = set(retrieved[:k])

    return len(expected_set & retrieved_set) / len(expected_set)


def reciprocal_rank(expected, retrieved, k=5):
    expected_set = set(expected)

    if not expected_set:
        return None

    for rank, filename in enumerate(retrieved[:k], start=1):
        if filename in expected_set:
            return 1.0 / rank

    return 0.0


def main():
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    evaluated = []

    for item in results:
        if item.get("status") != "completed":
            continue

        expected = item.get("expected_sources", [])
        retrieved = item.get("retrieved_sources", [])

        recall = recall_at_k(expected, retrieved)
        rr = reciprocal_rank(expected, retrieved)

        evaluated.append(
            {
                "id": item["id"],
                "category": item["category"],
                "recall_at_5": recall,
                "reciprocal_rank": rr,
            }
        )

    valid_recall = [
        item["recall_at_5"] for item in evaluated if item["recall_at_5"] is not None
    ]

    valid_rr = [
        item["reciprocal_rank"]
        for item in evaluated
        if item["reciprocal_rank"] is not None
    ]

    average_recall = sum(valid_recall) / len(valid_recall) if valid_recall else None

    mrr = sum(valid_rr) / len(valid_rr) if valid_rr else None

    summary = {
        "total_questions": len(results),
        "completed_questions": len(evaluated),
        "retrieval_evaluated_questions": len(valid_recall),
        "recall_at_5": average_recall,
        "mrr_at_5": mrr,
        "per_question": evaluated,
    }

    METRICS_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== RETRIEVAL METRICS ===")
    print("Evaluated questions:", len(valid_recall))
    print("Recall@5:", average_recall)
    print("MRR@5:", mrr)
    print("Saved to:", METRICS_FILE)


if __name__ == "__main__":
    main()

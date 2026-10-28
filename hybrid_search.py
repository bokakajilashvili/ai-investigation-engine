import re

from rank_bm25 import BM25Okapi

from vector_store import (
    client,
    COLLECTION_NAME,
    initialize_collection,
    generate_embeddings,
)


def tokenize(text: str) -> list[str]:
    """Tokenize English and Georgian text."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_all_chunks() -> list[dict]:
    """Read indexed chunks from Qdrant."""
    initialize_collection()

    chunks = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            chunks.append(
                {
                    "chunk_id": str(point.id),
                    "file_id": payload.get("file_id", ""),
                    "filename": payload.get("filename", ""),
                    "text": payload.get("text", ""),
                }
            )

        if offset is None:
            break

    return chunks


def reciprocal_rank_fusion(
    semantic_ranks: dict,
    keyword_ranks: dict,
    semantic_weight: float = 0.7,
    k: int = 60,
) -> dict:
    """Custom weighted RRF ranking algorithm."""

    keyword_weight = 1.0 - semantic_weight

    all_ids = set(semantic_ranks) | set(keyword_ranks)

    scores = {}

    for chunk_id in all_ids:
        score = 0.0

        if chunk_id in semantic_ranks:
            score += semantic_weight / (k + semantic_ranks[chunk_id])

        if chunk_id in keyword_ranks:
            score += keyword_weight / (k + keyword_ranks[chunk_id])

        scores[chunk_id] = score

    return scores


def hybrid_search(
    query: str,
    limit: int = 5,
    semantic_weight: float = 0.7,
) -> list[dict]:
    """Combine Qdrant semantic search and BM25 with RRF."""

    if not query.strip():
        return []

    if not 0 <= semantic_weight <= 1:
        raise ValueError("semantic_weight must be between 0 and 1")

    chunks = get_all_chunks()

    if not chunks:
        return []

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    # 1. BM25 keyword search
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    keyword_scores = {}

    if any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)

        raw_scores = bm25.get_scores(tokenize(query))

        keyword_scores = {
            chunk["chunk_id"]: float(score) for chunk, score in zip(chunks, raw_scores)
        }

    keyword_sorted = sorted(
        keyword_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    keyword_ranks = {
        chunk_id: rank
        for rank, (chunk_id, score) in enumerate(keyword_sorted, start=1)
        if score > 0
    }

    # 2. Semantic search
    semantic_scores = {}
    semantic_ranks = {}

    if semantic_weight > 0:
        query_vector = generate_embeddings([query])[0]

        semantic_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=min(max(limit * 5, 20), len(chunks)),
            with_payload=False,
        ).points

        for rank, point in enumerate(semantic_points, start=1):
            chunk_id = str(point.id)
            semantic_ranks[chunk_id] = rank
            semantic_scores[chunk_id] = float(point.score)

    # 3. Custom weighted RRF fusion
    fusion_scores = reciprocal_rank_fusion(
        semantic_ranks=semantic_ranks,
        keyword_ranks=keyword_ranks,
        semantic_weight=semantic_weight,
    )

    # 4. Rank and return results
    results = []

    for chunk_id, score in fusion_scores.items():
        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            continue

        results.append(
            {
                **chunk,
                "semantic_score": round(semantic_scores.get(chunk_id, 0.0), 4),
                "keyword_score": round(keyword_scores.get(chunk_id, 0.0), 4),
                "final_score": round(score, 6),
            }
        )

    results.sort(
        key=lambda item: (
            -item["final_score"],
            item["chunk_id"],
        )
    )

    return results[:limit]

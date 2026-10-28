
from qdrant_client import QdrantClient, models
from uuid import uuid4
from embeddings import generate_embeddings

COLLECTION_NAME = "documents"

client = QdrantClient(path="./qdrant_data")


def initialize_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )


def store_chunks(chunks: list[str], filename: str, file_id: str):
    initialize_collection()

    if not chunks:
        return 0

    vectors = generate_embeddings(chunks)

    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "file_id": file_id
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)


def search_documents(query: str, limit: int = 3):
    initialize_collection()

    query_vector = generate_embeddings([query])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    return [
        {
            "text": result.payload.get("text", ""),
            "filename": result.payload.get("filename", ""),
            "score": result.score
        }
        for result in results
    ]


def delete_document_vectors(file_id: str):
    initialize_collection()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_id",
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )
        ),
        wait=True
    )

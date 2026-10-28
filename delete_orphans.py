
from qdrant_client import models

from hybrid_search import get_all_chunks
from vector_store import client, COLLECTION_NAME
from document_registry import load_registry

registry = load_registry()
chunks = get_all_chunks()

orphan_ids = [
    chunk["chunk_id"]
    for chunk in chunks
    if not chunk["file_id"]
]

registered_before = {
    chunk["chunk_id"]
    for chunk in chunks
    if chunk["file_id"] in registry
}

print("Orphan chunks found:", len(orphan_ids))
print("Registered chunks:", len(registered_before))

confirmation = input(
    "Type DELETE to remove orphan chunks: "
)

if confirmation == "DELETE" and orphan_ids:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(
            points=orphan_ids
        ),
        wait=True,
    )

    remaining = get_all_chunks()
    remaining_ids = {
        chunk["chunk_id"] for chunk in remaining
    }

    print("Remaining chunks:", len(remaining))
    print(
        "Registered chunks preserved:",
        registered_before.issubset(remaining_ids)
    )
else:
    print("Deletion cancelled.")

client.close()

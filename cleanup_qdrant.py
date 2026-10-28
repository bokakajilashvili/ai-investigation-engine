
from collections import Counter

from hybrid_search import get_all_chunks
from document_registry import load_registry
from vector_store import client

registry = load_registry()
chunks = get_all_chunks()

registered_ids = set(registry.keys())

orphaned = [
    chunk
    for chunk in chunks
    if not chunk["file_id"]
]

unknown_ids = [
    chunk
    for chunk in chunks
    if chunk["file_id"]
    and chunk["file_id"] not in registered_ids
]

registered = [
    chunk
    for chunk in chunks
    if chunk["file_id"] in registered_ids
]

print("\n=== QDRANT CLEANUP PREVIEW ===")
print("Total chunks:", len(chunks))
print("Registered chunks:", len(registered))
print("Missing file_id:", len(orphaned))
print("Unknown file_id:", len(unknown_ids))

print("\n--- Old chunks ---")

for chunk in orphaned:
    print(
        "ID:", chunk["chunk_id"],
        "| Filename:", chunk["filename"],
        "| Preview:", chunk["text"][:70]
    )

print("\nNo data has been deleted.")

client.close()

from collections import Counter

from hybrid_search import get_all_chunks

chunks = get_all_chunks()

missing_ids = [chunk for chunk in chunks if not chunk["file_id"]]

duplicates = Counter((chunk["filename"], chunk["text"]) for chunk in chunks)

duplicate_count = sum(count - 1 for count in duplicates.values() if count > 1)

print("Total chunks:", len(chunks))
print("Chunks without file_id:", len(missing_ids))
print("Duplicate chunks:", duplicate_count)
print("Unique filenames:", len(set(chunk["filename"] for chunk in chunks)))

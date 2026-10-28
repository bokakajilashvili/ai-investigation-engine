
from vector_store import store_chunks, search_documents

chunks = [
    "Python is used for backend development.",
    "FastAPI is a Python web framework.",
    "Football is a popular sport."
]

stored = store_chunks(chunks, "test.txt")

print("Stored chunks:", stored)

results = search_documents(
    "Which programming language is used for backend?"
)

for result in results:
    print("\nText:", result["text"])
    print("Source:", result["filename"])
    print("Score:", round(result["score"], 4))

from vector_store import client
client.close()
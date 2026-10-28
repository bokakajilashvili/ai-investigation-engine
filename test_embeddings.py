
from embeddings import generate_embeddings

texts = [
    "Python is used for backend development.",
    "FastAPI is a Python web framework.",
    "Football is a popular sport."
]

vectors = generate_embeddings(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("First 5 values:", vectors[0][:5])

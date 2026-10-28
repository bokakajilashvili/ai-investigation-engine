
# AI Investigation Engine

An AI-powered document investigation system built with Python, FastAPI, Qdrant, Groq, and Streamlit.

## Features

- PDF, TXT, and DOCX document ingestion
- SHA-256 duplicate detection
- Semantic search with sentence-transformers
- Hybrid search using BM25 and Reciprocal Rank Fusion
- Retrieval-Augmented Generation (RAG)
- AI-generated answers with source citations
- Document management through a Streamlit interface
- Initial retrieval evaluation with Recall@5 and MRR@5

## Tech Stack

- Python
- FastAPI
- Streamlit
- Qdrant
- Sentence Transformers
- Groq API

## Local Setup

1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Create a `.env` file containing your Groq API key.
4. Start the backend:

   `uvicorn main:app --reload`

5. Start the frontend in another terminal:

   `streamlit run app.py`

## Evaluation

Initial evaluation covers 5 questions. The current document-level Recall@5 and MRR@5 are both 1.0 on 4 answerable questions.

These are preliminary results from a small dataset, not a comprehensive accuracy benchmark.

## Current Limitations

- Evaluation dataset needs expansion.
- Large-scale ingestion and testing are pending.
- Public deployment is pending.
- Advanced investigation features require further testing.

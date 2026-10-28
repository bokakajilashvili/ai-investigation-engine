# AI Investigation Engine

An AI-powered document investigation application developed for the Upgaming Applied AI Engineer technical assignment.

**Live Demo:** https://ai-investigation-engine-boka.streamlit.app

## Overview

AI Investigation Engine allows users to upload documents, index their contents, search for relevant evidence, and ask natural-language questions using Retrieval-Augmented Generation (RAG).

The application is deployed on Streamlit Community Cloud.

## Implemented Features

- PDF, TXT, and DOCX document ingestion
- Text extraction and chunking
- SHA-256 duplicate detection
- Document listing and deletion
- Semantic search with vector embeddings
- Hybrid retrieval combining semantic similarity and BM25
- Reciprocal Rank Fusion (RRF)
- Question answering using Groq-hosted LLMs
- Evidence citations and retrieved source passages
- Handling of insufficient evidence
- Interactive Streamlit user interface
- FastAPI endpoints available in the source code

## Technology Stack

- Python
- Streamlit
- FastAPI
- Qdrant (local vector storage)
- Sentence Transformers (all-MiniLM-L6-v2)
- BM25
- Groq API
- OpenAI-compatible Python SDK

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Start the Streamlit application:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Alternatively, run the FastAPI backend:

```bash
uvicorn main:app --reload
```

FastAPI documentation is available locally at `http://127.0.0.1:8000/docs`.

## Evaluation

An initial evaluation pipeline and a small test dataset are included in the repository.

The preliminary evaluation covered five questions. Retrieval measurements on four answerable questions produced:

- Recall@5: 1.00
- MRR@5: 1.00

These preliminary results are based on a small, manually selected dataset and are not representative of large-scale performance.

## Current Limitations

- The evaluation dataset has not yet been expanded to the requested 40 questions.
- The requested 50,000-document scale has not been demonstrated.
- A comprehensive comparison of multiple retrieval configurations remains incomplete.
- The deployed application uses local Qdrant storage; data persistence across cloud restarts is not guaranteed.
- Additional production hardening and automated testing are needed.

## AI-Assisted Development

AI assistance was used for implementation guidance, debugging, architectural planning, and documentation. The application was tested locally and through its deployed interface.

## Submission

**Source Code:** https://github.com/bokakajilashvili/ai-investigation-engine

**Live Application:** https://ai-investigation-engine-boka.streamlit.app

**Status:** Functional prototype demonstrating the core document investigation and RAG workflow.

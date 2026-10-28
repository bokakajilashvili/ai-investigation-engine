from pathlib import Path
from uuid import uuid4, UUID

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from document_parser import extract_text
from chunking import split_text

from vector_store import (
    store_chunks,
    search_documents,
    delete_document_vectors,
)

from document_registry import (
    calculate_hash,
    find_duplicate,
    register_document,
    remove_document,
    load_registry,
)
from hybrid_search import hybrid_search

from rag_engine import answer_question

app = FastAPI(
    title="AI Investigation Engine",
    description="Upgaming Applied AI Engineer Task",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


# -------------------------
# HOME
# -------------------------


@app.get("/")
def home():
    return {
        "message": "AI Investigation Engine is running!",
        "status": "success",
    }


# -------------------------
# HEALTH CHECK
# -------------------------


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# -------------------------
# DOCUMENT UPLOAD
# -------------------------


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and DOCX files are allowed",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 10 MB limit",
        )

    file_hash = calculate_hash(content)
    duplicate = find_duplicate(file_hash)

    if duplicate:
        return {
            "status": "duplicate",
            "message": "This document already exists",
            "existing_file_id": duplicate["file_id"],
            "existing_filename": duplicate["filename"],
        }

    file_id = str(uuid4())
    destination = UPLOAD_DIR / f"{file_id}{extension}"

    destination.write_bytes(content)

    try:
        extracted_text = extract_text(str(destination))
        chunks = split_text(extracted_text)

        if not chunks:
            raise ValueError("Document contains no extractable text")

        stored_count = store_chunks(
            chunks,
            filename,
            file_id,
        )

        register_document(
            file_id,
            filename,
            file_hash,
        )

    except Exception as exc:
        # Attempt to clean up partially indexed vectors.
        try:
            delete_document_vectors(file_id)
        except Exception:
            pass

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=("Document processing failed: " f"{type(exc).__name__}"),
        )

    return {
        "file_id": file_id,
        "filename": filename,
        "size_bytes": len(content),
        "status": "processed",
        "text_length": len(extracted_text),
        "text_preview": extracted_text[:500],
        "chunk_count": len(chunks),
        "chunks_preview": chunks[:3],
        "indexed_chunks": stored_count,
    }


# -------------------------
# SEARCH REQUEST MODEL
# -------------------------


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


# -------------------------
# SEMANTIC SEARCH
# -------------------------


@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty",
        )

    if not 1 <= request.limit <= 20:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20",
        )

    results = search_documents(
        query=request.query,
        limit=request.limit,
    )

    return {
        "query": request.query,
        "results": results,
        "total_results": len(results),
    }


# -------------------------
# LIST DOCUMENTS
# -------------------------


@app.get("/documents")
def list_documents():
    registry = load_registry()

    documents = [
        {
            "file_id": file_id,
            "filename": document["filename"],
            "hash": document["hash"],
        }
        for file_id, document in registry.items()
    ]

    return {
        "documents": documents,
        "total": len(documents),
    }


# -------------------------
# DELETE DOCUMENT
# -------------------------


@app.delete("/documents/{file_id}")
def delete_document(file_id: str):
    try:
        UUID(file_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID",
        )

    registry = load_registry()

    if file_id not in registry:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    try:
        # Remove vectors from Qdrant.
        delete_document_vectors(file_id)

        # Remove document from registry.
        removed = remove_document(file_id)

        if not removed:
            raise RuntimeError("Could not remove document from registry")

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document",
        )

    # Remove the original uploaded file.
    for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
        file_path.unlink(missing_ok=True)

    return {
        "status": "deleted",
        "file_id": file_id,
        "message": "Document deleted successfully",
    }


# -------------------------
# HYBRID SEARCH
# -------------------------


class HybridSearchRequest(BaseModel):
    query: str
    limit: int = 5
    semantic_weight: float = 0.7


@app.post("/search/hybrid")
def search_hybrid(request: HybridSearchRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty",
        )

    if not 1 <= request.limit <= 20:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20",
        )

    if not 0 <= request.semantic_weight <= 1:
        raise HTTPException(
            status_code=400,
            detail="semantic_weight must be between 0 and 1",
        )

    results = hybrid_search(
        query=request.query,
        limit=request.limit,
        semantic_weight=request.semantic_weight,
    )

    return {
        "query": request.query,
        "search_type": "hybrid",
        "semantic_weight": request.semantic_weight,
        "keyword_weight": round(1 - request.semantic_weight, 2),
        "total_results": len(results),
        "results": results,
    }


class AskRequest(BaseModel):
    question: str
    limit: int = 5


@app.post("/ask")
def ask_ai(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not 1 <= request.limit <= 20:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 20")

    try:
        result = answer_question(question=request.question, limit=request.limit)
        return result

    except Exception as exc:
        print(f"RAG error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )

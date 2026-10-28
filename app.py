import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AI Investigation Engine",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------

load_dotenv()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if "GROQ_MODEL" in st.secrets:
        os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]

except FileNotFoundError:
    pass

# -----------------------------------
# BACKEND IMPORTS
# -----------------------------------

from document_parser import extract_text
from chunking import split_text

from document_registry import (
    calculate_hash,
    find_duplicate,
    register_document,
    remove_document,
    load_registry,
)

from vector_store import (
    store_chunks,
    delete_document_vectors,
)

from hybrid_search import hybrid_search
from rag_engine import answer_question

# -----------------------------------
# CONSTANTS
# -----------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024

# -----------------------------------
# HEADER
# -----------------------------------

st.title("🔎 AI Investigation Engine")

st.caption(
    "Document Intelligence • Semantic Search • "
    "Hybrid Search • Retrieval-Augmented Generation"
)

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select page",
    [
        "Ask AI",
        "Hybrid Search",
        "Documents",
        "Upload Document",
    ],
)

st.sidebar.divider()

st.sidebar.caption("Upgaming Applied AI Engineer Project")

# -----------------------------------
# ASK AI
# -----------------------------------

if page == "Ask AI":

    st.header("Ask questions about your documents")

    question = st.text_area(
        "Your question",
        placeholder="What are the main requirements?",
        height=110,
    )

    limit = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=20,
        value=5,
        key="ask_limit",
    )

    if st.button("Ask AI", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching and analyzing documents..."):

                try:
                    result = answer_question(
                        question=question,
                        limit=limit,
                    )

                    st.subheader("AI Answer")
                    st.write(result.get("answer") or "No answer was generated.")

                    citations = result.get("citations", [])

                    if citations:
                        st.subheader("Citations")

                        for citation in citations:
                            st.caption(
                                f"{citation.get('citation', '')} "
                                f"{citation.get('filename', 'Unknown')}"
                            )

                    st.subheader("Retrieved Evidence")

                    sources = result.get("sources", [])

                    if not sources:
                        st.info("No evidence found.")

                    for index, source in enumerate(sources, start=1):
                        filename = source.get("filename", "Unknown")

                        with st.expander(f"[{index}] {filename}"):
                            st.write(source.get("text", ""))

                            st.caption("Chunk ID: " + str(source.get("chunk_id", "")))

                except Exception as exc:
                    st.error("AI request failed: " f"{type(exc).__name__}: {exc}")

# -----------------------------------
# HYBRID SEARCH
# -----------------------------------

elif page == "Hybrid Search":

    st.header("Hybrid Document Search")

    st.caption(
        "Search using semantic similarity and "
        "BM25 keyword matching with reciprocal rank fusion."
    )

    query = st.text_input(
        "Search query",
        placeholder="Enter keywords or a question",
    )

    limit = st.slider(
        "Maximum results",
        min_value=1,
        max_value=20,
        value=5,
        key="search_limit",
    )

    semantic_weight = st.slider(
        "Semantic search weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )

    st.caption(
        f"Semantic weight: {semantic_weight:.1f} | "
        f"Keyword weight: {1 - semantic_weight:.1f}"
    )

    if st.button("Search", type="primary"):

        if not query.strip():
            st.warning("Please enter a search query.")

        else:
            with st.spinner("Searching documents..."):

                try:
                    results = hybrid_search(
                        query=query,
                        limit=limit,
                        semantic_weight=semantic_weight,
                    )

                    st.metric(
                        "Retrieved chunks",
                        len(results),
                    )

                    if not results:
                        st.info("No matching documents found.")

                    for index, result in enumerate(results, start=1):
                        filename = result.get("filename", "Unknown")

                        with st.container(border=True):

                            st.subheader(f"{index}. {filename}")

                            st.write(result.get("text", ""))

                            st.caption("Chunk ID: " + str(result.get("chunk_id", "")))

                            if "score" in result:
                                st.caption(f"Score: {result['score']}")

                except Exception as exc:
                    st.error("Search failed: " f"{type(exc).__name__}: {exc}")

# -----------------------------------
# DOCUMENT LIST
# -----------------------------------

elif page == "Documents":

    st.header("Indexed Documents")

    if st.button("Refresh documents"):
        st.rerun()

    try:
        registry = load_registry()

        st.metric(
            "Total documents",
            len(registry),
        )

        if not registry:
            st.info("No documents uploaded yet.")

        for file_id, document in registry.items():

            with st.container(border=True):

                st.write(f"**{document['filename']}**")

                st.caption(f"Document ID: {file_id}")

                if st.button(
                    "Delete",
                    key=f"delete_{file_id}",
                ):

                    try:
                        delete_document_vectors(file_id)

                        removed = remove_document(file_id)

                        if not removed:
                            raise RuntimeError("Document could not be removed")

                        for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
                            file_path.unlink(missing_ok=True)

                        st.success("Document deleted successfully.")

                        st.rerun()

                    except Exception as exc:
                        st.error("Delete failed: " f"{type(exc).__name__}: {exc}")

    except Exception as exc:
        st.error("Could not load documents: " f"{type(exc).__name__}: {exc}")

# -----------------------------------
# DOCUMENT UPLOAD
# -----------------------------------

elif page == "Upload Document":

    st.header("Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT or DOCX file",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_file is not None:

        st.write(f"Selected: {uploaded_file.name}")

        if st.button(
            "Upload and Index",
            type="primary",
        ):

            filename = Path(uploaded_file.name).name

            extension = Path(filename).suffix.lower()

            content = uploaded_file.getvalue()

            if extension not in ALLOWED_EXTENSIONS:

                st.error("Only PDF, TXT and DOCX files are allowed.")

            elif len(content) > MAX_FILE_SIZE:

                st.error("File exceeds the 10 MB limit.")

            else:

                file_hash = calculate_hash(content)

                duplicate = find_duplicate(file_hash)

                if duplicate:

                    st.warning("This document already exists: " + duplicate["filename"])

                else:

                    file_id = str(uuid4())

                    destination = UPLOAD_DIR / f"{file_id}{extension}"

                    with st.spinner("Extracting and indexing document..."):

                        try:
                            destination.write_bytes(content)

                            extracted_text = extract_text(str(destination))

                            chunks = split_text(extracted_text)

                            if not chunks:
                                raise ValueError(
                                    "Document contains no " "extractable text"
                                )

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

                            st.success("Document uploaded successfully!")

                            st.json(
                                {
                                    "filename": filename,
                                    "chunks": len(chunks),
                                    "indexed_chunks": stored_count,
                                    "file_id": file_id,
                                }
                            )

                        except Exception as exc:

                            try:
                                delete_document_vectors(file_id)
                            except Exception:
                                pass

                            destination.unlink(missing_ok=True)

                            st.error(
                                "Document processing failed: "
                                f"{type(exc).__name__}: {exc}"
                            )

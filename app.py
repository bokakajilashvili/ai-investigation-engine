import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Investigation Engine", page_icon="🔎", layout="wide")

st.title("🔎 AI Investigation Engine")
st.caption("Document Intelligence • Hybrid Search • RAG")

st.sidebar.header("Navigation")

page = st.sidebar.radio("Select page", ["Ask AI", "Documents", "Upload Document"])


def api_request(method, endpoint, **kwargs):
    try:
        response = requests.request(
            method, f"{API_URL}{endpoint}", timeout=120, **kwargs
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        detail = str(exc)
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", detail)
            except ValueError:
                pass
        return None, detail


if page == "Ask AI":
    st.header("Ask questions about your documents")

    question = st.text_area(
        "Your question", placeholder="What are the main requirements?", height=110
    )

    limit = st.slider("Number of retrieved chunks", min_value=1, max_value=20, value=5)

    if st.button("Ask AI", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and analyzing..."):
                data, error = api_request(
                    "POST", "/ask", json={"question": question, "limit": limit}
                )

            if error:
                st.error(error)
            else:
                st.subheader("AI Answer")
                st.write(data.get("answer", ""))

                st.subheader("Retrieved Evidence")

                sources = data.get("sources", [])

                if not sources:
                    st.info("No evidence found.")

                for index, source in enumerate(sources, start=1):
                    filename = source.get("filename", "Unknown")

                    with st.expander(f"[{index}] {filename}"):
                        st.write(source.get("text", ""))
                        st.caption("Chunk ID: " + source.get("chunk_id", ""))


elif page == "Documents":
    st.header("Indexed Documents")

    if st.button("Refresh documents"):
        st.rerun()

    data, error = api_request("GET", "/documents")

    if error:
        st.error(error)
    else:
        documents = data.get("documents", [])

        st.metric("Total documents", len(documents))

        if not documents:
            st.info("No documents uploaded yet.")

        for doc in documents:
            with st.container(border=True):
                st.write(f"**{doc['filename']}**")
                st.caption(f"Document ID: {doc['file_id']}")

                if st.button("Delete", key=doc["file_id"]):
                    _, delete_error = api_request(
                        "DELETE", f"/documents/{doc['file_id']}"
                    )

                    if delete_error:
                        st.error(delete_error)
                    else:
                        st.success("Document deleted.")
                        st.rerun()


elif page == "Upload Document":
    st.header("Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT or DOCX file", type=["pdf", "txt", "docx"]
    )

    if uploaded_file is not None:
        st.write(f"Selected: {uploaded_file.name}")

        if st.button("Upload and Index", type="primary"):
            with st.spinner("Extracting and indexing document..."):
                data, error = api_request(
                    "POST",
                    "/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                )

            if error:
                st.error(error)
            elif data.get("status") == "duplicate":
                st.warning("This document already exists.")
            else:
                st.success("Document uploaded successfully!")

                st.json(
                    {
                        "filename": data.get("filename"),
                        "chunks": data.get("chunk_count"),
                        "file_id": data.get("file_id"),
                    }
                )

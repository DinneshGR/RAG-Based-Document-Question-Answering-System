"""
Streamlit frontend for the RAG Document Q&A system.

Talks to the FastAPI backend over HTTP. Run the backend first
(uvicorn app.main:app), then launch this with:

    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, answer, sources}
if "retrieval_config" not in st.session_state:
    st.session_state.retrieval_config = {"mode": "mmr", "top_k": 5, "fetch_k": 20, "lambda_mult": 0.5}


def api_get(path: str, **kwargs):
    return requests.get(f"{API_BASE_URL}{path}", timeout=30, **kwargs)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE_URL}{path}", timeout=180, **kwargs)


def api_delete(path: str, **kwargs):
    return requests.delete(f"{API_BASE_URL}{path}", timeout=30, **kwargs)


# ----------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.title("📄 RAG Document Q&A")

    st.subheader("Backend status")
    try:
        health = api_get("/health").json()
        if health["status"] == "ok":
            st.success("Backend healthy — Ollama + model ready")
        else:
            st.warning("Backend degraded")
            if not health["ollama_reachable"]:
                st.error("Ollama is not reachable. Run `ollama serve`.")
            elif not health["model_available"]:
                st.error(f"Model '{health['model_name']}' not pulled. Run `ollama pull {health['model_name']}`.")
            if not health["vector_store_ready"]:
                st.error("Vector store (ChromaDB) is not ready.")
    except requests.exceptions.RequestException:
        st.error("Cannot reach the backend API. Is FastAPI running?")

    st.divider()
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    if uploaded_file is not None and st.button("Index document", use_container_width=True):
        with st.spinner("Uploading and indexing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = api_post("/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Document uploaded successfully. Indexed {data['chunks_indexed']} chunks.")
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

    st.divider()
    st.subheader("Indexed documents")
    if st.button("Refresh list", use_container_width=True):
        st.rerun()
    try:
        docs_resp = api_get("/documents")
        docs = docs_resp.json().get("documents", [])
        if not docs:
            st.caption("No documents indexed yet.")
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"**{doc['document_name']}**\n\n{doc['chunk_count']} chunks")
            if col2.button("🗑️", key=f"del_{doc['document_id']}"):
                del_resp = api_delete(f"/documents/{doc['document_id']}")
                if del_resp.status_code == 200:
                    st.success("Document deleted.")
                    st.rerun()
                else:
                    st.error(del_resp.json().get("detail", "Delete failed."))
    except requests.exceptions.RequestException:
        st.caption("Could not load document list.")

    st.divider()
    st.subheader("Retrieval configuration")
    st.session_state.retrieval_config["mode"] = st.selectbox(
        "Retrieval mode", ["mmr", "similarity"], index=0
    )
    st.session_state.retrieval_config["top_k"] = st.slider("TOP_K", 1, 15, 5)
    st.session_state.retrieval_config["fetch_k"] = st.slider("FETCH_K", 5, 50, 20)
    st.session_state.retrieval_config["lambda_mult"] = st.slider("LAMBDA_MULT", 0.0, 1.0, 0.5)

# ----------------------------- Main area ----------------------------------
st.header("Ask a question about your documents")

question = st.chat_input("Type your question here...")

if question:
    cfg = st.session_state.retrieval_config
    with st.status("Searching relevant passages...", expanded=False) as status:
        try:
            status.update(label="Searching relevant passages...")
            payload = {
                "question": question,
                "mode": cfg["mode"],
                "top_k": cfg["top_k"],
                "fetch_k": cfg["fetch_k"],
                "lambda_mult": cfg["lambda_mult"],
            }
            status.update(label="Generating answer...")
            resp = api_post("/query", json=payload)

            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": data["answer"],
                        "sources": data["sources"],
                        "passages": data["retrieved_passages"],
                    }
                )
                status.update(label="Done", state="complete")
            else:
                status.update(label="Error", state="error")
                st.error(resp.json().get("detail", "Query failed."))
        except requests.exceptions.RequestException as exc:
            status.update(label="Error", state="error")
            st.error(f"Could not reach backend: {exc}")

# Render conversation history (most recent first is confusing; keep chronological)
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            st.markdown("**Sources:**")
            for s in turn["sources"]:
                page_str = f" — Page {s['page_number']}" if s.get("page_number") else ""
                st.markdown(f"- {s['document_name']}{page_str}")
        with st.expander("Show retrieved passages"):
            for i, p in enumerate(turn["passages"], start=1):
                page_str = f", page {p['page_number']}" if p.get("page_number") else ""
                sim_str = f" (similarity: {p['similarity']})" if p.get("similarity") is not None else ""
                st.markdown(f"**[{i}] {p['document_name']}{page_str}{sim_str}**")
                st.text(p["text"])

if not st.session_state.history:
    st.info("Upload a document from the sidebar, then ask a question to get started.")

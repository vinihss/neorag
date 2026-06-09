import os
import httpx
import streamlit as st
from pathlib import Path

API_URL = os.getenv("API_URL", "http://app:8000")


def ask(question: str, session_id: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_URL}/ask",
            json={"question": question, "session_id": session_id},
        )
        resp.raise_for_status()
        return resp.json()


def upload_file(file_path: str, session_id: str) -> dict:
    with open(file_path, "rb") as f:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{API_URL}/upload",
                files={"file": f},
                data={"session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json()


st.set_page_config(
    page_title="NeoRAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeoRAG")
st.caption("Multilingual RAG assistant with source citations")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.rerun()

    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md", "html", "docx"],
    )
    if uploaded_file is not None:
        tmp_dir = Path("/tmp/neorag_uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded_file.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Ingesting..."):
            try:
                result = upload_file(str(tmp_path), st.session_state.session_id)
                st.success(f"Ingested {result.get('chunks', 0)} chunks")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.header("About")
    st.markdown(
        "NeoRAG uses semantic search and LLM generation "
        "to answer questions based on your documents."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s.get('source', '?')}** (score: {s.get('score', 0):.3f})")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(prompt, st.session_state.session_id)
                answer = result.get("answer", "")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            score = s.get("score", 0)
                            source = s.get("source", "?")
                            text = s.get("text", "")[:200]
                            st.markdown(f"**{source}** (score: {score:.3f})")
                            st.caption(text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )

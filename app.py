import json

import streamlit as st

from uav_research_copilot.config import CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from uav_research_copilot.rag import RAGPipeline, SUPPORTED_IMPLEMENTATIONS
from uav_research_copilot.vector_store import LocalVectorStore


@st.cache_resource
def get_pipeline(implementation: str) -> RAGPipeline:
    store = LocalVectorStore(store_dir=VECTOR_STORE_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    return RAGPipeline(store, implementation=implementation)


def get_index_status() -> dict:
    metadata_path = VECTOR_STORE_DIR / "metadata.json"
    if not metadata_path.exists():
        return {"indexed": False, "chunks": 0, "papers": 0}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    papers = {item["paper_name"] for item in metadata}
    return {"indexed": True, "chunks": len(metadata), "papers": len(papers)}


def main() -> None:
    st.title("UAV Research Copilot")
    st.caption("Ask questions over local UAV research PDFs using retrieval-augmented generation.")

    with st.sidebar:
        st.header("Retrieval Settings")
        implementation = st.selectbox(
            "RAG implementation",
            options=SUPPORTED_IMPLEMENTATIONS,
            format_func=lambda item: "LangChain-like" if item == "langchain" else "LlamaIndex-like",
            index=0,
        )
        top_k = st.slider("top_k", min_value=1, max_value=10, value=4)
        chunk_size = st.slider("chunk_size", min_value=200, max_value=1500, value=CHUNK_SIZE, step=50)
        show_prompt_comparison = st.checkbox("Show prompt comparison", value=True)

    status = get_index_status()
    total_pdfs = len(list(DATA_DIR.glob("*.pdf")))

    st.subheader("Paper / Index Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("PDF files found", total_pdfs)
    col2.metric("Indexed papers", status["papers"])
    col3.metric("Indexed chunks", status["chunks"])

    if not status["indexed"]:
        st.warning("No vector index found. Run `python ingest.py` first.")
        return

    question = st.text_input("Ask a research question")
    style = st.selectbox("Prompt style", ["precise", "structured"], index=0)

    if question:
        pipeline = get_pipeline(implementation)
        response = pipeline.answer(
            question=question,
            prompt_style=style,
            top_k=top_k,
            chunk_size=chunk_size,
        )

        st.subheader("Answer")
        st.caption(
            f"Backend: {response['implementation']} | top_k={response['top_k']} | chunk_size={response['chunk_size']}"
        )
        st.write(response["answer"])

        if show_prompt_comparison:
            st.subheader("Prompt Comparison")
            precise_response = pipeline.answer(question=question, prompt_style="precise", top_k=top_k, chunk_size=chunk_size)
            structured_response = pipeline.answer(
                question=question,
                prompt_style="structured",
                top_k=top_k,
                chunk_size=chunk_size,
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Precise Prompt**")
                st.code(precise_response["prompt_used"], language="markdown")
                st.write(precise_response["answer"])
            with right:
                st.markdown("**Structured Prompt**")
                st.code(structured_response["prompt_used"], language="markdown")
                st.write(structured_response["answer"])

        st.subheader("Source Chunks")
        for idx, chunk in enumerate(response["source_chunks"], start=1):
            with st.expander(f"{idx}. {chunk['paper_name']} (chunk {chunk['chunk_index']}, score={chunk['score']:.3f})"):
                st.write(chunk["text"])


if __name__ == "__main__":
    main()

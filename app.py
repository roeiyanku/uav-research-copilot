import json

import streamlit as st

from uav_research_copilot.config import DATA_DIR, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from uav_research_copilot.rag import RAGPipeline
from uav_research_copilot.vector_store import LocalVectorStore


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    store = LocalVectorStore(store_dir=VECTOR_STORE_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    return RAGPipeline(store)


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
        pipeline = get_pipeline()
        response = pipeline.answer(question=question, prompt_style=style)

        st.subheader("Answer")
        st.write(response["answer"])

        st.subheader("Source Chunks")
        for idx, chunk in enumerate(response["source_chunks"], start=1):
            with st.expander(f"{idx}. {chunk['paper_name']} (chunk {chunk['chunk_index']}, score={chunk['score']:.3f})"):
                st.write(chunk["text"])


if __name__ == "__main__":
    main()

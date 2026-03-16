from uav_research_copilot.chunking import chunk_documents
from uav_research_copilot.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
)
from uav_research_copilot.document_loader import load_papers_from_directory
from uav_research_copilot.vector_store import LocalVectorStore


def main() -> None:
    papers = load_papers_from_directory(DATA_DIR)
    chunks = chunk_documents(papers, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    vector_store = LocalVectorStore(store_dir=VECTOR_STORE_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    stats = vector_store.build(chunks)

    print(f"Loaded papers: {len(papers)}")
    print(f"Built chunks: {stats['num_chunks']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Vector store saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()

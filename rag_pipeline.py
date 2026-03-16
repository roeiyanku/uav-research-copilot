import argparse

from uav_research_copilot.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from uav_research_copilot.rag import RAGPipeline
from uav_research_copilot.vector_store import LocalVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to UAV Research Copilot")
    parser.add_argument("question", type=str, help="Question over the paper collection")
    parser.add_argument("--style", type=str, default="precise", choices=["precise", "structured"])
    args = parser.parse_args()

    vector_store = LocalVectorStore(store_dir=VECTOR_STORE_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    pipeline = RAGPipeline(vector_store)
    response = pipeline.answer(question=args.question, prompt_style=args.style)

    print("\nAnswer:\n")
    print(response["answer"])
    print("\nSources:\n")
    for chunk in response["source_chunks"]:
        print(f"- {chunk['paper_name']} | chunk {chunk['chunk_index']} | score={chunk['score']:.4f}")


if __name__ == "__main__":
    main()

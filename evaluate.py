from uav_research_copilot.config import DEFAULT_EVAL_OUTPUT, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from uav_research_copilot.evaluation import evaluate_prompt_styles, save_eval_results
from uav_research_copilot.rag import RAGPipeline
from uav_research_copilot.vector_store import LocalVectorStore


def main() -> None:
    vector_store = LocalVectorStore(store_dir=VECTOR_STORE_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    pipeline = RAGPipeline(vector_store)

    results = evaluate_prompt_styles(pipeline)
    save_eval_results(results, DEFAULT_EVAL_OUTPUT)

    print(f"Saved evaluation results to: {DEFAULT_EVAL_OUTPUT}")
    for row in results:
        print(f"- {row['prompt_style']}: {row['question']} -> {row['top_source_paper']}")


if __name__ == "__main__":
    main()

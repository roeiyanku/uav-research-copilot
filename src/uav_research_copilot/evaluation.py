import csv
from typing import Dict, List

from .config import PROMPT_STYLE_PRECISE, PROMPT_STYLE_STRUCTURED
from .rag import RAGPipeline


EVAL_QUESTIONS = [
    "What acoustic features are commonly used for UAV detection?",
    "What machine learning models are used for UAV sound classification?",
    "What are common limitations of acoustic UAV detection systems?",
]


def evaluate_prompt_styles(rag_pipeline: RAGPipeline, questions: List[str] = EVAL_QUESTIONS) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for question in questions:
        for style in [PROMPT_STYLE_PRECISE, PROMPT_STYLE_STRUCTURED]:
            response = rag_pipeline.answer(question=question, prompt_style=style)
            rows.append(
                {
                    "question": question,
                    "prompt_style": style,
                    "answer": response["answer"],
                    "num_sources": len(response["source_chunks"]),
                    "top_source_paper": response["source_chunks"][0]["paper_name"] if response["source_chunks"] else "",
                }
            )
    return rows


def save_eval_results(rows: List[Dict[str, object]], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question", "prompt_style", "answer", "num_sources", "top_source_paper"],
        )
        writer.writeheader()
        writer.writerows(rows)

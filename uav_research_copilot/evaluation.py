import csv
import json
from typing import Dict, List

from .config import EVAL_DATASET_PATH, PROMPT_STYLE_PRECISE, PROMPT_STYLE_STRUCTURED
from .rag import RAGPipeline


def load_eval_questions(dataset_path=EVAL_DATASET_PATH) -> List[str]:
    if not dataset_path.exists():
        return []
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def evaluate_prompt_styles(rag_pipeline: RAGPipeline, questions: List[str] | None = None) -> List[Dict[str, object]]:
    question_set = questions if questions is not None else load_eval_questions()

    rows: List[Dict[str, object]] = []
    for question in question_set:
        for style in [PROMPT_STYLE_PRECISE, PROMPT_STYLE_STRUCTURED]:
            response = rag_pipeline.answer(question=question, prompt_style=style)
            rows.append(
                {
                    "question": question,
                    "prompt_style": style,
                    "implementation": response["implementation"],
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
            fieldnames=["question", "prompt_style", "implementation", "answer", "num_sources", "top_source_paper"],
        )
        writer.writeheader()
        writer.writerows(rows)

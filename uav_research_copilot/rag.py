import re
from typing import Dict, List

from .config import TOP_K_RETRIEVAL
from .prompts import build_prompt
from .vector_store import LocalVectorStore


class RAGPipeline:
    """Retrieval-augmented QA pipeline over paper chunks."""

    def __init__(self, vector_store: LocalVectorStore) -> None:
        self.vector_store = vector_store

    def answer(self, question: str, prompt_style: str = "precise", top_k: int = TOP_K_RETRIEVAL) -> Dict[str, object]:
        source_chunks = self.vector_store.search(question=question, top_k=top_k)
        context = "\n\n".join(
            f"[{idx+1}] {chunk['paper_name']} (chunk {chunk['chunk_index']}): {chunk['text']}"
            for idx, chunk in enumerate(source_chunks)
        )
        _prompt = build_prompt(style=prompt_style, question=question, context=context)
        answer = self._synthesize_answer(question=question, chunks=source_chunks, prompt_style=prompt_style)

        return {
            "question": question,
            "prompt_style": prompt_style,
            "answer": answer,
            "source_chunks": source_chunks,
            "prompt_used": _prompt,
        }

    def _synthesize_answer(self, question: str, chunks: List[Dict[str, str]], prompt_style: str) -> str:
        if not chunks:
            return "I do not know based on the provided papers."

        keywords = self._extract_keywords(question)
        candidate_sentences: List[str] = []
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
            for sentence in sentences:
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in keywords):
                    candidate_sentences.append(sentence.strip())

        selected = candidate_sentences[:4] if candidate_sentences else [chunk["text"][:240] for chunk in chunks[:3]]

        if prompt_style == "structured":
            bullets = "\n".join(f"- {s}" for s in selected[:4])
            return (
                f"Direct answer: {' '.join(selected[:2])}\n\n"
                f"Key evidence:\n{bullets}\n\n"
                "Gaps: Evidence may be partial across the available chunks."
            )

        return " ".join(selected[:4])

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        stop_words = {
            "what",
            "which",
            "how",
            "is",
            "are",
            "the",
            "a",
            "an",
            "for",
            "of",
            "in",
            "to",
            "and",
            "on",
            "used",
        }
        words = re.findall(r"[a-zA-Z]{3,}", question.lower())
        return [w for w in words if w not in stop_words]

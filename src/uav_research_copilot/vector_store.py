import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


class LocalVectorStore:
    """Local numpy-based vector store with lightweight hashed embeddings."""

    def __init__(self, store_dir: Path, embedding_model_name: str = "local-hash-embedding") -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.npy"
        self.metadata_path = self.store_dir / "metadata.json"
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = 384

    def build(self, chunk_records: List[Dict[str, str]]) -> Dict[str, int]:
        texts = [record["text"] for record in chunk_records]
        embeddings = self._embed_texts(texts)
        normalized = self._normalize(embeddings)
        np.save(self.index_path, normalized)
        self.metadata_path.write_text(json.dumps(chunk_records, indent=2), encoding="utf-8")

        return {"num_chunks": len(chunk_records), "embedding_dim": int(normalized.shape[1])}

    def search(self, question: str, top_k: int) -> List[Dict[str, str]]:
        embeddings = np.load(self.index_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        query_embedding = self._normalize(self._embed_texts([question]))
        scores = embeddings @ query_embedding[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, str]] = []
        for idx in top_indices:
            record = metadata[int(idx)].copy()
            record["score"] = float(scores[idx])
            results.append(record)
        return results

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            if not tokens:
                continue
            for token in tokens:
                col = hash(token) % self.embedding_dim
                matrix[row, col] += 1.0
        return matrix

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms

from typing import Dict, Iterable, List


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping fixed-size chunks."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_documents(
    documents: Iterable[Dict[str, str]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, str]]:
    """Chunk all document records into chunk-level records."""
    chunk_records: List[Dict[str, str]] = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{doc['paper_id']}_{idx}",
                    "paper_id": doc["paper_id"],
                    "paper_name": doc["paper_name"],
                    "chunk_index": idx,
                    "text": chunk,
                }
            )
    return chunk_records

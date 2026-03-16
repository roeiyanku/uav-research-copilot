from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def load_pdf_text(pdf_path: Path) -> str:
    """Extract text from a single PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def load_papers_from_directory(directory: Path) -> List[Dict[str, str]]:
    """Load all PDF papers from a directory into document records."""
    records: List[Dict[str, str]] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        text = load_pdf_text(pdf_path)
        if not text:
            continue
        records.append(
            {
                "paper_id": pdf_path.stem,
                "paper_name": pdf_path.name,
                "text": text,
            }
        )
    return records

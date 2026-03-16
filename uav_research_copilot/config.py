from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "papers"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "results"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "local-hash-embedding"
TOP_K_RETRIEVAL = 4
DEFAULT_EVAL_OUTPUT = RESULTS_DIR / "eval_results.csv"

PROMPT_STYLE_PRECISE = "precise"
PROMPT_STYLE_STRUCTURED = "structured"

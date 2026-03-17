from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "papers"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "uav_eval_questions.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "local-hash-embedding"
TOP_K_RETRIEVAL = 4
DEFAULT_EVAL_OUTPUT = RESULTS_DIR / "eval_results.csv"

PROMPT_STYLE_PRECISE = "precise"
PROMPT_STYLE_STRUCTURED = "structured"

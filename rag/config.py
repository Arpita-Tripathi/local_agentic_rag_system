"""Central settings. Everything tunable lives here, not in the modules."""
import os
from dataclasses import dataclass, field
from importlib import metadata

try:  # .env is optional: local mode must work without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

VALID_MODES = ("local", "cloud")


def _pypdf_tag() -> str:
    try:
        return f"pypdf@{metadata.version('pypdf')}"
    except metadata.PackageNotFoundError:
        return "pypdf"


def _env_mode() -> str:
    mode = os.environ.get("RAG_MODE", "local").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"RAG_MODE must be one of {VALID_MODES}, got {mode!r}.")
    return mode


def _env_unit_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number between 0 and 1, got {raw!r}.") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    return value


@dataclass(frozen=True)
class Config:
    # upload limits
    max_pdf_mb: int = 50
    max_pages: int = 300

    # extraction / chunking
    min_page_chars: int = 25  # fewer chars than this => page treated as image-only
    chunk_size: int = 1000
    chunk_overlap: int = 150
    parser_version: str = field(default_factory=_pypdf_tag)

    # OCR (Tesseract) and layout parsing (Docling); both are optional installs
    ocr_langs: str = field(default_factory=lambda: os.environ.get("RAG_OCR_LANGS", "hin+eng"))
    tesseract_cmd: str = field(default_factory=lambda: os.environ.get("RAG_TESSERACT_CMD", ""))
    ocr_max_pixels: int = 40_000_000  # skip absurdly large embedded images instead of exhausting RAM
    layout: str = field(default_factory=lambda: os.environ.get("RAG_LAYOUT", "auto"))  # off | auto | on

    # retrieval + answer
    top_k: int = 5
    # Applied to chunks that carry a "score" (dense cosine similarity, 0..1; see rag/retrieval/search.py).
    # 0.20 is the best trade-off measured for paraphrase-multilingual-MiniLM-L12-v2 on the sample set, one
    # PDF at a time (README, "Choosing the score threshold"): it abstains on 5 of 5 off-topic questions and
    # keeps the gold passage for 40 of 41 answerable ones. 0.15 keeps 41 of 41 but abstains on only 3 of 5
    # off-topic; 0.30 keeps just 37 of 41. That is only 5 off-topic questions, so treat it as a starting
    # point and re-tune on real PDFs. It does not suit the English-only all-MiniLM-L6-v2 (off-topic
    # scores reach 0.66).
    # Override with RAG_SCORE_THRESHOLD.
    score_threshold: float = field(default_factory=lambda: _env_unit_float("RAG_SCORE_THRESHOLD", 0.20))
    max_retries: int = 1  # at most one query rewrite
    max_query_chars: int = 400  # cap on a rewritten search query
    max_output_tokens: int = 512

    # providers
    mode: str = field(default_factory=_env_mode)  # "local" (default) or "cloud" (needs consent)
    model_dir: str = field(default_factory=lambda: os.environ.get("RAG_MODEL_DIR", "./models"))
    litert_model: str = field(default_factory=lambda: os.environ.get("RAG_LITERT_MODEL", "gemma-4-E4B-it"))
    cloud_model: str = field(default_factory=lambda: os.environ.get("RAG_CLOUD_MODEL", ""))
    cloud_timeout_s: int = 60

    # run log: one JSON line per question so a failure can be reproduced (stays on this machine)
    log_questions: bool = field(default_factory=lambda: os.environ.get("RAG_LOG_QUESTIONS", "1") != "0")

    # paths (no hardcoded /content/... anywhere)
    cache_dir: str = field(default_factory=lambda: os.environ.get("RAG_CACHE_DIR", "./index_cache"))


CONFIG = Config()

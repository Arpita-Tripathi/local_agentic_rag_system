"""Central settings. Everything tunable lives here, not in the modules."""
import os
from dataclasses import dataclass, field

try:  # .env is optional: local mode must work without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Config:
    # upload limits
    max_pdf_mb: int = 50
    max_pages: int = 300

    # extraction / chunking
    min_page_chars: int = 25  # fewer chars than this => page treated as image-only
    chunk_size: int = 1000
    chunk_overlap: int = 150
    parser_version: str = "pypdf"

    # retrieval + answer
    top_k: int = 5
    score_threshold: float = 0.15
    max_retries: int = 1  # at most one query rewrite
    max_output_tokens: int = 512

    # providers
    mode: str = "local"  # "local" (default) or "cloud" (needs consent)
    model_dir: str = field(default_factory=lambda: os.environ.get("RAG_MODEL_DIR", "./models"))
    litert_model: str = field(default_factory=lambda: os.environ.get("RAG_LITERT_MODEL", "gemma-4-E4B-it"))
    cloud_model: str = field(default_factory=lambda: os.environ.get("RAG_CLOUD_MODEL", ""))


CONFIG = Config()

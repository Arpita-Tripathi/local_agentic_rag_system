import pytest

from tests.fakes import FakeEmbedder


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch, tmp_path_factory):
    """Tests never download or load the real embedding model, and never write into ./index_cache.

    Retrieval's own tests pass their own FakeEmbedder and cache_dir explicitly.
    """
    monkeypatch.setattr("rag.retrieval.get_embedder", lambda: FakeEmbedder())
    cache = str(tmp_path_factory.mktemp("index_cache"))
    from dataclasses import replace

    from rag.config import CONFIG

    monkeypatch.setattr("rag.retrieval.index.CONFIG", replace(CONFIG, cache_dir=cache))

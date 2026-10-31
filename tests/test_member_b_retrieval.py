"""Member B: hybrid retrieval, index persistence and invalidation. Fake embedder, no model download."""

import dataclasses
import json
import os

import pytest

from rag import search_backend
from rag.config import CONFIG
from rag.retrieval import build_search, delete_index
from rag.retrieval.index import index_dir
from rag.retrieval.search import rrf_fuse
from tests.fakes import FakeEmbedder

CHUNKS = [
    {
        "id": "c0000",
        "doc_hash": "h",
        "page": 1,
        "parser": "pypdf@6",
        "type": "text",
        "heading": "",
        "text": "The LSTM predicts the next token from the previous hidden state.",
    },
    {
        "id": "c0001",
        "doc_hash": "h",
        "page": 2,
        "parser": "pypdf@6",
        "type": "text",
        "heading": "",
        "text": "Training used ten epochs on one GPU with a batch size of 32.",
    },
    {
        "id": "c0002",
        "doc_hash": "h",
        "page": 3,
        "parser": "pypdf@6",
        "type": "text",
        "heading": "",
        "text": "यह मॉडल दस युग तक प्रशिक्षित किया गया।",
    },
]


def _search(tmp_path, chunks=CHUNKS, embedder=None, doc_hash="h"):
    return build_search(chunks, doc_hash, embedder or FakeEmbedder(), str(tmp_path))


# ---------- search ----------
def test_ranks_the_relevant_chunk_first_with_a_cosine_score(tmp_path):
    hits = _search(tmp_path)("how many epochs of training", 5)
    assert hits[0]["id"] == "c0001"
    assert 0.15 <= hits[0]["score"] <= 1 and all(0 <= h["score"] <= 1 for h in hits)
    assert hits[0]["page"] == 2 and hits[0]["text"] == CHUNKS[1]["text"]  # chunk record passes through


def test_score_stays_above_the_pipeline_threshold_for_a_good_hit(tmp_path):
    # RRF values are ~0.03; the reported score must be on the scale of CONFIG.score_threshold.
    assert _search(tmp_path)("LSTM predicts the next token", 1)[0]["score"] >= CONFIG.score_threshold


def test_unanswerable_question_returns_nothing(tmp_path):
    search = _search(tmp_path)
    assert search("zebra quantum chromodynamics", 5) == []
    assert search("", 5) == [] and search("   ", 5) == []


def test_top_k_is_respected(tmp_path):
    search = _search(tmp_path)
    assert len(search("the training epochs model token", 1)) == 1
    assert search("training", 0) == []


def test_hindi_query_finds_hindi_chunk(tmp_path):
    assert _search(tmp_path)("मॉडल प्रशिक्षित", 3)[0]["id"] == "c0002"


def test_search_does_not_mutate_the_chunks(tmp_path):
    before = json.dumps(CHUNKS, sort_keys=True)
    _search(tmp_path)("training epochs", 3)
    assert json.dumps(CHUNKS, sort_keys=True) == before and "score" not in CHUNKS[0]


def test_no_chunks_is_not_an_error(tmp_path):
    assert _search(tmp_path, chunks=[])("anything", 5) == []


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf_fuse([[3, 1, 2], [1, 3]])
    assert fused[3] == pytest.approx(1 / 61 + 1 / 62) and fused[1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 63) and fused[2] < fused[3]
    assert rrf_fuse([]) == {}


def test_keyword_only_match_is_rejected_by_the_cosine_gate(tmp_path):
    chunks = [{"id": "c0", "page": 1, "parser": "p", "text": " ".join(["filler"] * 40 + ["needle"])}]
    search = _search(tmp_path, chunks=chunks)
    assert search("needle", 1) == []  # BM25 matches, but the chunk is not really about it


# ---------- persistence ----------
def test_index_reloads_and_gives_the_same_results(tmp_path):
    first_embedder = FakeEmbedder()
    first_search = _search(tmp_path, embedder=first_embedder)
    assert first_embedder.embedded == len(CHUNKS)
    first = first_search("training epochs", 3)

    second_embedder = FakeEmbedder()
    second_search = _search(tmp_path, embedder=second_embedder)
    assert second_embedder.embedded == 0  # loaded from disk, nothing re-embedded
    assert second_search("training epochs", 3) == first


def test_index_meta_records_what_it_was_built_from(tmp_path):
    _search(tmp_path)
    meta = json.loads((tmp_path / "retrieval" / "h" / "meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_model"] == "fake-embedder" and meta["parsers"] == ["pypdf@6"]
    assert meta["chunk_size"] == CONFIG.chunk_size and meta["chunk_overlap"] == CONFIG.chunk_overlap
    assert meta["doc_hash"] == "h" and meta["n_chunks"] == 3 and len(meta["chunks_sha256"]) == 64


def test_changing_chunk_settings_invalidates_the_index(tmp_path, monkeypatch):
    _search(tmp_path)
    monkeypatch.setattr("rag.retrieval.index.CONFIG", dataclasses.replace(CONFIG, chunk_size=500))
    embedder = FakeEmbedder()
    _search(tmp_path, embedder=embedder)
    assert embedder.embedded == len(CHUNKS)


def test_changing_the_embedding_model_invalidates_the_index(tmp_path):
    _search(tmp_path)
    embedder = FakeEmbedder(name="another-model")
    _search(tmp_path, embedder=embedder)
    assert embedder.embedded == len(CHUNKS)


def test_changing_the_parser_or_text_invalidates_the_index(tmp_path):
    _search(tmp_path)
    reparsed = [{**c, "parser": "docling"} for c in CHUNKS]
    embedder = FakeEmbedder()
    _search(tmp_path, chunks=reparsed, embedder=embedder)
    assert embedder.embedded == len(CHUNKS)

    edited = [{**CHUNKS[0], "text": "Different text."}, *CHUNKS[1:]]
    embedder = FakeEmbedder()
    _search(tmp_path, chunks=edited, embedder=embedder)
    assert embedder.embedded == len(CHUNKS)


def test_corrupt_index_is_rebuilt_not_a_crash(tmp_path):
    _search(tmp_path)
    (tmp_path / "retrieval" / "h" / "embeddings.npy").write_bytes(b"not a numpy file")
    embedder = FakeEmbedder()
    search = _search(tmp_path, embedder=embedder)
    assert embedder.embedded == len(CHUNKS)
    assert search("training epochs", 1)[0]["id"] == "c0001"


def test_unwritable_cache_still_searches(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    search = build_search(CHUNKS, "h", FakeEmbedder(), str(blocker / "sub"))  # parent is a file
    assert search("training epochs", 1)[0]["id"] == "c0001"


def test_delete_index_removes_the_saved_files(tmp_path):
    _search(tmp_path)
    assert (tmp_path / "retrieval" / "h").exists()
    delete_index("h", str(tmp_path))
    assert not (tmp_path / "retrieval" / "h").exists()
    delete_index("h", str(tmp_path))  # already gone: fine


def test_doc_hash_is_never_used_as_a_path(tmp_path):
    victim = tmp_path / "keep"
    victim.mkdir()
    for bad in ("../keep", "..", "a/b", ""):
        assert index_dir(bad, str(tmp_path)) is None
        delete_index(bad, str(tmp_path))
    assert victim.exists()
    search = build_search(CHUNKS, "../escape", FakeEmbedder(), str(tmp_path))  # unsafe hash: memory only
    assert search("training", 1) and not os.path.exists(tmp_path.parent / "escape")


# ---------- the app seam ----------
def test_search_backend_picks_up_the_real_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.retrieval.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("rag.retrieval.index.CONFIG", dataclasses.replace(CONFIG, cache_dir=str(tmp_path)))
    search, name = search_backend.build_search(CHUNKS, "h")
    assert name == "hybrid (fake-embedder + BM25)" and name != search_backend.STUB_NAME
    assert search("training epochs", 1)[0]["id"] == "c0001"
    search_backend.delete_index("h")
    assert not (tmp_path / "retrieval" / "h").exists()

"""Evidence-backed translation: caching, checks, language choice, privacy of the cache."""
import json

from rag.providers import FakeProvider
from rag.translate import clear_doc_translations, target_language, translate_chunk

EN = {"id": "c0001", "doc_hash": "docA", "page": 2, "type": "text",
      "text": "Training used 10 epochs on a laptop with 16 GB of memory."}
HI = {"id": "c0002", "doc_hash": "docA", "page": 3, "type": "text",
      "text": "रामपुर में 12 वर्षामापी यंत्र लगाए गए थे।"}


def test_target_language_is_the_other_one():
    assert target_language(EN) == "hi" and target_language(HI) == "en"


def test_translation_is_generated_once_then_cached(tmp_path):
    gen = FakeProvider(["प्रशिक्षण में 16 जीबी मेमोरी वाले लैपटॉप पर 10 युग लगे।"])
    first = translate_chunk(EN, gen, cache_dir=str(tmp_path))
    second = translate_chunk(EN, gen, cache_dir=str(tmp_path))
    assert first["cached"] is False and second["cached"] is True
    assert first["translation"] == second["translation"] and first["target"] == "hi"
    assert len(gen.prompts) == 1


def test_prompt_names_both_languages_and_treats_the_passage_as_untrusted(tmp_path):
    gen = FakeProvider(["x"])
    translate_chunk(EN, gen, cache_dir=str(tmp_path))
    prompt = gen.prompts[0]
    assert "from English to Hindi (Devanagari script)" in prompt and "untrusted" in prompt
    assert "Keep every number" in prompt and EN["text"] in prompt


def test_hindi_passage_is_translated_to_english(tmp_path):
    gen = FakeProvider(["12 rain gauges were installed in Rampur."])
    out = translate_chunk(HI, gen, cache_dir=str(tmp_path))
    assert out["target"] == "en" and "from Hindi (Devanagari script) to English" in gen.prompts[0]
    assert out["check"]["level"] == "checked"


def test_a_translation_that_drops_a_number_is_flagged(tmp_path):
    out = translate_chunk(EN, FakeProvider(["प्रशिक्षण में कई युग लगे।"]), cache_dir=str(tmp_path))
    assert out["check"]["level"] == "warning" and out["check"]["missing_numbers"] == ["10", "16"]


def test_check_is_recomputed_on_a_cache_hit(tmp_path):
    gen = FakeProvider(["प्रशिक्षण में 16 जीबी पर 10 युग लगे।"])
    translate_chunk(EN, gen, cache_dir=str(tmp_path))
    assert translate_chunk(EN, gen, cache_dir=str(tmp_path))["check"]["level"] == "checked"


def test_cache_is_keyed_on_text_target_and_provider(tmp_path):
    translate_chunk(EN, FakeProvider(["one"]), cache_dir=str(tmp_path))
    edited = {**EN, "text": EN["text"] + " More."}
    assert translate_chunk(edited, FakeProvider(["two"]), cache_dir=str(tmp_path))["cached"] is False
    other_target = translate_chunk(EN, FakeProvider(["three"]), target="en", cache_dir=str(tmp_path))
    assert other_target["cached"] is False

    class Other(FakeProvider):
        name = "other"

    assert translate_chunk(EN, Other(["four"]), cache_dir=str(tmp_path))["cached"] is False


def test_corrupt_cache_entry_is_translated_again(tmp_path):
    translate_chunk(EN, FakeProvider(["good"]), cache_dir=str(tmp_path))
    for f in (tmp_path / "translations").iterdir():
        f.write_text("{broken")
    out = translate_chunk(EN, FakeProvider(["fresh"]), cache_dir=str(tmp_path))
    assert out["translation"] == "fresh" and out["cached"] is False


def test_empty_translation_is_not_cached(tmp_path):
    out = translate_chunk(EN, FakeProvider(["   "]), cache_dir=str(tmp_path))
    assert out == {"translation": "", "target": "hi", "cached": False, "check": None}
    assert not (tmp_path / "translations").exists()


def test_clearing_a_document_removes_only_its_translations(tmp_path):
    other = {**EN, "doc_hash": "docB"}
    translate_chunk(EN, FakeProvider(["a"]), cache_dir=str(tmp_path))
    translate_chunk(other, FakeProvider(["b"]), cache_dir=str(tmp_path))
    assert clear_doc_translations("docA", str(tmp_path)) == 1
    assert translate_chunk(EN, FakeProvider(["a2"]), cache_dir=str(tmp_path))["cached"] is False
    assert translate_chunk(other, FakeProvider([]), cache_dir=str(tmp_path))["cached"] is True
    assert clear_doc_translations("docA", str(tmp_path / "missing")) == 0


def test_cache_record_stores_the_document_hash_for_purging(tmp_path):
    translate_chunk(EN, FakeProvider(["x"]), cache_dir=str(tmp_path))
    record = json.loads(next((tmp_path / "translations").iterdir()).read_text(encoding="utf-8"))
    assert record["doc_hash"] == "docA" and record["chunk_id"] == "c0001"

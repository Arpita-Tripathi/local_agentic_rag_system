"""Answer-support check, language handling and cross-lingual answering."""
import pytest

from rag.answer import build_prompt
from rag.lang import detect_language, is_cross_lingual, other_language
from rag.pipeline import ask
from rag.providers import FakeProvider
from rag.runlog import log_run
from rag.support import check_support, check_translation, numbers_in

PASSAGE = "The laptop had 16 GB of memory. The study ran in March 2025 at Riverbend Data Lab in Pune."


# ---------- numbers ----------
def test_numbers_normalise_digits_separators_and_words():
    assert numbers_in("१२३ and 1,000 and 1,00,000") == {"123", "1000", "100000"}
    assert numbers_in("ten epochs, twenty-five users, दस और पाँच", read_words=True) >= {"10", "25", "5"}
    assert numbers_in("ten epochs") == set()  # words are read on the source side only
    assert numbers_in("0.50 and 2.0") == {"0.5", "2"}


def test_list_markers_are_not_numbers_to_verify():
    assert numbers_in("1. First point\n2. Second point") == set()


def test_hindi_article_ek_is_never_a_number_to_check():
    out = check_support("यह एक मॉडल है", ["The model has 3 layers."])
    assert out["level"] == "nothing_to_check"


# ---------- answers ----------
def test_supported_answer_is_verified():
    out = check_support("It had 16 GB of memory.", [PASSAGE], "How much memory did it have?")
    assert out["level"] == "verified" and out["unsupported"] == {"numbers": [], "names": []}


def test_invented_number_and_name_are_flagged():
    out = check_support("It had 32 GB and ran in Berlin at Riverbend.", [PASSAGE])
    assert out["level"] == "unverified"
    assert out["unsupported"]["numbers"] == ["32"] and out["unsupported"]["names"] == ["Berlin"]


def test_number_word_in_the_source_supports_a_digit_answer():
    assert check_support("It used 10 epochs.", ["Training used ten epochs."])["level"] == "verified"


def test_numbers_repeated_from_the_question_are_allowed():
    assert check_support("Page 3 says no.", ["Nothing here."], "What does page 3 say?")["level"] == "verified"


def test_nothing_to_check_for_plain_answers():
    assert check_support("Yes, it did.", [PASSAGE])["level"] == "nothing_to_check"


def test_sentence_initial_words_are_not_names():
    assert check_support("The lab ran it. However it was fine.", [PASSAGE])["level"] == "nothing_to_check"


def test_hindi_answer_over_english_source_checks_numbers_across_scripts():
    assert check_support("मेमोरी २ जीबी थी", ["It had 2 GB."])["level"] == "verified"
    out = check_support("मेमोरी ३ जीबी थी", ["It had 2 GB."])
    assert out["level"] == "unverified" and out["unsupported"]["numbers"] == ["3"]


def test_names_are_not_compared_across_scripts():
    out = check_support("अध्ययन रिवरबेंड लैब ने किया, 16 जीबी", [PASSAGE])
    assert out["checked"]["names"] == [] and out["level"] == "verified"


# ---------- translations ----------
def test_translation_keeping_numbers_passes():
    out = check_translation("Training used 10 epochs on 16 GB.", "प्रशिक्षण में 16 जीबी पर दस युग लगे।")
    assert out["level"] == "checked" and out["missing_numbers"] == []


def test_translation_dropping_a_number_is_flagged():
    out = check_translation("Training used 10 epochs on 16 GB.", "प्रशिक्षण में 16 जीबी लगे।")
    assert out["level"] == "warning" and out["missing_numbers"] == ["10"]


def test_translation_with_implausible_length_is_flagged():
    assert check_translation("A short sentence here.", "x" * 400)["level"] == "warning"


# ---------- language ----------
def test_language_detection():
    assert detect_language("रामपुर में कितने यंत्र लगे?") == "hi"
    assert detect_language("How many gauges?") == "en"
    assert detect_language("kitne gauges lage?") == "en"  # Hinglish is answered in English
    assert detect_language("मॉडल का accuracy क्या है") == "hi"
    assert detect_language("") == "en" and other_language("hi") == "en" and other_language("en") == "hi"


def test_cross_lingual_detection():
    assert is_cross_lingual("कितनी मेमोरी?", [PASSAGE])
    assert not is_cross_lingual("How much memory?", [PASSAGE])
    assert not is_cross_lingual("कितनी मेमोरी?", [PASSAGE, "यह हिंदी है"])


def test_prompt_asks_for_the_question_language():
    chunks = [{"id": "c0", "page": 1, "text": PASSAGE}]
    assert "Hindi (Devanagari script)" in build_prompt("कितनी मेमोरी थी?", chunks)
    assert "in English," in build_prompt("How much memory?", chunks)
    assert "translate the facts" in build_prompt("How much memory?", chunks)


# ---------- pipeline ----------
CHUNKS = [{"id": "c0000", "page": 2, "text": PASSAGE, "parser": "pypdf", "score": 0.9}]


def test_ask_attaches_support_and_language_info():
    gen = FakeProvider(["Answer: 16 GB.\nSources: [C1]"])
    out = ask("How much memory?", lambda q, k: CHUNKS, gen)
    assert out["found"] and out["support"]["level"] == "verified"
    assert out["question_lang"] == "en" and out["cross_lingual"] is False


def test_ask_flags_an_unsupported_number_but_still_answers():
    gen = FakeProvider(["Answer: 64 GB.\nSources: [C1]"])
    out = ask("How much memory?", lambda q, k: CHUNKS, gen)
    assert out["found"] and out["support"]["level"] == "unverified"
    assert out["support"]["unsupported"]["numbers"] == ["64"]


def test_hindi_question_over_english_passage_is_cross_lingual():
    gen = FakeProvider(["Answer: १६ जीबी।\nSources: [C1]"])
    out = ask("लैपटॉप में कितनी मेमोरी थी?", lambda q, k: CHUNKS, gen)
    assert out["cross_lingual"] is True and out["question_lang"] == "hi"
    assert out["support"]["level"] == "verified"
    assert "Hindi (Devanagari script)" in gen.prompts[0]


def test_not_found_result_has_no_support_info():
    out = ask("q", lambda q, k: [], FakeProvider(["rewrite"]), max_retries=1)
    assert out["found"] is False and out["support"] is None


def test_run_log_records_support_level_and_cross_lingual(tmp_path):
    import json

    gen = FakeProvider(["Answer: 16 GB.\nSources: [C1]"])
    out = ask("How much memory?", lambda q, k: CHUNKS, gen)
    log_run("How much memory?", out, "h", cache_dir=str(tmp_path))
    rec = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8"))
    assert rec["support"] == "verified" and rec["cross_lingual"] is False


@pytest.mark.parametrize("answer", ["Answer: 16 GB.\nSources: [C1]", "Answer: Pune.\nSources: [C1]"])
def test_support_never_changes_the_cited_pages(answer):
    out = ask("Where?", lambda q, k: CHUNKS, FakeProvider([answer]))
    assert [c["page"] for c in out["citations"]] == [2]


# ---------- regression: "एक" / "one" are not numbers a translation must keep ----------
def test_hindi_article_ek_in_the_source_does_not_demand_a_number_in_the_translation():
    source = "1. परिचय यह रिपोर्ट एक काल्पनिक गाँव रामपुर के बारे में है। आँकड़े वर्ष 2024 में एकत्र किए गए।"
    translation = "1. Introduction This report concerns a fictional village, Rampur. Data: 2024."
    out = check_translation(source, translation)
    assert out["level"] == "checked" and out["missing_numbers"] == []


def test_english_one_in_the_source_does_not_demand_a_number_in_the_translation():
    out = check_translation("One of the 31 failures was a scan.", "31 विफलताओं में से एक स्कैन थी।")
    assert out["level"] == "checked"


def test_ek_and_one_can_still_satisfy_a_number_that_is_looked_for():
    assert check_support("1 file.", ["There was one file."])["level"] == "verified"
    assert "1" in numbers_in("एक गाँव", read_words=True)
    assert "1" not in numbers_in("एक गाँव", read_words=True, include_ambiguous=False)


def test_real_numbers_are_still_required_in_a_translation():
    out = check_translation("तीन महीने और 12 यंत्र", "three months")
    assert out["level"] == "warning" and out["missing_numbers"] == ["12"]


# ---------- regression: Hindi words with vowel signs must stay whole ----------
def test_hindi_number_words_with_vowel_signs_are_read():
    assert numbers_in("पाँच यंत्र और सात, तीन महीने", read_words=True) == {"3", "5", "7"}
    assert numbers_in("यंत्र पाँच।", read_words=True) == {"5"}  # a sentence-final danda does not join the word
    assert numbers_in("छः और छह", read_words=True) == {"6"}


def test_a_translation_may_spell_numbers_as_hindi_words():
    out = check_translation("There were 5 gauges and 3 months.", "पाँच यंत्र थे और तीन महीने।")
    assert out["level"] == "checked" and out["missing_numbers"] == []


def test_devanagari_digits_are_still_digits_not_word_characters():
    assert numbers_in("२०२४ में १२ यंत्र") == {"2024", "12"}

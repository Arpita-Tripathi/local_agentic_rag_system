"""Per-page extraction quality, including legacy Hindi fonts (Kruti Dev and similar)."""
import pytest
from pypdf import PdfReader

from rag.ingest import ingest_pdf
from rag.ingest.extract import extract_pages
from rag.ingest.quality import assess_page, ocr_quality, orphan_ratio, page_fonts, summarize

GOOD = "The laptop had 16 GB of memory and the study ran in March 2025 at a small lab."
LEGACY_TEXT = 'fganh Hkk"kk esa ;g ,d mnkgj.k gS A vkt ekSle vPNk gS A'  # what Kruti Dev text extracts as


def _pdf(path, pages):
    """A valid PDF with one page per (font_name, text). The font is not embedded: only its name matters."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for i, (font, text) in enumerate(pages):
        page_id, content_id, font_id = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET"
        objs += [
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
            f"<< /Type /Font /Subtype /Type1 /BaseFont /{font} >>",
        ]
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return str(path)


# ---------- scoring ----------
def test_normal_text_scores_good():
    q = assess_page(GOOD, ["Helvetica"])
    assert q["level"] == "good" and q["score"] == 1.0 and q["reasons"] == [] and not q["legacy_font"]


@pytest.mark.parametrize(
    "font", ["KrutiDev010", "ABCDEF+Kruti-Dev-011", "DevLys 010", "Chanakya", "Shree-Dev-0714"])
def test_legacy_hindi_font_names_are_recognised(font):
    q = assess_page(LEGACY_TEXT, ["Helvetica", font])
    assert q["legacy_font"] and q["level"] == "poor" and q["score"] < 0.1
    assert "legacy Hindi font" in q["reasons"][0]


@pytest.mark.parametrize("font", ["Mangal", "NotoSansDevanagari", "Arial-BoldMT", "Aparajita"])
def test_unicode_fonts_are_not_flagged(font):
    assert not assess_page(GOOD, [font])["legacy_font"]


def test_empty_page_is_none():
    assert assess_page("   ", []) == {"score": 0.0, "level": "none", "legacy_font": False,
                                      "reasons": ["no text layer on this page"]}


def test_unreadable_symbols_lower_the_score():
    q = assess_page("ok text " * 5 + "�" * 12, [])
    assert q["score"] < 0.85 and any("unreadable symbols" in r for r in q["reasons"])


def test_words_starting_with_a_vowel_sign_are_a_warning_sign():
    broken = "ा कि ो म ि क ु र " + "सही शब्द है " * 2
    assert orphan_ratio(broken) > 0.25
    q = assess_page(broken, [])
    assert q["level"] in ("fair", "poor") and any("vowel sign" in r for r in q["reasons"])
    assert orphan_ratio("यह एक सही हिंदी वाक्य है जिसमें कोई समस्या नहीं") == 0.0


def test_correct_hindi_scores_good():
    text = "रामपुर में पिछले साल बारह वर्षामापी यंत्र लगाए गए थे और उनके आँकड़े हर सप्ताह दर्ज हुए।"
    assert assess_page(text, ["Mangal"])["level"] == "good"


def test_very_short_text_is_penalised_but_not_poor():
    q = assess_page("Page 3", [])
    assert q["level"] == "good" and q["score"] == 0.85 and "very little text" in q["reasons"][0]


def test_summary_counts_levels():
    qs = [assess_page(GOOD, []), assess_page("", []), ocr_quality(), assess_page(LEGACY_TEXT, ["Chanakya"])]
    assert summarize(qs) == {"good": 1, "fair": 0, "poor": 1, "none": 1, "ocr": 1}


# ---------- real PDFs ----------
def test_page_fonts_reads_the_declared_font_names(tmp_path):
    path = _pdf(tmp_path / "f.pdf", [("KrutiDev010", LEGACY_TEXT)])
    assert page_fonts(PdfReader(path).pages[0]) == ["KrutiDev010"]


def test_legacy_font_page_is_discarded_and_sent_to_ocr(tmp_path):
    path = _pdf(tmp_path / "m.pdf", [("Helvetica", GOOD), ("KrutiDev010", LEGACY_TEXT)])
    pages = extract_pages(path)
    assert pages[0]["quality"]["level"] == "good" and not pages[0]["image_only"]
    assert pages[1]["garbled"] and pages[1]["image_only"] and pages[1]["text"] == ""
    assert pages[1]["quality"]["legacy_font"]


def test_ingest_reports_quality_per_page_and_warns_about_the_old_font(tmp_path):
    path = _pdf(tmp_path / "m.pdf", [("Helvetica", GOOD), ("KrutiDev010", LEGACY_TEXT)])
    out = ingest_pdf(path, layout="off", ocr=False)
    assert [q["level"] for q in out["page_quality"]] == ["good", "poor"]
    assert out["quality_summary"]["good"] == 1 and out["quality_summary"]["poor"] == 1
    assert out["image_only_pages"] == [2]
    assert any("old Hindi font" in w and "Pages 2" in w for w in out["warnings"])
    assert all("fganh" not in c["text"] for c in out["chunks"])  # the gibberish was never indexed


def test_ocr_pages_are_reported_as_ocr_quality(tmp_path, monkeypatch):
    path = _pdf(tmp_path / "m.pdf", [("Helvetica", GOOD), ("KrutiDev010", LEGACY_TEXT)])

    def fake_ocr(p, pages):
        done = [{**pg, "text": "यह पृष्ठ ओसीआर से पढ़ा गया है और काफ़ी लंबा है।", "image_only": False,
                 "type": "ocr", "parser": "tesseract@5"} if pg["image_only"] else pg for pg in pages]
        return done, []

    monkeypatch.setattr("rag.ingest.ocr_pages", fake_ocr)
    out = ingest_pdf(path, layout="off")
    assert [q["level"] for q in out["page_quality"]] == ["good", "ocr"]
    assert out["image_only_pages"] == [] and {c["type"] for c in out["chunks"]} == {"text", "ocr"}


def test_a_page_of_ordinary_text_is_good_end_to_end(tmp_path):
    out = ingest_pdf(_pdf(tmp_path / "g.pdf", [("Helvetica", GOOD)]), layout="off")
    assert out["page_quality"][0]["level"] == "good" and out["warnings"] == []


# ---------- regression: NUL characters from lost Hindi vowel signs ----------
BROKEN_HINDI = "1. प\x00रचय यह \x00रपोट\x00 एक काल्पिनक गाँव रामपुर म\x00 वषा\x00 के आँकड़ों का सार प्र\x00ुत करती है।"


def test_lost_vowel_signs_are_scored_and_explained_in_plain_words():
    q = assess_page(BROKEN_HINDI, ["NirmalaUI"])
    assert q["level"] in ("fair", "poor") and not q["legacy_font"]
    assert any("lost in extraction" in r and "vowel signs" in r for r in q["reasons"])


def test_other_unreadable_symbols_keep_the_generic_wording():
    q = assess_page("plain english text " * 4 + "�" * 12, [])
    assert any("unreadable symbols" in r for r in q["reasons"])


def test_nul_characters_never_reach_the_stored_text(monkeypatch):
    class Page:
        def extract_text(self):
            return BROKEN_HINDI

        def get(self, key):
            return None

    class Reader:
        def __init__(self, path):
            self.pages = [Page()]

    monkeypatch.setattr("rag.ingest.extract.PdfReader", Reader)
    page = extract_pages("x.pdf")[0]
    assert "\x00" not in page["text"] and "�" not in page["text"]
    assert "रामपुर" in page["text"]  # the readable words are kept
    assert page["quality"]["level"] in ("fair", "poor")  # but the page is still marked as damaged
    assert not page["image_only"]


def test_ordinary_control_characters_like_newline_and_tab_are_kept(monkeypatch):
    class Page:
        def extract_text(self):
            return "Line one of the page text.\nLine two\twith a tab, long enough to count."

        def get(self, key):
            return None

    class Reader:
        def __init__(self, path):
            self.pages = [Page()]

    monkeypatch.setattr("rag.ingest.extract.PdfReader", Reader)
    assert "\n" in extract_pages("x.pdf")[0]["text"] and "\t" in extract_pages("x.pdf")[0]["text"]

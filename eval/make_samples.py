"""Regenerate the sample PDFs in data/ and data/expected_outcomes.json.

    python -m eval.make_samples

Everything here is original, invented text, so the files are safe to share and the answers are known
exactly. eval/questions.json is written against this content; `python -m eval.run_eval --check` proves
each expected page really contains its evidence.
"""

import json
import pathlib
from typing import Dict, List

from eval.pdfgen import build_pdf, build_scanned_pdf, corrupt, wrap

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# ---------- English report (4 pages) ----------
EN_PAGES = [
    [
        ("h", "1. Overview"),
        (
            "p",
            "This report describes a small study of question answering over PDF documents. The study was run "
            "by the Riverbend Data Lab in March 2025. The team evaluated three retrieval methods on 120 "
            "questions. All experiments ran on a laptop with 16 GB of memory and no GPU.",
        ),
    ],
    [
        ("h", "2. Method"),
        (
            "p",
            "Documents were split into passages of 1000 characters with an overlap of 150 characters. Each "
            "passage was embedded with a MiniLM model that produces vectors of 384 dimensions. "
            "Keyword search "
            "used the BM25 algorithm. The two rankings were merged with reciprocal rank fusion using a "
            "constant of 60. The system returns at most five passages for each question.",
        ),
    ],
    [
        ("h", "3. Results"),
        (
            "p",
            "Hybrid search reached a recall of 91 percent at five passages. Keyword search alone reached 78 "
            "percent and embedding search alone reached 85 percent. The median answer time was 4.2 seconds. "
            "The slowest question took 19 seconds. Scanned pages were the main source of errors, causing 14 "
            "of the 31 failures.",
        ),
    ],
    [
        ("h", "4. Limitations"),
        (
            "p",
            "The study used only English and Hindi documents. Tables were converted to plain text, which "
            "sometimes lost row boundaries. The authors recommend repeating the study with at least 500 "
            "questions. No private documents were used; every file was public.",
        ),
    ],
    [
        ("h", "5. Data Collection"),
        (
            "p",
            "The documents came from three public sources: government reports, university course notes and "
            "open textbooks. In total the team collected 48 files with 1350 pages. Files larger than 50 "
            "megabytes were excluded. Each file was checked by two reviewers before it was added.",
        ),
    ],
    [
        ("h", "6. Question Writing"),
        (
            "p",
            "Two reviewers wrote the 120 questions by hand. Ninety questions had an answer in the documents "
            "and thirty did not. Each answerable question lists the page that holds the evidence. Questions "
            "were written in English, Hindi and a mixture of both.",
        ),
    ],
    [
        ("h", "7. Answer Generation"),
        (
            "p",
            "A local language model with four billion parameters wrote the answers. "
            "The model was told to say "
            "not found when the passages did not contain the answer. The prompt allowed at most 512 tokens "
            "of output. A cloud model was never used.",
        ),
    ],
    [
        ("h", "8. Conclusions"),
        (
            "p",
            "Combining keyword and embedding search gave the best recall without extra cost. The system "
            "refused to answer 27 of the 30 unanswerable questions. The authors plan to test a multilingual "
            "embedding model next. The code and the questions will be released under an open licence.",
        ),
    ],
]

# ---------- Hindi report (3 pages) ----------
HI_PAGES = [
    [
        ("h", "1. परिचय"),
        (
            "p",
            "यह रिपोर्ट एक काल्पनिक गाँव रामपुर में वर्षा के आँकड़ों का सार प्रस्तुत करती है। आँकड़े वर्ष 2024 में "
            "तीन महीनों तक एकत्र किए गए। गाँव में कुल पाँच वर्षामापी यंत्र लगाए गए थे।",
        ),
    ],
    [
        ("h", "2. परिणाम"),
        (
            "p",
            "जुलाई में सबसे अधिक वर्षा हुई, कुल 320 मिलीमीटर। अगस्त में 210 मिलीमीटर वर्षा दर्ज की गई। सितंबर में "
            "केवल 95 मिलीमीटर वर्षा हुई। सबसे अधिक वर्षा एक ही दिन में 85 मिलीमीटर थी। आँकड़े गाँव के स्कूल के "
            "शिक्षक श्री मोहन लाल ने दर्ज किए।",
        ),
    ],
    [
        ("h", "3. सुझाव"),
        (
            "p",
            "किसानों को जुलाई के पहले सप्ताह में धान की बुवाई करने की सलाह दी गई। गाँव में एक छोटा तालाब बनाने "
            "का प्रस्ताव रखा गया। अगले वर्ष सात वर्षामापी यंत्र लगाने की योजना है।",
        ),
    ],
]

# ---------- Table-heavy report (2 pages). Cells are drawn in columns. ----------
TABLE_PAGES = [
    [
        ["Table 1. Retrieval quality by method"],
        [""],
        ["Method", "Recall at 5", "Median seconds"],
        ["Keyword only", "72", "0.9"],
        ["Embedding only", "81", "1.2"],
        ["Hybrid", "88", "1.4"],
    ],
    [
        ["Table 2. Failures by document type"],
        [""],
        ["Type", "Questions", "Failures"],
        ["Text PDF", "60", "7"],
        ["Table PDF", "30", "12"],
        ["Scanned PDF", "30", "19"],
        ["Total", "120", "38"],
    ],
]

# ---------- Scanned notes (image-only, needs OCR) ----------
SCANNED_PAGES = [
    [
        "Field Notes",
        "",
        "The weather station in Lakeside recorded 42 millimetres of rain on Tuesday.",
        "The lake level rose by 7 centimetres.",
        "The station manager is Priya Nair.",
    ],
    [
        "Equipment",
        "",
        "The station uses a tipping bucket gauge bought in 2019.",
        "The battery is replaced every 6 months.",
    ],
]

# ---------- Empty pages: text, blank, text, blank, text ----------
EMPTY_TEXT = [
    "The first page has a short paragraph about river levels in spring.",
    "The third page lists the sensor names used at the northern bridge.",
    "The fifth page closes the document with contact details for the data team.",
]


def _lines(blocks, width: int) -> List[str]:
    out: List[str] = []
    for kind, text in blocks:
        out += wrap(text, width)
        out.append("")
    return out[:-1]


def _wrapped(pages, width):
    return [_lines(blocks, width) for blocks in pages]


OUTCOMES: Dict[str, Dict] = {
    "sample_text_en.pdf": {
        "kind": "text",
        "pages": 8,
        "expected": "Ingests with pypdf. Every page yields text chunks under the pypdf parser; no warnings.",
    },
    "sample_text_hi.pdf": {
        "kind": "text (Hindi)",
        "pages": 3,
        "expected": "Ingests with pypdf. The text layer is Devanagari; chunks contain it unchanged.",
    },
    "sample_tables.pdf": {
        "kind": "table-heavy",
        "pages": 2,
        "expected": "Ingests with pypdf (layout off). Numbers and row labels stay in the chunk text. "
        "With layout=auto and Docling installed, table chunks are produced instead.",
    },
    "sample_scanned.pdf": {
        "kind": "scanned (image-only)",
        "pages": 2,
        "expected": "Without Tesseract: rejected with a message that says what to install. With Tesseract "
        "and eng data: both pages become type=ocr chunks.",
    },
    "sample_corrupt.pdf": {
        "kind": "corrupt",
        "pages": None,
        "expected": "Rejected with a clear 'appears to be corrupt' message. No crash.",
    },
    "sample_empty_pages.pdf": {
        "kind": "empty pages",
        "pages": 5,
        "expected": "Ingests. Pages 2 and 4 have no text and are reported as image-only; chunks come only "
        "from pages 1, 3 and 5.",
    },
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    (DATA / "sample_text_en.pdf").write_bytes(build_pdf(_wrapped(EN_PAGES, 88)))
    (DATA / "sample_text_hi.pdf").write_bytes(build_pdf(_wrapped(HI_PAGES, 55), script="devanagari"))
    (DATA / "sample_tables.pdf").write_bytes(build_pdf(TABLE_PAGES))
    build_scanned_pdf(SCANNED_PAGES, str(DATA / "sample_scanned.pdf"))
    page_texts = (EMPTY_TEXT[0], "", EMPTY_TEXT[1], "", EMPTY_TEXT[2])
    (DATA / "sample_empty_pages.pdf").write_bytes(build_pdf([[t] if t else [] for t in page_texts]))
    (DATA / "sample_corrupt.pdf").write_bytes(corrupt(build_pdf(_wrapped(EN_PAGES, 88))))
    (DATA / "expected_outcomes.json").write_text(
        json.dumps(OUTCOMES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(OUTCOMES)} sample files to {DATA}")


if __name__ == "__main__":
    main()

# Local-first PDF Q&A

Ask questions about a PDF and get an answer with page citations, or "I could not find this in the PDF."
Works in English and Hindi. Runs on your machine by default; cloud is opt-in and asks for consent.

- **Evidence first.** Every answer lists the passages it used. Citation page numbers come from stored
  metadata, never from the model. A citation the model invents is dropped, and an answer left with no valid
  citation is withheld.
- **Says "not found" instead of guessing.** A weak search result or a model that finds no support abstains.
- **Local by default.** Gemma runs on-device through LiteRT. Cloud mode sends only the retrieved excerpts,
  only after you consent, and never silently as a fallback.
- **Measured.** `python -m eval.run_eval` reproduces the numbers below. The Benchmark page in the app shows them.

## What is supported

| | Supported |
|---|---|
| Text PDFs (English, Hindi) | Yes, with `pypdf` |
| Scanned PDFs | Yes, if Tesseract with Hindi and English data is installed. Without it the app says what to install |
| Tables | Kept as plain text. Better table structure needs Docling (optional, large) |
| Corrupt, encrypted, oversize, non-PDF files | Rejected with a clear message |
| Pages with no text and no image | Reported as empty; the other pages still work |

**Not supported:** several PDFs in one question (one document at a time), handwriting, charts and figures
without text, PDFs over 50 MB or 300 pages (configurable in `rag/config.py`).

## Setup

Python 3.10 or newer. About 5 GB of disk for the local model, plus about 0.5 GB for the embedding model.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

**Local model (Gemma, about 3.7 GB):**

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('litert-community/gemma-4-E4B-it-litert-lm', allow_patterns=['gemma-4-E4B-it.litertlm'], local_dir='./models')"
```

If the download is refused, accept the model's terms on its Hugging Face page and run `huggingface-cli login`.
The first question after start-up is slow while the model loads.

**OCR (only for scanned PDFs):** install Tesseract with Hindi and English language data.
Windows: the UB Mannheim build, tick **Hindi** under *Additional language data*. Linux:
`apt install tesseract-ocr tesseract-ocr-hin`. If it is not on PATH, set `RAG_TESSERACT_CMD` in `.env`.

**Better tables (optional, heavy):** `pip install docling==2.145.0`, then set `RAG_LAYOUT=auto`.

**Cloud mode (optional):** create your own key, put it in the environment as `OPENROUTER_API_KEY`
(never in source or chat) and set `RAG_CLOUD_MODEL`. Check the provider's current limits and terms first.
Never send a sensitive PDF.

## Run

```powershell
streamlit run app.py
```

Upload a PDF, ask a question, click a citation to see its page and snippet. The Benchmark page (sidebar)
shows the measured results.

## Tests and benchmark

```powershell
pytest                                   # no model download, no network: a fake embedder is used
ruff check .
python -m eval.make_samples              # regenerate data/sample_*.pdf and data/expected_outcomes.json
python -m eval.run_eval --check          # prove every expected page in eval/questions.json holds its evidence
python -m eval.run_eval --embed-model paraphrase-multilingual-MiniLM-L12-v2 --out eval/results/multilingual.json
python -m eval.run_eval --library        # one index over every sample PDF (harder)
python -m eval.run_eval --answers local  # also run Gemma: answer support and citation accuracy
python -m eval.bench_parsers --out eval/results/parsers.json
```

The embedding model is chosen with `RAG_EMBED_MODEL` (default `paraphrase-multilingual-MiniLM-L12-v2`, about 470 MB,
downloaded on first use; this needs internet once). `RAG_SCORE_THRESHOLD` sets the score cut-off.

## Benchmark results

63 hand-written questions over 6 invented sample PDFs: English, Hindi, Hinglish (Roman-script Hindi),
code-switched, table and scanned cases; 46 answerable, 17 unanswerable. The sample text is original, so the
files are free to share. 56 of the 63 questions ran: the 7 scanned-page questions need Tesseract, which was not
installed for these runs. Retrieval only, no language model. Run on 2026-10-31 on Windows 11, Python 3.12.10,
an 18-thread Intel CPU (retrieval runs on the CPU). Each result file in `eval/results/` records its own
platform, settings and every per-question row; the Benchmark page reads them.

One index per PDF (how the app works), score threshold 0.20:

| Embedding model | Recall@1 | Recall@3 | Recall@5 | Hindi R@1 | Hinglish R@1 | Code-switched R@1 | Off-topic abstained |
|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (default) | 95.1% | 100% | 100% | 90% | 100% | 75% | 100% |
| all-MiniLM-L6-v2 | 80.5% | 100% | 100% | 90% | 80% | 50% | 60% |

With all sample PDFs in one index (`--library`, harder): multilingual Recall@1 92.7%, Recall@5 100%,
off-topic abstained 80%; all-MiniLM-L6-v2 Recall@1 78.0%, Recall@5 100%, off-topic abstained 60%. Median search
time is about 22 ms (multilingual) and 12 ms (all-MiniLM) per PDF. The saved multilingual run has one
slow outlier (p95 2.2 s, likely model warm-up), so treat p95 as noisy. The multilingual model is the default
because it ranks better on Hinglish and mixed-script questions, and because its scores separate off-topic
questions from answerable ones (below).

### Choosing the score threshold

The app drops passages whose cosine similarity is below `RAG_SCORE_THRESHOLD` (default 0.20) before the model is
asked. Measured with the multilingual model:

| Threshold | One index per PDF: gold passage kept | Off-topic abstained | One index for all PDFs: kept | Off-topic abstained |
|---|---|---|---|---|
| 0.15 | 100% | 60% | 100% | 40% |
| **0.20 (default)** | 97.6% | 100% | 97.6% | 80% |
| 0.25 | 95.1% | 100% | 95.1% | 80% |
| 0.30 | 90.2% | 100% | 90.2% | 100% |

0.20 abstains on every off-topic question for a single PDF while losing the gold passage for one answerable
question of 41. A higher value loses more answerable questions, and a lower one lets off-topic questions through.
Two limits: there are only 5 off-topic questions, so this is a starting point to re-tune on real PDFs; and
near-miss questions (on topic, answer absent) score 0.29 to 0.64, so no sensible threshold catches them. Only the
model can abstain on those. The English-only all-MiniLM-L6-v2 has no such gap (off-topic questions reach 0.66),
so the default threshold does not suit it.

Read these with care:

- **Recall@5 is saturated.** Each sample PDF has only 2 to 8 passages, so a top-5 list holds most of the
  document. Recall@1 and the one-index run are the informative numbers. A larger corpus is needed for a real Recall@5.
- **Answer-level numbers are not measured yet.** Answer support, citation accuracy and unsupported claims need
  a Gemma run (`--answers local`), and the local model has not been run for this benchmark. Until that is done,
  no claim is made about them.
- **Docling is not benchmarked here.** `pypdf` kept the evidence for 41 of 46 answerable questions; the 5
  misses are the scanned PDF, which has no text layer. Run `eval.bench_parsers` with Docling installed to compare.

Targets in `docs/implementation.md` (Recall@5 of at least 90%, citation accuracy of at least 95%, zero
unsupported claims) are targets, not results.

## Limitations

- The sample set is small and synthetic. Real PDFs will do worse, especially scanned and table-heavy ones.
- Hindi quality depends on the embedding model. The default multilingual model handles Devanagari and
  Hinglish far better than the English-only all-MiniLM-L6-v2, but it is still small and only tested on a few
  Hindi questions.
- Hindi PDFs whose fonts have no Unicode mapping extract as garbage. Such pages are detected, discarded and
  sent to OCR, but extraction that looks plausible and is wrong cannot be detected.
- OCR accuracy depends on scan quality. Wrong OCR text can still be cited, but with the page number it came from.
- Answers come from a small local model and can be wrong even when the citation is right. Check the snippet.

## Licenses

Check these again before a release.

| Component | License |
|---|---|
| pypdf (default parser) | BSD-3-Clause |
| PyMuPDF | AGPL. **Not used**, deliberately |
| Docling (optional) | MIT |
| Tesseract (optional) | Apache-2.0 |
| sentence-transformers, rank-bm25, Streamlit | Apache-2.0 |
| Embedding models (MiniLM family) | Apache-2.0 |
| Gemma weights | Under Google's Gemma terms. Read the model card before redistributing |

## Layout

```
app.py, pages/       Streamlit UI and the Benchmark page
rag/                 ingest, retrieval, answer, citations, providers, config
eval/                sample PDF generator, golden questions, benchmark runners, saved results
data/                sample PDFs and expected_outcomes.json
tests/               pytest suite
docs/implementation.md   plan, roles and phase gates
```

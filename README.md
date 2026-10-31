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

Python 3.10 or newer. About 5 GB of disk for the local model.

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

The embedding model is chosen with `RAG_EMBED_MODEL` (default `all-MiniLM-L6-v2`).

## Benchmark results

63 hand-written questions over 6 invented sample PDFs: English, Hindi, Hinglish (Roman-script Hindi),
code-switched, table and scanned cases; 46 answerable, 17 unanswerable. The sample text is original, so
the files are free to share. Measured with Python 3.12 on a Windows 11 laptop (AMD Radeon 610M, no discrete
GPU), retrieval only, one index per PDF. 56 of the 63 questions ran: the 7 scanned-page questions need
Tesseract, which was not installed for these runs. See `eval/results/` for the full per-question data.

| Embedding model | Recall@1 | Recall@3 | Recall@5 | Hinglish R@1 | Code-switched R@1 |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (default) | 80.5% | 100% | 100% | 80% | 50% |
| paraphrase-multilingual-MiniLM-L12-v2 | 95.1% | 100% | 100% | 100% | 75% |

With all sample PDFs in one index (`--library`), Recall@1 is 78.0% against 92.7% and Recall@5 is 100% for
both. Median search time is about 19 ms and 42 ms (the multilingual model is larger and slower; timings
vary between runs). The multilingual model ranks better, most clearly on
Hinglish and mixed-script questions.

Read these with care:

- **Recall@5 is saturated.** Each sample PDF has only 2 to 8 passages, so a top-5 list holds most of the
  document. Recall@1 and the library run are the informative numbers. A larger corpus is needed for a real Recall@5.
- **The score threshold is mis-set.** The app drops passages scoring below 0.15, but cosine scores for
  unrelated questions go far higher: up to 0.66 with the default model and 0.64 with the multilingual one.
  Retrieval abstained on only 3 of the 5 off-topic questions, and on none of the 10 near-miss questions
  (on topic, answer absent). For the multilingual model, a threshold near 0.30 would have abstained on all 5
  off-topic questions while keeping 93% of answerable ones. With 5 off-topic questions that is a hint, not a
  tuned value. Near-miss questions cannot be caught by a score; only the model can abstain on those.
- **Answer-level numbers are not measured yet.** Answer support, citation accuracy and unsupported claims need
  a Gemma run (`--answers local`). Until that is done, no claim is made about them.
- **Docling is not benchmarked here.** `pypdf` kept the evidence for 41 of 46 answerable questions; the 5
  misses are the scanned PDF, which has no text layer. Run `eval.bench_parsers` with Docling installed to compare.

Targets in `docs/implementation.md` (Recall@5 of at least 90%, citation accuracy of at least 95%, zero
unsupported claims) are targets, not results.

## Limitations

- The sample set is small and synthetic. Real PDFs will do worse, especially scanned and table-heavy ones.
- Hindi quality depends on the embedding model. The default English-only model is weak on Devanagari.
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

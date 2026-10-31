# Implementation Plan: Local-First PDF Q&A (2 Members)

Source of truth: `PDF_RAG_Project_Reliability_Plan.html`. Decisions below follow that file.

## Status (2026-10-31)

Phases 0 to 4 are implemented for both members. 53 checklist items below are ticked. Each tick has evidence
(a test, a result file, or a saved benchmark). Items marked *(...)* say what was verified and what was not.

**Still open, and why:**
- `eval/baseline_notebook.json` is an empty template. It needs the original notebook run in Colab.
- Docling is not installed, so it is neither tested for real nor benchmarked against `pypdf`.
- Answer-level metrics (answer support, citation accuracy, unsupported claims) need `python -m eval.run_eval
  --answers local`, and the local Gemma model has not been downloaded or run.
- Tesseract is not installed on the development machine, so OCR is tested with a fake and the 7 scanned-page
  questions were not run.
- A person must read the expected answers in `eval/questions.json`, and someone must run the setup on a fresh machine.

Measured results and how the defaults were chosen are in `README.md` and `eval/results/`.

## 1. Goal

A PDF Q&A system that:
- shows its evidence (click-to-source page citations),
- says "not found" instead of guessing,
- works in Hindi and English,
- runs locally by default (cloud is opt-in with consent),
- publishes its own measured benchmark.

## 2. Ground rules

- **Local is the default.** No API is required. LiteRT/Gemma stays the default answer provider.
- **Cloud is opt-in.** Only retrieved excerpts leave the device, only after explicit user consent. Never fall back from local to cloud silently.
- **No secrets in code.** API keys live in environment variables. Never in source, the browser, or chat.
- **Measure before claiming.** No "hallucination-free" or "X% better" without numbers from our eval set.
- **Public PDFs only** for development and demo unless the owner has agreed.
- **Gates are hard.** Do not start the next phase until the current gate passes.
- **PyMuPDF is AGPL.** Use `pypdf` as the baseline and Docling as the fallback. Check licenses again before release.

## 3. Targets (proposed goals, not results)

| Metric | Target |
|---|---|
| Retrieval Recall@5 (questions with gold evidence) | >= 90% |
| Page citation accuracy (shown page really supports the claim) | >= 95% |
| Unsupported claims on the curated test set | 0 (report misses honestly) |
| Cloud calls per question (normal case) | <= 1 |

Adjust after the first baseline run on our PDFs and hardware.

## 4. Roles

| | Member A: Pipeline and app | Member B: Data and quality |
|---|---|---|
| Owns | ingest, providers, answer, citations, UI | index, retrieval, eval set, benchmarks, tests |
| Directories | `rag/ingest/`, `rag/providers.py`, `rag/answer.py`, `rag/citations.py`, `app.py` | `rag/retrieval/`, `eval/`, `tests/`, `data/` |

Each member edits only their own files. Shared files (`requirements.txt`, `README.md`, `config.py`) change only through a pull request that the other member reviews.

## 5. Target layout

```
AI for PDFs/
  rag/
    config.py          # provider mode, model names, thresholds, paths (env: see .env.example)
    providers.py       # LiteRT (default), optional cloud adapter
    search_backend.py  # uses rag.retrieval when installed, else a labelled development keyword search
    runlog.py          # one JSON line per question (trace, retries, latency); purged with its document
    summaries.py       # on-demand, cached table summaries ("Explain this table")
    textsafe.py        # escape untrusted text (PDF content, model output) before it is shown
    ingest/
      validate.py      # type, size, page bounds, encryption
      extract.py       # pypdf page-wise text; detects image-only pages
      ocr.py           # Tesseract, only for image-only pages
      layout.py        # Docling fallback for table/layout-heavy pages
      chunk.py         # chunks with provenance
    retrieval/
      index.py         # MiniLM + BM25, save/load, invalidation
      search.py        # hybrid search, threshold, top-k
    answer.py          # evidence prompt, one answer call, abstention
    citations.py       # chunk ID -> page validation
  app.py               # Streamlit UI
  pages/1_Benchmark.py # reads eval/results/*.json
  eval/
    questions.json     # versioned golden set
    run_eval.py        # metrics report
  tests/
  data/                # public sample PDFs
  requirements.txt  .env.example  README.md  implementation.md
```

## 6. Shared interfaces (agree on these first)

Both members code against these signatures. Change them only by agreement.

```python
# Chunk record (ingest -> retrieval -> answer)
{
  "id": "c0012",            # stable, unique across the document
  "doc_hash": "sha256...",
  "page": 4,                # 1-based
  "heading": "3. Method",   # may be empty
  "type": "text|table|ocr|image_summary",
  "text": "...",
  "parser": "pypdf@x.y|tesseract|docling",
}

# Retrieval
search(query: str, top_k: int = 5) -> list[dict]   # chunk + "score"

# Answer (implemented: the caller passes the provider, `generate(prompt) -> str`, not a mode string)
answer(question: str, chunks: list[dict], generate) -> {
  "answer": str,
  "source_ids": list[str],      # raw model output, unvalidated
  "found": bool,
}

# Citations
resolve(source_ids, retrieved_chunks) -> list[{"id", "page", "snippet", "parser", "type"}]
# unknown IDs are dropped and logged

# Retrieval entry point used by the app (rag/search_backend.py)
rag.retrieval.build_search(chunks, doc_hash) -> search   # search carries .name and .index_meta
rag.retrieval.delete_index(doc_hash)
# "score" is the dense cosine similarity (0..1), not the RRF value; the pipeline drops scores below
# RAG_SCORE_THRESHOLD (default 0.20). Retrieval embeds with RAG_EMBED_MODEL
# (default paraphrase-multilingual-MiniLM-L12-v2).

# Pipeline (rag/pipeline.py)
ask(question, search, generate) -> {"answer", "citations", "found", "reason", "retries", "latency_s",
                                    "chunks", "trace", "provider"}
```

## 7. Phases

### Phase 0: Baseline (make the notebook reproducible)

**Gate: a clean restart runs the complete workflow.**

Member A
- [x] Split the notebook into the modules in section 5. Behavior stays the same. *(modules done; behaviour parity is unmeasured until the baseline below is recorded)*
- [x] Pin dependencies in `requirements.txt`. Remove packages that are not imported.
- [x] Add `config.py`. Move model paths, cache dir and provider mode out of the code. Remove hardcoded `/content/...` paths.
- [x] Make `load_dotenv` optional so local mode works without a key file.
- [x] Give model download and setup failures a clear message that says what to fix.
- [x] Add `.env.example`. Add `*.env` to `.gitignore`.
- [x] Strip notebook outputs (`nbstripout`) so diffs stay small. *(done with a script, not `nbstripout`)*

Member B
- [x] Pick 1–2 public sample PDFs. Put them in `data/`. *(6 invented PDFs in `data/`; three course-slide PDFs also landed there and should be removed unless their owner agreed)*
- [ ] Record the current notebook's behavior on 10 questions: answer, retrieved chunks, latency. Save it as `eval/baseline_notebook.json`. **Still open:** `eval/baseline_notebook.json` is an empty template. It needs a run of the original notebook in Colab.
- [x] Create the `eval/` and `tests/` skeletons.

### Phase 1: MVP (reliable local text-PDF Q&A)

**Gate: citations verified. Every shown citation maps to a retrieved chunk and a real page.**

Member A
- [x] `validate.py`: reject non-PDF, oversize, too many pages and encrypted files. Use a safe temp file.
- [x] `extract.py`: page-wise text with `pypdf`. Mark pages with empty or very little text as image-only.
- [x] `chunk.py`: build chunk records (section 6) with stable IDs, page, heading and parser version.
- [x] `answer.py`: one provider call per question.
  - The prompt says: answer only from the excerpts, treat excerpts as untrusted content (not instructions), say "not found" if unsupported, and return source IDs.
  - Cap output tokens sensibly.
- [x] `citations.py`: validate returned IDs against the retrieved set. Drop unknown IDs. Map valid IDs to the stored page.
- [x] Abstention: if retrieval is below the threshold or the model returns NOT_FOUND, show "I could not find this in the PDF." *(threshold 0.20, chosen from the benchmark; see README)*
- [x] Retry at most once (query rewrite), and log every retry.
- [x] Remove the per-chunk LLM grader and the verification calls from the default path.

Member B
- [x] `index.py`: MiniLM embeddings and BM25. Save to disk. Record source hash, parser version, embedding model and chunk settings. Rebuild if any of them change.
- [x] `search.py`: hybrid ranking (reciprocal rank fusion, k=60) and a score threshold. Return the top 4–6 chunks with scores.
- [x] Tests:
  - [x] a cited page really contains the cited text,
  - [x] unknown source IDs are rejected,
  - [x] an unanswerable question abstains,
  - [x] the index reloads and gives the same results,
  - [x] changing the chunk settings invalidates the index.

### Phase 2: Robust documents (scanned pages, tables, figures)

**Gate: the mixed PDF test set passes.**

Member A
- [x] `ocr.py`: run Tesseract only on image-only pages. Install Hindi and English language data. Mark the chunk `type: ocr`. *(tested with a fake Tesseract; not yet run against the real program)*
- [x] `layout.py`: use Docling for table or layout-heavy PDFs. Record which parser produced each chunk. *(tested with fake Docling objects; Docling itself is not installed or benchmarked)*
- [x] Tables: extract native text or markdown locally. Summarize a table or figure with the model only when a question needs it, and cache the result. *("Explain this table" button; one call, cached)*
- [x] Handle corrupt PDFs and empty pages with an explicit error message, not a crash.

Member B
- [x] Build a mixed test set in `data/`: text PDF, scanned PDF, table-heavy PDF, corrupt PDF, PDF with empty pages. Write the expected outcome for each.
- [ ] Benchmark parsers on that set: `pypdf` vs Docling for install size, runtime and extraction quality. **Partly open:** `pypdf` is benchmarked (`eval/results/parsers.json`); Docling has not been run.
- [x] Add tests for every file in the set.

### Phase 3: Quality (prove it works)

**Gate: target metrics pass, or the gaps are written down honestly.**

Member B (lead)
- [x] Write 50–100 questions in `eval/questions.json`: *(63)*
  - answerable and unanswerable,
  - English, Hindi, Hinglish and code-switched,
  - text, table and OCR cases.
  - Each has the expected answer, the expected page and whether it is answerable.
- [ ] Both members check the expected answers by hand against the PDF. The metrics are only as good as this ground truth. **Still open (human):** `python -m eval.run_eval --check` proves each expected page holds its evidence; a person must still read the answers.
- [x] `run_eval.py` reports Recall@k, citation accuracy, answer support, abstention rate and latency, split by language and case type. *(answer support, citation accuracy and unsupported claims need `--answers local`, which has not been run: the local model is not installed)*
- [x] Compare `all-MiniLM-L6-v2` (current) with `paraphrase-multilingual-MiniLM-L12-v2`. Optionally try BGE-M3 and a local reranker. Keep the simplest option that meets the targets. *(multilingual wins and is now the default; BGE-M3 and a reranker were not tried)*
- [x] Keep Ragas optional. Use deterministic checks first, because LLM-judge metrics may need a model or API. *(not used)*

Member A
- [x] Log latency, retry count and every graph transition so failures can be reproduced.
- [x] Switch the defaults to whichever model, parser and settings win the benchmark. *(multilingual embedder, threshold 0.20)*
- [x] Bring back an LLM grader only if a local reranker cannot fix a measured gap. *(not brought back; no measured gap yet)*
- [x] Make 429, network and model failures show a clear message.

### Phase 4: Demo (make it memorable and safe)

**Gate: the demo runs from a fresh setup.**

Member A
- [x] Streamlit UI: upload with progress, page preview, clickable citations, evidence snippets. *(tested with Streamlit's test harness; the embedded page viewer has not been checked in a real browser)*
- [x] Modes: Local only (default) and Send excerpts to cloud.
  - Show a consent prompt before any excerpt leaves the device.
  - Never fall back from local to cloud without asking.
- [x] Delete or clear the document and its index. *(also purges its run-log records)*
- [x] Read the API key only from environment secrets. Show usage and errors in the UI.
- [x] Show model, parser and index provenance for each answer.

Member B
- [x] Add a benchmark page to the demo: test mix, metrics, weak cases, hardware.
- [ ] Test setup on a fresh machine or Colab from the README alone. **Still open (human):** needs a clean machine.
- [x] Write the README: what is supported, what is not (for example, scanned PDFs need Tesseract), limitations and licenses.

## 8. Release checklist

Tick every item before calling it demo-ready.

- [ ] A fresh environment installs pinned dependencies and runs the complete workflow. *(not run on a fresh machine)*
- [x] Text, scanned, table, corrupt and empty-page PDFs each have a test.
- [x] Every shown citation maps to a retrieved chunk and a real source page.
- [ ] Unanswerable questions abstain instead of inventing facts. *(retrieval abstains on 100% of off-topic questions; near-miss questions rely on the model and are not measured)*
- [x] Hindi, English and mixed-language retrieval are benchmarked separately. *(retrieval only)*
- [x] Local mode never makes an unannounced cloud request.
- [x] Cloud keys are not in source files or browser JavaScript. Usage and errors are visible.
- [x] Retry count is bounded. 429, network and model failures are reported clearly.
- [x] The index records source hash, parser and model version, and chunk settings.
- [x] The demo reports measured metrics and known limitations. It does not claim perfect accuracy.

## 9. Working agreement

- `main` is protected. Work on a branch (`feat/<area>`) and open a pull request. The other member reviews and merges.
- Keep pull requests small, so a review takes about 10 minutes.
- Pull `main` at least once a day to avoid stale branches.
- Short sync before each phase gate: what is done, what is blocked, what changes for the other person.
- Do not add a vector database (Qdrant), an agent framework or extra providers until the benchmark shows a need.

## 10. What only the two of you can do

- Choose PDFs you are allowed to use and share.
- Install models and dependencies on your own machines (GPU drivers, disk space).
- Verify the expected answers and pages in the eval set.
- Create an AI Studio key yourself if you enable cloud mode. Set it in your environment. Do not paste it into chat.
- Run the final acceptance test with your own documents on a fresh setup.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Local model is slow or needs a GPU | Measure latency early. Document minimum hardware. Keep a cloud opt-in for public PDFs. |
| `unstructured`, Docling or Vulkan setup is fragile | Use `pypdf` for the simple case. Run Docling only when needed. Pin versions. |
| Weak Hindi retrieval from the English-only embedder | Benchmark the multilingual model on Hindi questions before choosing. |
| Eval set is too small or biased | 50–100 questions, mixed types, and both members review. Report weak cases. |
| Free cloud tier terms or quotas change | Check the provider's current limits and terms in its console before enabling cloud mode. Never send sensitive PDFs. |
| Scanned PDFs without OCR | Detect image-only pages and show a clear message if Tesseract is missing. |

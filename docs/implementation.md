# Implementation Plan: Local-First PDF Q&A (2 Members)

Source of truth: `PDF_RAG_Project_Reliability_Plan.html`. Decisions below follow that file.

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
    config.py          # provider mode, model names, thresholds, paths
    providers.py       # LiteRT (default), optional cloud adapter
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

# Answer
answer(question: str, chunks: list[dict], mode: str) -> {
  "answer": str,
  "source_ids": list[str],      # raw model output, unvalidated
}

# Citations
resolve(source_ids, retrieved_chunks) -> list[{"id", "page", "snippet"}]
# unknown IDs are dropped and logged
```

## 7. Phases

### Phase 0: Baseline (make the notebook reproducible)

**Gate: a clean restart runs the complete workflow.**

Member A
- [ ] Split the notebook into the modules in section 5. Behavior stays the same.
- [ ] Pin dependencies in `requirements.txt`. Remove packages that are not imported.
- [ ] Add `config.py`. Move model paths, cache dir and provider mode out of the code. Remove hardcoded `/content/...` paths.
- [ ] Make `load_dotenv` optional so local mode works without a key file.
- [ ] Give model download and setup failures a clear message that says what to fix.
- [ ] Add `.env.example`. Add `*.env` to `.gitignore`.
- [ ] Strip notebook outputs (`nbstripout`) so diffs stay small.

Member B
- [ ] Pick 1–2 public sample PDFs. Put them in `data/`.
- [ ] Record the current notebook's behavior on 10 questions: answer, retrieved chunks, latency. Save it as `eval/baseline_notebook.json`.
- [ ] Create the `eval/` and `tests/` skeletons.

### Phase 1: MVP (reliable local text-PDF Q&A)

**Gate: citations verified. Every shown citation maps to a retrieved chunk and a real page.**

Member A
- [ ] `validate.py`: reject non-PDF, oversize, too many pages and encrypted files. Use a safe temp file.
- [ ] `extract.py`: page-wise text with `pypdf`. Mark pages with empty or very little text as image-only.
- [ ] `chunk.py`: build chunk records (section 6) with stable IDs, page, heading and parser version.
- [ ] `answer.py`: one provider call per question.
  - The prompt says: answer only from the excerpts, treat excerpts as untrusted content (not instructions), say "not found" if unsupported, and return source IDs.
  - Cap output tokens sensibly.
- [ ] `citations.py`: validate returned IDs against the retrieved set. Drop unknown IDs. Map valid IDs to the stored page.
- [ ] Abstention: if retrieval is below the threshold or the model returns NOT_FOUND, show "I could not find this in the PDF."
- [ ] Retry at most once (query rewrite), and log every retry.
- [ ] Remove the per-chunk LLM grader and the verification calls from the default path.

Member B
- [ ] `index.py`: MiniLM embeddings and BM25. Save to disk. Record source hash, parser version, embedding model and chunk settings. Rebuild if any of them change.
- [ ] `search.py`: hybrid ranking (reciprocal rank fusion, k=60) and a score threshold. Return the top 4–6 chunks with scores.
- [ ] Tests:
  - [ ] a cited page really contains the cited text,
  - [ ] unknown source IDs are rejected,
  - [ ] an unanswerable question abstains,
  - [ ] the index reloads and gives the same results,
  - [ ] changing the chunk settings invalidates the index.

### Phase 2: Robust documents (scanned pages, tables, figures)

**Gate: the mixed PDF test set passes.**

Member A
- [ ] `ocr.py`: run Tesseract only on image-only pages. Install Hindi and English language data. Mark the chunk `type: ocr`.
- [ ] `layout.py`: use Docling for table or layout-heavy PDFs. Record which parser produced each chunk.
- [ ] Tables: extract native text or markdown locally. Summarize a table or figure with the model only when a question needs it, and cache the result.
- [ ] Handle corrupt PDFs and empty pages with an explicit error message, not a crash.

Member B
- [ ] Build a mixed test set in `data/`: text PDF, scanned PDF, table-heavy PDF, corrupt PDF, PDF with empty pages. Write the expected outcome for each.
- [ ] Benchmark parsers on that set: `pypdf` vs Docling for install size, runtime and extraction quality.
- [ ] Add tests for every file in the set.

### Phase 3: Quality (prove it works)

**Gate: target metrics pass, or the gaps are written down honestly.**

Member B (lead)
- [ ] Write 50–100 questions in `eval/questions.json`:
  - answerable and unanswerable,
  - English, Hindi, Hinglish and code-switched,
  - text, table and OCR cases.
  - Each has the expected answer, the expected page and whether it is answerable.
- [ ] Both members check the expected answers by hand against the PDF. The metrics are only as good as this ground truth.
- [ ] `run_eval.py` reports Recall@k, citation accuracy, answer support, abstention rate and latency, split by language and case type.
- [ ] Compare `all-MiniLM-L6-v2` (current) with `paraphrase-multilingual-MiniLM-L12-v2`. Optionally try BGE-M3 and a local reranker. Keep the simplest option that meets the targets.
- [ ] Keep Ragas optional. Use deterministic checks first, because LLM-judge metrics may need a model or API.

Member A
- [ ] Log latency, retry count and every graph transition so failures can be reproduced.
- [ ] Switch the defaults to whichever model, parser and settings win the benchmark.
- [ ] Bring back an LLM grader only if a local reranker cannot fix a measured gap.
- [ ] Make 429, network and model failures show a clear message.

### Phase 4: Demo (make it memorable and safe)

**Gate: the demo runs from a fresh setup.**

Member A
- [ ] Streamlit UI: upload with progress, page preview, clickable citations, evidence snippets.
- [ ] Modes: Local only (default) and Send excerpts to cloud.
  - Show a consent prompt before any excerpt leaves the device.
  - Never fall back from local to cloud without asking.
- [ ] Delete or clear the document and its index.
- [ ] Read the API key only from environment secrets. Show usage and errors in the UI.
- [ ] Show model, parser and index provenance for each answer.

Member B
- [ ] Add a benchmark page to the demo: test mix, metrics, weak cases, hardware.
- [ ] Test setup on a fresh machine or Colab from the README alone.
- [ ] Write the README: what is supported, what is not (for example, scanned PDFs need Tesseract), limitations and licenses.

## 8. Release checklist

Tick every item before calling it demo-ready.

- [ ] A fresh environment installs pinned dependencies and runs the complete workflow.
- [ ] Text, scanned, table, corrupt and empty-page PDFs each have a test.
- [ ] Every shown citation maps to a retrieved chunk and a real source page.
- [ ] Unanswerable questions abstain instead of inventing facts.
- [ ] Hindi, English and mixed-language retrieval are benchmarked separately.
- [ ] Local mode never makes an unannounced cloud request.
- [ ] Cloud keys are not in source files or browser JavaScript. Usage and errors are visible.
- [ ] Retry count is bounded. 429, network and model failures are reported clearly.
- [ ] The index records source hash, parser and model version, and chunk settings.
- [ ] The demo reports measured metrics and known limitations. It does not claim perfect accuracy.

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

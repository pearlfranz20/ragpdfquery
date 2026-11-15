# PDF RAG System

A production-shaped Retrieval-Augmented Generation system for asking questions
over a corpus of thousands of PDFs, with citations, multilingual support,
hallucination scoring, and an admin dashboard.

## Architecture

```
                     ┌────────────────────┐
  PDFs  ───ingest──▶ │  Ingestion Pipeline │
                     │  extract → chunk    │
                     │  → embed → store    │
                     └─────────┬───────────┘
                               │
                        Postgres + pgvector
                               │
Question ─▶ FastAPI ─▶ LangGraph Agent ─▶ Answer + Citations
                               │
                      DeepEval (offline + inline
                        hallucination check)

                     Streamlit Admin Dashboard
                     (ingestion status, eval runs,
                      corpus stats, query tester)
```

### Why these choices

- **LangGraph** over PydanticAI: the pipeline needs a real *cycle* — retrieve →
  grade → generate → check-hallucination → retry-with-more-context — which
  maps naturally onto LangGraph's stateful graph. (A PydanticAI version would
  work too; swap `app/agent/graph.py` for an `Agent` + manual loop if you
  prefer that stack — the retrieval/embedding/eval layers underneath are
  framework-agnostic and don't need to change.)
- **Bedrock or OpenAI**: `app/agent/llm.py` and `app/ingestion/embeddings.py`
  are thin provider-abstraction layers switched by `LLM_PROVIDER` /
  `EMBEDDING_PROVIDER` env vars, so you can run entirely on Bedrock (good for
  AWS-native / data-residency requirements) or entirely on OpenAI, or mix
  (e.g. Bedrock Titan embeddings + GPT for generation).
- **Postgres + pgvector**: keeps operational surface small — one database for
  document metadata, chunks, vectors, ingestion job state, and eval results.
  Scales to low millions of chunks with an IVFFLAT/HNSW index, which covers
  "thousands of PDFs" comfortably.
- **Textract with local OCR fallback**: text-native PDFs are parsed directly
  (fast, cheap, free); scanned/complex-layout PDFs go through Textract for
  OCR + layout; if AWS isn't configured, a local `pytesseract` fallback keeps
  the pipeline runnable without cloud credentials for dev/testing.
- **DeepEval**: used both as an offline eval suite (`scripts/run_eval.py`
  against a golden test set — faithfulness, answer relevancy, contextual
  precision/recall) and optionally inline as a cheap faithfulness check with
  a retry loop before an answer is returned to the user.

## Project layout

```
app/
  config.py            # env-driven settings
  db.py                # SQLAlchemy engine/session
  models.py             # ORM models (documents, chunks, jobs, eval_runs)
  schemas.py            # Pydantic request/response models
  main.py                # FastAPI app entrypoint
  ingestion/
    extract.py           # PDF text/layout extraction (Textract + local fallback)
    language.py           # per-page language detection
    chunking.py            # multiple chunking strategies
    embeddings.py           # Bedrock/OpenAI embedding provider
    pipeline.py               # orchestrates a single PDF end-to-end
  retrieval/
    vector_store.py       # pgvector similarity search + filters
  agent/
    llm.py                # Bedrock/OpenAI chat provider
    prompts.py             # system/user prompt templates
    graph.py                 # LangGraph state graph (the RAG agent)
  api/
    routes_ingest.py       # upload / trigger ingestion, job status
    routes_query.py         # ask a question, get answer + citations
    routes_admin.py          # corpus stats, eval runs, reindex
  eval/
    deepeval_suite.py       # DeepEval metrics + test runner
    golden_dataset.jsonl      # example eval test cases
dashboard/
  streamlit_app.py          # admin UI
scripts/
  ingest_folder.py           # CLI: bulk-ingest a directory of PDFs
  run_eval.py                  # CLI: run the DeepEval suite, store results
  generate_golden_dataset.py     # CLI: auto-generate golden eval cases from your own corpus
db/init.sql                     # schema + pgvector extension
docker-compose.yml
requirements.txt
.env.example
```

## Setup

```bash
cp .env.example .env        # fill in your provider credentials
docker compose up -d db     # starts Postgres 16 + pgvector
pip install -r requirements.txt

# create schema
psql "$DATABASE_URL" -f db/init.sql

# bulk ingest a folder of PDFs (thousands is fine — it's resumable/idempotent
# per-file via a content hash, and safe to re-run)
python scripts/ingest_folder.py /path/to/pdfs --workers 8

# run the API
uvicorn app.main:app --reload --port 8000

# ask a question
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "What were the termination clauses in the 2023 vendor contracts?"}'

# admin dashboard
streamlit run dashboard/streamlit_app.py

# offline eval
python scripts/run_eval.py
```

## Chunking strategies

Selectable via `CHUNKING_STRATEGY` env var or per-ingest-request override:

- `recursive` (default) — recursive character splitting with token-aware
  sizing (LangChain `RecursiveCharacterTextSplitter`), good general default.
- `section` — splits on detected headings/section boundaries first, then
  recursively splits any oversized sections. Better for structured docs
  (contracts, policies, technical manuals).
- `page` — one chunk per page (with overlap into neighboring pages), useful
  when page-level citation granularity matters more than semantic coherence.
- `semantic` — embeds sentences and splits at points of low semantic
  similarity between consecutive sentences (embedding-based breakpoint
  detection). Highest quality, highest ingestion cost.

All strategies preserve `page_number`, `source_document_id`, and a
`char_offset` so every chunk can be cited back to an exact page.

## Citations

Every chunk stored carries `document_id`, `filename`, `page_number`, and
`chunk_index`. The generation prompt requires the model to tag claims with
`[doc_id:page]` markers; `app/agent/graph.py`'s `format_citations` node
resolves those markers against the retrieved chunk metadata and returns a
structured `citations` array (filename, page, snippet, download link)
alongside the prose answer — never just a bare doc name.

## Multilingual support

- Ingestion: `app/ingestion/language.py` detects the language of each page
  (`langdetect`) and stores it on the chunk. Embeddings use a multilingual
  model (Bedrock Titan Embed Text v2, or OpenAI `text-embedding-3-large`,
  both of which embed well across languages into a shared space) so a
  question in French can retrieve relevant English source chunks and vice
  versa.
- Query: the agent detects the question's language and instructs the LLM to
  answer in that language regardless of the source document's language,
  while citations still point at the original-language source text.

## Hallucination evaluation

- **Inline** (optional, `ENABLE_INLINE_FAITHFULNESS_CHECK=true`): after
  generation, a DeepEval `FaithfulnessMetric` scores the answer against
  retrieved context; scores below `FAITHFULNESS_THRESHOLD` trigger one retry
  with a stricter "only use the provided context" prompt before returning
  (with the score surfaced in the API response either way).
- **Offline**: `scripts/run_eval.py` runs the full DeepEval suite
  (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall,
  Hallucination) against `app/eval/golden_dataset.jsonl` and writes results
  to the `eval_runs` table, visible in the admin dashboard's Eval tab as a
  trend over time.

## Building a real golden dataset from your corpus

`app/eval/golden_dataset.jsonl` ships with placeholder questions — replace
them before trusting eval numbers. Three ways to get real ones, roughly
least to most effort:

**1. Auto-generate from your ingested chunks (fastest)**

```bash
python scripts/generate_golden_dataset.py \
    --num-documents 20 --contexts-per-document 3 --goldens-per-context 2 \
    --output app/eval/golden_dataset_generated.jsonl
```

This pulls real chunks out of Postgres for a random sample of your
documents, groups adjacent chunks into small contexts, and uses DeepEval's
`Synthesizer` to write grounded questions plus an `expected_output` derived
from that context (not hallucinated — it's a paraphrase of text that's
actually there). Output is in the same JSONL format as `golden_dataset.jsonl`,
tagged with the `document_id` each case came from.

Always spot-check before merging into your real eval set — Synthesizer is
good but not infallible on dense tables, boilerplate legal text, or pages
that are mostly headers/footers. Delete anything that doesn't read like a
question a real user would actually ask, then fold the rest into
`app/eval/golden_dataset.jsonl`.

You can push further with DeepEval's `EvolutionConfig` to generate harder
variants (multi-hop reasoning, comparative questions across documents,
hypotheticals) — see `Synthesizer(evolution_config=...)` in the
[DeepEval docs](https://deepeval.com/docs/synthesizer-generate-from-contexts).

**2. Mine real questions once the app is live**

Every query is already logged to the `query_log` table (`app/models.py`).
After a few weeks of real usage:

```sql
SELECT question, answer, citations, faithfulness_score
FROM query_log
ORDER BY created_at DESC
LIMIT 200;
```

Pull the interesting ones — especially any with a low `faithfulness_score`,
or questions people had to rephrase (a sign the first answer missed) — write
the *correct* expected answer by hand, and add them as golden cases. This is
the highest-signal source because it's what people actually ask, not what an
LLM guesses they'd ask.

**3. Manual curation with a subject-matter expert**

For anything high-stakes (contracts, compliance, medical/financial content),
have someone who actually knows the corpus write 15–30 questions directly,
covering: straightforward lookups, questions that require synthesizing two
documents, and a few that are deliberately unanswerable from the corpus (to
verify the system says "I don't know" instead of guessing — see the
`unanswerable_question` example already in the dataset). SME-written cases
are slower to produce but catch domain-specific failure modes synthetic
generation tends to miss.

## Admin dashboard

Streamlit app with four tabs:

1. **Ingestion** — job queue status, per-file success/failure, retry-failed
   button, corpus size (docs/pages/chunks).
2. **Query tester** — ask questions, see the answer, citations, retrieved
   chunks, and latency breakdown (retrieve/generate/eval).
3. **Evaluation** — DeepEval metric trends over time, drill into individual
   failing test cases.
4. **Corpus browser** — search/filter documents by filename, language,
   ingestion date; view/delete a document (cascades to its chunks).

## Notes / production hardening TODOs

This is a complete, runnable skeleton, not a toy — but a few things worth
calling out before a real production deploy:

- Add auth (the API/dashboard have no auth wired in — put them behind your
  SSO/API gateway).
- Swap the naive thread-pool ingestion worker for a real queue (SQS + workers,
  or Celery) once you're past a few thousand PDFs per batch.
- Tune the pgvector index (`ivfflat` lists / `hnsw` m,ef) to your actual
  corpus size — defaults in `db/init.sql` are reasonable for ~100k-1M chunks.
- Add PII scanning/redaction in the ingestion pipeline if your PDFs contain
  sensitive personal data.

PDF uploads are streamed to temporary storage with a 50 MiB default limit and
must begin with a PDF signature. Set `MAX_UPLOAD_SIZE_BYTES` in `.env` to
change the limit.

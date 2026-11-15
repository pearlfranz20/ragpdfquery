-- RAG system schema
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source PDF
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,   -- sha256, makes re-ingestion idempotent
    source_path     TEXT,
    num_pages       INTEGER,
    primary_language TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|completed|failed
    error_message   TEXT,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Embedding dimension: must match OPENAI_EMBEDDING_DIM / BEDROCK_EMBEDDING_DIM
-- for the provider you use (see .env.example). Default here is 1536, matching
-- the default OpenAI text-embedding-3-large truncated to 1536 dims; use 1024
-- for Bedrock Titan v2. Change before first ingest, since pgvector requires a
-- fixed dimension. Note the ivfflat/hnsw indexes below cap out at 2000 dims,
-- so the full 3072-dim OpenAI output cannot be indexed without halfvec.
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER,
    text            TEXT NOT NULL,
    token_count     INTEGER,
    language        TEXT,
    char_offset     INTEGER,
    chunking_strategy TEXT,
    embedding       vector(1536),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- Vector index — IVFFLAT is fine up to ~1M rows; switch to HNSW for larger
-- corpora or lower-latency needs (requires pgvector >= 0.5).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Tracks bulk ingestion jobs (one row per file within a batch run)
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id        UUID NOT NULL,
    document_id     UUID REFERENCES documents(id) ON DELETE SET NULL,
    filename        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued', -- queued|running|succeeded|failed
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_batch ON ingestion_jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);

-- Stores every answered query for auditing + the inline faithfulness score
CREATE TABLE IF NOT EXISTS query_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question            TEXT NOT NULL,
    detected_language   TEXT,
    answer              TEXT,
    citations           JSONB,
    retrieved_chunk_ids UUID[],
    faithfulness_score  DOUBLE PRECISION,
    retried             BOOLEAN DEFAULT FALSE,
    latency_ms          INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Offline DeepEval suite runs, for the dashboard's Eval tab
CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    metric_name     TEXT NOT NULL,
    test_case_name  TEXT NOT NULL,
    score           DOUBLE PRECISION,
    passed          BOOLEAN,
    threshold       DOUBLE PRECISION,
    reason          TEXT,
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_run_at ON eval_runs(run_at);

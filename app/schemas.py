import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    document_ids: list[uuid.UUID] | None = None   # optional scoping to a subset of the corpus
    language_override: str | None = None            # force answer language (else auto-detect)


class Citation(BaseModel):
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    chunk_id: uuid.UUID
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    detected_language: str
    citations: list[Citation]
    faithfulness_score: float | None
    retried: bool
    latency_ms: int


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    num_pages: int | None
    primary_language: str | None
    error_message: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestionJobOut(BaseModel):
    id: uuid.UUID
    batch_id: uuid.UUID
    filename: str
    status: str
    error_message: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CorpusStats(BaseModel):
    num_documents: int
    num_documents_by_status: dict[str, int]
    num_chunks: int
    num_pages: int
    languages: dict[str, int]


class EvalSummary(BaseModel):
    metric_name: str
    avg_score: float
    pass_rate: float
    num_cases: int
    run_at: datetime

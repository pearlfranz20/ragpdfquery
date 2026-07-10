from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Chunk, Document

settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    text: str
    similarity: float


def similarity_search(
    db: Session,
    query_embedding: list[float],
    top_k: int | None = None,
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    min_similarity = min_similarity if min_similarity is not None else settings.retrieval_min_similarity

    # pgvector cosine_distance = 1 - cosine_similarity
    distance = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(Chunk, Document.filename, (1 - distance).label("similarity"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == "completed")
        .order_by(distance)
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    rows = db.execute(stmt).all()

    results = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity=float(similarity),
        )
        for chunk, filename, similarity in rows
        if similarity >= min_similarity
    ]
    return results

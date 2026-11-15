from __future__ import annotations

import hashlib
import logging
import os
import uuid

from app.config import get_settings
from app.db import session_scope
from app.ingestion import chunking
from app.ingestion.embeddings import get_embedding_provider
from app.ingestion.extract import extract_pdf
from app.ingestion.language import detect_document_language, detect_language
from app.models import Chunk, Document

logger = logging.getLogger(__name__)
settings = get_settings()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(
    pdf_path: str,
    filename: str | None = None,
    strategy: str | None = None,
    *,
    persist_source: bool = True,
) -> uuid.UUID:
    """
    End-to-end ingestion of a single PDF: extract -> chunk -> embed -> store.
    Idempotent: re-ingesting a file with the same content hash is a no-op if
    it already completed successfully.

    `persist_source=False` stores no source_path on the document. Use it when
    `pdf_path` is a temporary file that will not exist after ingestion (API
    uploads), so a later reindex does not point at a missing file.

    Returns the document_id.
    """
    filename = filename or os.path.basename(pdf_path)
    strategy = strategy or settings.chunking_strategy
    content_hash = _hash_file(pdf_path)

    with session_scope() as db:
        existing = db.query(Document).filter_by(content_hash=content_hash).one_or_none()
        if existing and existing.status == "completed":
            logger.info("Skipping %s — already ingested (doc %s)", filename, existing.id)
            return existing.id

        if existing:
            document = existing
            document.status = "processing"
            document.error_message = None
        else:
            document = Document(
                filename=filename,
                content_hash=content_hash,
                source_path=pdf_path if persist_source else None,
                status="processing",
            )
            db.add(document)
        db.flush()
        document_id = document.id

    try:
        pages = extract_pdf(pdf_path)
        page_texts = [p.text for p in pages]
        doc_language = detect_document_language(page_texts)

        embed_fn = get_embedding_provider()

        if strategy == "recursive":
            raw_chunks = chunking.chunk_recursive(pages, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        elif strategy == "page":
            raw_chunks = chunking.chunk_page(pages)
        elif strategy == "section":
            raw_chunks = chunking.chunk_section(pages, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        elif strategy == "semantic":
            raw_chunks = chunking.chunk_semantic(pages, embed_fn=lambda texts: embed_fn.embed(texts), max_tokens=settings.chunk_size_tokens)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        if not raw_chunks:
            raise ValueError("No extractable text found in PDF (extraction + OCR both produced nothing)")

        texts = [c.text for c in raw_chunks]
        vectors = embed_fn.embed(texts)

        with session_scope() as db:
            # Clear any previous chunks for this document (re-ingest case)
            db.query(Chunk).filter_by(document_id=document_id).delete()

            for raw, vector in zip(raw_chunks, vectors, strict=True):
                chunk_lang = detect_language(raw.text, default=doc_language)
                db.add(
                    Chunk(
                        document_id=document_id,
                        chunk_index=raw.chunk_index,
                        page_number=raw.page_number,
                        text=raw.text,
                        token_count=chunking.count_tokens(raw.text),
                        language=chunk_lang,
                        char_offset=raw.char_offset,
                        chunking_strategy=strategy,
                        embedding=vector,
                    )
                )

            document = db.get(Document, document_id)
            document.num_pages = len(pages)
            document.primary_language = doc_language
            document.status = "completed"
            document.error_message = None

        logger.info("Ingested %s: %d pages, %d chunks, lang=%s", filename, len(pages), len(raw_chunks), doc_language)
        return document_id

    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to ingest %s", filename)
        with session_scope() as db:
            document = db.get(Document, document_id)
            document.status = "failed"
            document.error_message = str(e)[:2000]
        raise

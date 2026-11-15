import logging
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, session_scope
from app.ingestion.pipeline import ingest_pdf
from app.models import Document, IngestionJob
from app.schemas import DocumentOut, IngestionJobOut

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

PDF_SIGNATURE = b"%PDF-"
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _update_job(job_id: uuid.UUID, **fields) -> None:
    with session_scope() as db:
        job = db.get(IngestionJob, job_id)
        if job is None:
            logger.warning("Ingestion job %s disappeared before status update", job_id)
            return
        for name, value in fields.items():
            setattr(job, name, value)


def _run_ingestion_job(
    job_id: uuid.UUID,
    pdf_path: str,
    filename: str,
    strategy: str | None,
    *,
    persist_source: bool = True,
    cleanup_dir: str | None = None,
) -> None:
    """
    Background worker for a single ingestion job.

    `cleanup_dir` is a temp directory owned by this job (created by the upload
    endpoint) that is removed when the job finishes. Reindex jobs operate on
    the document's original source path and pass no cleanup dir, so that path
    is never touched.
    """
    _update_job(job_id, status="running", started_at=_utcnow())
    try:
        document_id = ingest_pdf(pdf_path, filename=filename, strategy=strategy, persist_source=persist_source)
        _update_job(job_id, status="succeeded", document_id=document_id, finished_at=_utcnow())
    except Exception as e:  # noqa: BLE001
        logger.exception("Ingestion job %s failed for %s", job_id, filename)
        _update_job(job_id, status="failed", error_message=str(e)[:2000], finished_at=_utcnow())
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def _save_upload_to_temp(file: UploadFile) -> tuple[str, str]:
    """
    Stream an upload into a fresh temp directory, enforcing the size limit and
    checking the PDF signature. Returns (tmp_dir, tmp_path). On any failure the
    temp directory is removed before the exception propagates.
    """
    tmp_dir = tempfile.mkdtemp(prefix="pdf-upload-")
    try:
        total_size = 0
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", dir=tmp_dir, delete=False) as tmp_file:
            tmp_path = tmp_file.name
            while chunk := file.file.read(UPLOAD_READ_CHUNK_BYTES):
                total_size += len(chunk)
                if total_size > settings.max_upload_size_bytes:
                    raise HTTPException(413, f"PDF exceeds the {settings.max_upload_size_bytes}-byte upload limit")
                tmp_file.write(chunk)

        with open(tmp_path, "rb") as tmp_file:
            if tmp_file.read(len(PDF_SIGNATURE)) != PDF_SIGNATURE:
                raise HTTPException(400, "Uploaded file does not have a valid PDF signature")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir, tmp_path


@router.post("/upload", response_model=IngestionJobOut)
def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks, strategy: str | None = None, db: Session = Depends(get_db)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    tmp_dir, tmp_path = _save_upload_to_temp(file)

    try:
        job = IngestionJob(batch_id=uuid.uuid4(), filename=filename, status="queued")
        db.add(job)
        db.commit()
        db.refresh(job)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Uploaded files live only in the temp dir, which is deleted after the job,
    # so no source_path is persisted; reindexing requires a re-upload.
    background_tasks.add_task(
        _run_ingestion_job, job.id, tmp_path, filename, strategy, persist_source=False, cleanup_dir=tmp_dir
    )
    return job


@router.get("/jobs/{batch_id}", response_model=list[IngestionJobOut])
def get_batch_status(batch_id: uuid.UUID, db: Session = Depends(get_db)):
    jobs = db.query(IngestionJob).filter_by(batch_id=batch_id).all()
    if not jobs:
        raise HTTPException(404, "Batch not found")
    return jobs


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(status: str | None = None, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    q = db.query(Document)
    if status:
        q = q.filter_by(status=status)
    return q.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()


@router.delete("/documents/{document_id}")
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    db.delete(doc)  # cascades to chunks
    db.commit()
    return {"deleted": str(document_id)}


@router.post("/documents/{document_id}/reindex", response_model=IngestionJobOut)
def reindex_document(document_id: uuid.UUID, background_tasks: BackgroundTasks, strategy: str | None = None, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if not doc.source_path or not os.path.isfile(doc.source_path):
        raise HTTPException(400, "Original source file unavailable for reindex; re-upload instead")

    job = IngestionJob(batch_id=uuid.uuid4(), filename=doc.filename, status="queued", document_id=doc.id)
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_ingestion_job, job.id, doc.source_path, doc.filename, strategy)
    return job

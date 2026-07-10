import logging
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

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

_executor = ThreadPoolExecutor(max_workers=settings.ingest_max_workers)


def _run_ingestion_job(job_id: uuid.UUID, tmp_path: str, filename: str, strategy: str | None):
    with session_scope() as db:
        job = db.query(IngestionJob).get(job_id)
        job.status = "running"
        from datetime import datetime, timezone

        job.started_at = datetime.now(timezone.utc)

    try:
        document_id = ingest_pdf(tmp_path, filename=filename, strategy=strategy)
        with session_scope() as db:
            job = db.query(IngestionJob).get(job_id)
            job.status = "succeeded"
            job.document_id = document_id
            from datetime import datetime, timezone

            job.finished_at = datetime.now(timezone.utc)
    except Exception as e:  # noqa: BLE001
        with session_scope() as db:
            job = db.query(IngestionJob).get(job_id)
            job.status = "failed"
            job.error_message = str(e)[:2000]
            from datetime import datetime, timezone

            job.finished_at = datetime.now(timezone.utc)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@router.post("/upload", response_model=IngestionJobOut)
def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks, strategy: str | None = None, db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = f"{tmp_dir}/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    batch_id = uuid.uuid4()
    job = IngestionJob(batch_id=batch_id, filename=file.filename, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_ingestion_job, job.id, tmp_path, file.filename, strategy)
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
    doc = db.query(Document).get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    db.delete(doc)  # cascades to chunks
    db.commit()
    return {"deleted": str(document_id)}


@router.post("/documents/{document_id}/reindex", response_model=IngestionJobOut)
def reindex_document(document_id: uuid.UUID, background_tasks: BackgroundTasks, strategy: str | None = None, db: Session = Depends(get_db)):
    doc = db.query(Document).get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if not doc.source_path:
        raise HTTPException(400, "Original source path unavailable for reindex; re-upload instead")

    batch_id = uuid.uuid4()
    job = IngestionJob(batch_id=batch_id, filename=doc.filename, status="queued", document_id=doc.id)
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_ingestion_job, job.id, doc.source_path, doc.filename, strategy)
    return job

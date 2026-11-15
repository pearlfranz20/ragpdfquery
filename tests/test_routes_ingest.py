"""
Tests for the upload endpoint and the background ingestion job's temp-file
handling. No database is needed: the FastAPI DB dependency is overridden with
an in-memory fake and the job runner's DB writes are stubbed out.
"""
import io
import os
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_ingest
from app.db import get_db


class FakeSession:
    """Just enough of a SQLAlchemy Session for the upload route."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, obj):
        # Mimic DB-assigned defaults so the response model can serialize.
        obj.id = obj.id or uuid.uuid4()
        obj.created_at = datetime.now(UTC)

    def close(self):
        pass


@pytest.fixture
def fake_db():
    return FakeSession()


@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(routes_ingest.router, prefix="/ingest")
    app.dependency_overrides[get_db] = lambda: fake_db
    return TestClient(app)


@pytest.fixture
def recorded_jobs(monkeypatch):
    """Replace the background worker with a recorder so no ingestion runs."""
    calls = []

    def fake_run(job_id, pdf_path, filename, strategy, *, persist_source=True, cleanup_dir=None):
        with open(pdf_path, "rb") as f:
            content = f.read()
        calls.append(
            {
                "job_id": job_id,
                "pdf_path": pdf_path,
                "filename": filename,
                "strategy": strategy,
                "persist_source": persist_source,
                "cleanup_dir": cleanup_dir,
                "content": content,
            }
        )
        if cleanup_dir:
            routes_ingest.shutil.rmtree(cleanup_dir, ignore_errors=True)

    monkeypatch.setattr(routes_ingest, "_run_ingestion_job", fake_run)
    return calls


def upload(client, filename, content, **params):
    return client.post("/ingest/upload", files={"file": (filename, io.BytesIO(content), "application/pdf")}, params=params)


def test_upload_rejects_non_pdf_extension(client, recorded_jobs):
    resp = upload(client, "notes.txt", b"%PDF-1.4 looks like a pdf but is not named one")
    assert resp.status_code == 400
    assert "Only PDF" in resp.json()["detail"]
    assert recorded_jobs == []


def test_upload_rejects_bad_signature(client, recorded_jobs):
    resp = upload(client, "fake.pdf", b"this is not a pdf")
    assert resp.status_code == 400
    assert "signature" in resp.json()["detail"]
    assert recorded_jobs == []


def test_upload_rejects_oversize_file(client, recorded_jobs, monkeypatch):
    monkeypatch.setattr(routes_ingest.settings, "max_upload_size_bytes", 10)
    resp = upload(client, "big.pdf", b"%PDF-1.7 " + b"x" * 100)
    assert resp.status_code == 413
    assert recorded_jobs == []


def test_upload_queues_job_and_hands_off_temp_dir(client, fake_db, recorded_jobs):
    body = b"%PDF-1.7\n%fake content\n"
    resp = upload(client, "../../etc/report.pdf", body, strategy="page")
    assert resp.status_code == 200, resp.text

    payload = resp.json()
    assert payload["status"] == "queued"
    assert payload["filename"] == "report.pdf"  # directory components stripped
    assert len(fake_db.added) == 1

    assert len(recorded_jobs) == 1
    call = recorded_jobs[0]
    assert call["filename"] == "report.pdf"
    assert call["strategy"] == "page"
    assert call["content"] == body
    assert call["persist_source"] is False  # uploads must not persist the temp path
    assert call["cleanup_dir"] is not None
    assert os.path.dirname(call["pdf_path"]) == call["cleanup_dir"]
    assert not os.path.exists(call["cleanup_dir"])  # worker removed it


def test_upload_removes_temp_dir_on_rejection(client, recorded_jobs, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_ingest.tempfile, "tempdir", str(tmp_path))
    upload(client, "fake.pdf", b"not a pdf")
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def stubbed_job_db(monkeypatch):
    updates = []
    monkeypatch.setattr(routes_ingest, "_update_job", lambda job_id, **fields: updates.append(fields))
    return updates


def test_run_job_removes_cleanup_dir_on_success(monkeypatch, tmp_path, stubbed_job_db):
    doc_id = uuid.uuid4()
    monkeypatch.setattr(routes_ingest, "ingest_pdf", lambda *a, **kw: doc_id)
    pdf = tmp_path / "upload.pdf"
    pdf.write_bytes(b"%PDF-")

    routes_ingest._run_ingestion_job(uuid.uuid4(), str(pdf), "upload.pdf", None, persist_source=False, cleanup_dir=str(tmp_path))

    assert not tmp_path.exists()
    assert stubbed_job_db[-1]["status"] == "succeeded"
    assert stubbed_job_db[-1]["document_id"] == doc_id


def test_run_job_removes_cleanup_dir_on_failure(monkeypatch, tmp_path, stubbed_job_db):
    def boom(*a, **kw):
        raise ValueError("extraction failed")

    monkeypatch.setattr(routes_ingest, "ingest_pdf", boom)
    pdf = tmp_path / "upload.pdf"
    pdf.write_bytes(b"%PDF-")

    routes_ingest._run_ingestion_job(uuid.uuid4(), str(pdf), "upload.pdf", None, cleanup_dir=str(tmp_path))

    assert not tmp_path.exists()
    assert stubbed_job_db[-1]["status"] == "failed"
    assert "extraction failed" in stubbed_job_db[-1]["error_message"]


def test_run_job_leaves_source_alone_without_cleanup_dir(monkeypatch, tmp_path, stubbed_job_db):
    monkeypatch.setattr(routes_ingest, "ingest_pdf", lambda *a, **kw: uuid.uuid4())
    pdf = tmp_path / "corpus.pdf"
    pdf.write_bytes(b"%PDF-")

    routes_ingest._run_ingestion_job(uuid.uuid4(), str(pdf), "corpus.pdf", None)

    assert pdf.exists()  # reindex jobs must never delete the original source

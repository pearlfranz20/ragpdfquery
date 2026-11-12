from fastapi import APIRouter, Depends
from sqlalchemy import Integer, func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Chunk, Document, EvalRun
from app.schemas import CorpusStats, EvalSummary

router = APIRouter()


@router.get("/stats", response_model=CorpusStats)
def corpus_stats(db: Session = Depends(get_db)):
    num_documents = db.query(func.count(Document.id)).scalar() or 0
    num_chunks = db.query(func.count(Chunk.id)).scalar() or 0
    num_pages = db.query(func.coalesce(func.sum(Document.num_pages), 0)).scalar() or 0

    status_rows = db.query(Document.status, func.count(Document.id)).group_by(Document.status).all()
    lang_rows = db.query(Document.primary_language, func.count(Document.id)).group_by(Document.primary_language).all()

    return CorpusStats(
        num_documents=num_documents,
        num_documents_by_status={s or "unknown": c for s, c in status_rows},
        num_chunks=num_chunks,
        num_pages=int(num_pages),
        languages={lang or "unknown": c for lang, c in lang_rows},
    )


@router.get("/eval/latest", response_model=list[EvalSummary])
def latest_eval_summary(db: Session = Depends(get_db)):
    latest_run_at = db.query(func.max(EvalRun.run_at)).scalar()
    if not latest_run_at:
        return []

    rows = (
        db.query(
            EvalRun.metric_name,
            func.avg(EvalRun.score).label("avg_score"),
            func.avg(EvalRun.passed.cast(Integer)).label("pass_rate"),
            func.count(EvalRun.id).label("num_cases"),
        )
        .filter(EvalRun.run_at == latest_run_at)
        .group_by(EvalRun.metric_name)
        .all()
    )

    return [
        EvalSummary(
            metric_name=r.metric_name,
            avg_score=float(r.avg_score or 0),
            pass_rate=float(r.pass_rate or 0),
            num_cases=r.num_cases,
            run_at=latest_run_at,
        )
        for r in rows
    ]


@router.get("/eval/history")
def eval_history(metric_name: str | None = None, db: Session = Depends(get_db)):
    q = db.query(EvalRun)
    if metric_name:
        q = q.filter_by(metric_name=metric_name)
    rows = q.order_by(EvalRun.run_at.desc()).limit(500).all()
    return [
        {
            "run_at": r.run_at,
            "metric_name": r.metric_name,
            "test_case_name": r.test_case_name,
            "score": r.score,
            "passed": r.passed,
            "reason": r.reason,
        }
        for r in rows
    ]

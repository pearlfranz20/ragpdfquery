import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.graph import answer_question
from app.db import get_db
from app.models import QueryLog
from app.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()

    result = answer_question(
        db,
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
        language_override=request.language_override,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    log = QueryLog(
        question=request.question,
        detected_language=result.get("language"),
        answer=result.get("generation"),
        citations=result.get("citations", []),
        retrieved_chunk_ids=[c.chunk_id for c in result.get("retrieved", [])],
        faithfulness_score=result.get("faithfulness_score"),
        retried=result.get("retried", False),
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()

    return QueryResponse(
        question=request.question,
        answer=result.get("generation", ""),
        detected_language=result.get("language", "en"),
        citations=result.get("citations", []),
        faithfulness_score=result.get("faithfulness_score"),
        retried=result.get("retried", False),
        latency_ms=latency_ms,
    )

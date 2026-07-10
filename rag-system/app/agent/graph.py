from __future__ import annotations

import logging
import uuid
from typing import Any, TypedDict

from langdetect import LangDetectException, detect
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent.llm import get_chat_provider
from app.agent.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_TEMPLATE,
    STRICT_RETRY_SYSTEM_SUFFIX,
    format_excerpts,
)
from app.config import get_settings
from app.ingestion.embeddings import get_embedding_provider
from app.retrieval.vector_store import RetrievedChunk, similarity_search

logger = logging.getLogger(__name__)
settings = get_settings()

_LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "zh-cn": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "ru": "Russian",
}


class RagState(TypedDict, total=False):
    question: str
    language: str
    top_k: int
    document_ids: list[uuid.UUID] | None
    retrieved: list[RetrievedChunk]
    graded: list[RetrievedChunk]
    generation: str
    faithfulness_score: float | None
    retry_count: int
    retried: bool
    citations: list[dict[str, Any]]


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.lower(), code)


def build_graph(db: Session):
    chat = get_chat_provider()
    embedder = get_embedding_provider()

    def detect_language_node(state: RagState) -> RagState:
        try:
            code = state.get("language") or detect(state["question"])
        except LangDetectException:
            code = "en"
        return {"language": code}

    def retrieve_node(state: RagState) -> RagState:
        query_vec = embedder.embed([state["question"]])[0]
        retrieved = similarity_search(
            db,
            query_vec,
            top_k=state.get("top_k") or settings.retrieval_top_k,
            document_ids=state.get("document_ids"),
        )
        return {"retrieved": retrieved}

    def grade_node(state: RagState) -> RagState:
        """Lightweight relevance filter: drop chunks the LLM judges irrelevant.
        Runs only if we have more than a couple of candidates — skips the cost
        when there's little to filter."""
        retrieved = state["retrieved"]
        if len(retrieved) <= 2:
            return {"graded": retrieved}

        graded: list[RetrievedChunk] = []
        for chunk in retrieved:
            verdict = chat.complete(
                GRADE_SYSTEM_PROMPT,
                GRADE_USER_TEMPLATE.format(question=state["question"], passage=chunk.text[:1500]),
                max_tokens=5,
                temperature=0.0,
            )
            if verdict.strip().lower().startswith("y"):
                graded.append(chunk)

        # Never fall through to zero context if grading was overly strict —
        # better to answer with the top similarity hits than to refuse.
        return {"graded": graded or retrieved[:3]}

    def generate_node(state: RagState) -> RagState:
        chunks = state.get("graded") or state.get("retrieved") or []
        if not chunks:
            return {
                "generation": "I couldn't find any relevant information in the document corpus to answer this question.",
                "faithfulness_score": None,
            }

        excerpts = format_excerpts(chunks)
        system = GENERATE_SYSTEM_PROMPT.format(language=_lang_name(state["language"]))
        if state.get("retry_count", 0) > 0:
            system += STRICT_RETRY_SYSTEM_SUFFIX

        answer = chat.complete(
            system,
            GENERATE_USER_TEMPLATE.format(question=state["question"], excerpts=excerpts),
            max_tokens=1024,
            temperature=0.1,
        )
        return {"generation": answer}

    def faithfulness_node(state: RagState) -> RagState:
        if not settings.enable_inline_faithfulness_check:
            return {"faithfulness_score": None}

        chunks = state.get("graded") or state.get("retrieved") or []
        if not chunks or not state.get("generation"):
            return {"faithfulness_score": None}

        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase

            metric = FaithfulnessMetric(threshold=settings.faithfulness_threshold, model=settings.deepeval_model, include_reason=False)
            test_case = LLMTestCase(
                input=state["question"],
                actual_output=state["generation"],
                retrieval_context=[c.text for c in chunks],
            )
            metric.measure(test_case)
            return {"faithfulness_score": float(metric.score)}
        except Exception as e:  # noqa: BLE001
            logger.warning("Inline faithfulness check failed (%s); skipping", e)
            return {"faithfulness_score": None}

    def should_retry(state: RagState) -> str:
        score = state.get("faithfulness_score")
        retry_count = state.get("retry_count", 0)
        if score is not None and score < settings.faithfulness_threshold and retry_count < 1:
            return "retry"
        return "finish"

    def bump_retry_node(state: RagState) -> RagState:
        return {"retry_count": state.get("retry_count", 0) + 1, "retried": True}

    def format_citations_node(state: RagState) -> RagState:
        chunks = state.get("graded") or state.get("retrieved") or []
        citations = [
            {
                "document_id": str(c.document_id),
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_id": str(c.chunk_id),
                "snippet": c.text[:300],
            }
            for c in chunks
        ]
        return {"citations": citations, "retried": state.get("retried", False)}

    graph = StateGraph(RagState)
    graph.add_node("detect_language", detect_language_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_faithfulness", faithfulness_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("format_citations", format_citations_node)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", "check_faithfulness")
    graph.add_conditional_edges(
        "check_faithfulness", should_retry, {"retry": "bump_retry", "finish": "format_citations"}
    )
    graph.add_edge("bump_retry", "generate")
    graph.add_edge("format_citations", END)

    return graph.compile()


def answer_question(
    db: Session,
    question: str,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    language_override: str | None = None,
) -> RagState:
    app = build_graph(db)
    initial: RagState = {
        "question": question,
        "top_k": top_k or settings.retrieval_top_k,
        "document_ids": document_ids,
        "retry_count": 0,
        "retried": False,
    }
    if language_override:
        initial["language"] = language_override
    return app.invoke(initial)

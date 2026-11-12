"""
Offline evaluation harness. Runs every test case in the golden dataset
through the live RAG agent, scores the result with DeepEval metrics, and
persists results to the `eval_runs` table so the admin dashboard can chart
trends over time.

Run via `python scripts/run_eval.py`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.agent.graph import answer_question
from app.config import get_settings
from app.models import EvalRun

logger = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"


@dataclass
class GoldenCase:
    name: str
    question: str
    expected_answer: str
    document_ids: list[str] | None = None


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenCase]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(GoldenCase(**data))
    return cases


def run_eval_suite(db: Session, dataset_path: Path = DEFAULT_DATASET_PATH) -> list[EvalRun]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    cases = load_golden_dataset(dataset_path)
    metrics = [
        FaithfulnessMetric(threshold=0.6, model=settings.deepeval_model),
        AnswerRelevancyMetric(threshold=0.6, model=settings.deepeval_model),
        ContextualPrecisionMetric(threshold=0.6, model=settings.deepeval_model),
        ContextualRecallMetric(threshold=0.6, model=settings.deepeval_model),
        HallucinationMetric(threshold=0.4, model=settings.deepeval_model),
    ]

    results: list[EvalRun] = []

    for case in cases:
        logger.info("Evaluating case: %s", case.name)
        try:
            agent_result = answer_question(
                db,
                question=case.question,
                document_ids=[i for i in (case.document_ids or [])] or None,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent failed on case %s", case.name)
            for metric in metrics:
                results.append(
                    EvalRun(
                        metric_name=type(metric).__name__,
                        test_case_name=case.name,
                        score=0.0,
                        passed=False,
                        threshold=getattr(metric, "threshold", None),
                        reason=f"Agent execution failed: {e}",
                    )
                )
            continue

        retrieved_context = [c.text for c in agent_result.get("graded") or agent_result.get("retrieved") or []]

        test_case = LLMTestCase(
            input=case.question,
            actual_output=agent_result.get("generation", ""),
            expected_output=case.expected_answer,
            retrieval_context=retrieved_context,
            context=retrieved_context,
        )

        for metric in metrics:
            try:
                metric.measure(test_case)
                score = float(metric.score)
                passed = bool(metric.is_successful())
                reason = getattr(metric, "reason", None)
            except Exception as e:  # noqa: BLE001
                logger.warning("Metric %s failed on case %s: %s", type(metric).__name__, case.name, e)
                score, passed, reason = 0.0, False, str(e)

            results.append(
                EvalRun(
                    metric_name=type(metric).__name__,
                    test_case_name=case.name,
                    score=score,
                    passed=passed,
                    threshold=getattr(metric, "threshold", None),
                    reason=reason,
                    metadata_={"question": case.question, "answer": agent_result.get("generation", "")},
                )
            )

    db.add_all(results)
    db.commit()
    logger.info("Eval suite complete: %d metric results across %d cases", len(results), len(cases))
    return results

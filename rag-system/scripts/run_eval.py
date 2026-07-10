#!/usr/bin/env python
"""
Run the DeepEval offline evaluation suite against app/eval/golden_dataset.jsonl
and persist results to the eval_runs table.

Usage:
    python scripts/run_eval.py [--dataset path/to/custom.jsonl]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import session_scope  # noqa: E402
from app.eval.deepeval_suite import DEFAULT_DATASET_PATH, run_eval_suite  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    args = parser.parse_args()

    with session_scope() as db:
        results = run_eval_suite(db, dataset_path=Path(args.dataset))

    by_metric: dict[str, list[float]] = {}
    for r in results:
        by_metric.setdefault(r.metric_name, []).append(r.score or 0.0)

    print("\n=== Eval Summary ===")
    for metric, scores in by_metric.items():
        avg = sum(scores) / len(scores)
        print(f"{metric:30s} avg={avg:.3f}  n={len(scores)}")


if __name__ == "__main__":
    main()

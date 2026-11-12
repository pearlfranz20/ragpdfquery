#!/usr/bin/env python
"""
Generate a real golden evaluation dataset from your already-ingested corpus,
instead of hand-writing test questions.

How it works:
1. Pulls chunks for a sample of completed documents out of Postgres.
2. Groups each document's chunks into "contexts" (small groups of adjacent
   chunks) — DeepEval generates one or more questions per context.
3. Runs DeepEval's Synthesizer over those contexts to produce grounded
   (input, expected_output) pairs — the expected_output is generated FROM
   the context, so it's an honest paraphrase of what's actually in your
   documents, not a hallucinated guess.
4. Writes results to a .jsonl file in the same format app/eval/golden_dataset.jsonl
   uses, including which document_id each case is grounded in (so the eval
   run can optionally scope retrieval to that one document to sanity check
   your retriever, not just the generator).

Usage:
    python scripts/generate_golden_dataset.py \\
        --num-documents 20 \\
        --contexts-per-document 3 \\
        --goldens-per-context 2 \\
        --output app/eval/golden_dataset_generated.jsonl

Requires OPENAI_API_KEY (or your configured DEEPEVAL_MODEL provider) since
the Synthesizer itself calls an LLM to write the questions.

After generating, SPOT-CHECK the output before trusting it for eval — the
Synthesizer is good but not infallible, especially on dense tables or
boilerplate legal text. Delete or edit any case that doesn't read as a
question a real user would actually ask.
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import session_scope  # noqa: E402
from app.models import Chunk, Document  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_golden_dataset")


def slugify(text: str, max_len: int = 40) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len] or "case"


def build_contexts(num_documents: int, contexts_per_document: int, chunks_per_context: int, seed: int):
    """Returns list of (document_id, filename, context_texts) tuples."""
    random.seed(seed)
    results = []
    with session_scope() as db:
        documents = (
            db.query(Document)
            .filter(Document.status == "completed")
            .order_by(Document.created_at.desc())
            .limit(500)
            .all()
        )
        if not documents:
            logger.error("No completed documents found. Ingest some PDFs first.")
            sys.exit(1)

        sample_docs = random.sample(documents, min(num_documents, len(documents)))

        for doc in sample_docs:
            chunks = (
                db.query(Chunk)
                .filter(Chunk.document_id == doc.id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            if len(chunks) < chunks_per_context:
                continue

            # Slide a small window over the chunk sequence to build a few
            # coherent multi-chunk contexts per document.
            max_start = len(chunks) - chunks_per_context
            starts = sorted(random.sample(range(max_start + 1), min(contexts_per_document, max_start + 1)))
            for start in starts:
                window = chunks[start : start + chunks_per_context]
                results.append((str(doc.id), doc.filename, [c.text for c in window]))

    logger.info("Built %d contexts from %d documents", len(results), len(sample_docs))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-documents", type=int, default=20)
    parser.add_argument("--contexts-per-document", type=int, default=3)
    parser.add_argument("--chunks-per-context", type=int, default=3)
    parser.add_argument("--goldens-per-context", type=int, default=2)
    parser.add_argument("--output", type=str, default="app/eval/golden_dataset_generated.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from deepeval.synthesizer import Synthesizer

    contexts = build_contexts(args.num_documents, args.contexts_per_document, args.chunks_per_context, args.seed)
    if not contexts:
        logger.error("No contexts built — do your documents have enough chunks? Try lowering --chunks-per-context.")
        sys.exit(1)

    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[ctx for _, _, ctx in contexts],
        include_expected_output=True,
        max_goldens_per_context=args.goldens_per_context,
    )

    # Map generated goldens back to the document they came from so we can
    # record document_ids in the output (contexts list order is preserved).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w") as f:
        for i, golden in enumerate(goldens):
            # golden.context is the exact context list it was generated from;
            # match it back to (document_id, filename) by content.
            doc_id, filename = None, None
            for d_id, fname, ctx in contexts:
                if golden.context == ctx:
                    doc_id, filename = d_id, fname
                    break

            record = {
                "name": f"generated_{slugify(filename or 'doc')}_{i}",
                "question": golden.input,
                "expected_answer": golden.expected_output or "",
                "document_ids": [doc_id] if doc_id else [],
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    logger.info("Wrote %d generated golden test cases to %s", written, out_path)
    logger.info("Spot-check the file, then merge the ones you trust into app/eval/golden_dataset.jsonl")


if __name__ == "__main__":
    main()

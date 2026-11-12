#!/usr/bin/env python
"""
Bulk-ingest every PDF in a directory (recursively).

Usage:
    python scripts/ingest_folder.py /path/to/pdfs --workers 8 --strategy recursive

Safe to re-run: ingestion is idempotent per file content hash, so already
-completed files are skipped and only new/changed/failed files are (re)processed.
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.pipeline import ingest_pdf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_folder")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Directory containing PDFs (searched recursively)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--strategy", type=str, default=None, choices=[None, "recursive", "page", "section", "semantic"])
    args = parser.parse_args()

    pdf_paths = sorted(Path(args.folder).rglob("*.pdf"))
    if not pdf_paths:
        logger.error("No PDFs found under %s", args.folder)
        sys.exit(1)

    logger.info("Found %d PDFs. Ingesting with %d workers...", len(pdf_paths), args.workers)

    succeeded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ingest_pdf, str(p), p.name, args.strategy): p for p in pdf_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
                succeeded += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.error("Failed: %s (%s)", path, e)

    logger.info("Done. %d succeeded, %d failed out of %d total.", succeeded, failed, len(pdf_paths))
    sys.exit(0 if failed == 0 else 2)


if __name__ == "__main__":
    main()

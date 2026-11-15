"""
Extracts per-page text (and light layout signal) from a PDF.

Strategy:
1. Try native text extraction with PyMuPDF (fast, free, works for the large
   majority of "born-digital" PDFs).
2. For any page that comes back empty/near-empty (a strong signal it's a
   scanned image), fall back to OCR:
   - AWS Textract if `settings.use_textract` is true and credentials work
     (handles scans, tables, multi-column layouts well).
   - Otherwise local `pytesseract` OCR on a rasterized page image, so the
     pipeline still works with zero cloud dependency for dev/testing.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MIN_CHARS_TO_SKIP_OCR = 20  # pages with fewer native chars than this are treated as scans


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str
    used_ocr: bool


def extract_pdf(pdf_path: str) -> list[PageContent]:
    doc = fitz.open(pdf_path)
    pages: list[PageContent] = []

    for i, page in enumerate(doc, start=1):
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_CHARS_TO_SKIP_OCR:
            pages.append(PageContent(page_number=i, text=native_text, used_ocr=False))
            continue

        # Likely a scanned page — OCR it.
        logger.info("Page %d of %s looks scanned; running OCR", i, pdf_path)
        image = _rasterize_page(page)
        ocr_text = (
            _ocr_with_textract(image)
            if settings.use_textract
            else _ocr_with_tesseract(image)
        )
        pages.append(PageContent(page_number=i, text=ocr_text.strip(), used_ocr=True))

    doc.close()
    return pages


def _rasterize_page(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _ocr_with_tesseract(image: Image.Image) -> str:
    import pytesseract

    return pytesseract.image_to_string(image)


def _ocr_with_textract(image: Image.Image) -> str:
    """
    Calls AWS Textract's synchronous DetectDocumentText API on a single
    rasterized page. Falls back to local OCR if Textract isn't reachable
    (e.g. missing credentials in a dev environment), so ingestion never
    hard-fails just because AWS isn't configured.
    """
    try:
        import boto3

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        client = boto3.client("textract", region_name=settings.aws_region)
        response = client.detect_document_text(Document={"Bytes": buf.getvalue()})

        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        logger.warning("Textract call failed (%s); falling back to local OCR", e)
        return _ocr_with_tesseract(image)

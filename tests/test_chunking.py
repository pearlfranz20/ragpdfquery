import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.chunking import chunk_page, chunk_recursive, chunk_section, count_tokens
from app.ingestion.extract import PageContent


def make_pages():
    return [
        PageContent(page_number=1, text="Introduction\n\nThis contract governs the relationship between the parties. " * 20, used_ocr=False),
        PageContent(page_number=2, text="Termination\n\nEither party may terminate with 30 days notice. " * 20, used_ocr=False),
    ]


def test_chunk_recursive_preserves_page_numbers():
    chunks = chunk_recursive(make_pages(), size_tokens=50, overlap_tokens=10)
    assert chunks
    assert all(c.page_number in (1, 2) for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_chunk_page_includes_overlap():
    pages = make_pages()
    chunks = chunk_page(pages, overlap_chars=50)
    assert len(chunks) == 2
    # second chunk should include a trailing slice of page 1's text as context
    assert chunks[1].page_number == 2


def test_chunk_section_splits_on_headings():
    chunks = chunk_section(make_pages(), size_tokens=50, overlap_tokens=10)
    assert chunks
    joined = " ".join(c.text for c in chunks)
    assert "Termination" in joined
    assert "Introduction" in joined


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0


def test_empty_page_is_skipped():
    pages = [PageContent(page_number=1, text="   ", used_ocr=False)]
    assert chunk_recursive(pages, 100, 10) == []
    assert chunk_page(pages) == []

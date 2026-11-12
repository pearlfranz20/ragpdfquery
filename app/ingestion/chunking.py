"""
Chunking strategies. Each strategy takes the page-level extraction output
and returns a flat list of Chunk dataclasses carrying enough metadata
(page_number, char_offset) to cite back to an exact source location.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.extract import PageContent

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class RawChunk:
    text: str
    page_number: int
    char_offset: int
    chunk_index: int


# --------------------------------------------------------------------------
# Strategy: recursive character splitting (default, good general purpose)
# --------------------------------------------------------------------------
def chunk_recursive(pages: list[PageContent], size_tokens: int, overlap_tokens: int) -> list[RawChunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=size_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[RawChunk] = []
    idx = 0
    for page in pages:
        if not page.text.strip():
            continue
        pieces = splitter.split_text(page.text)
        offset = 0
        for piece in pieces:
            found = page.text.find(piece, offset)
            char_offset = found if found != -1 else offset
            chunks.append(RawChunk(text=piece, page_number=page.page_number, char_offset=char_offset, chunk_index=idx))
            offset = char_offset + len(piece)
            idx += 1
    return chunks


# --------------------------------------------------------------------------
# Strategy: one chunk per page, with a little overlap into neighbors.
# Best when page-level citation granularity matters more than topical
# coherence (e.g. contracts where "see page 12" needs to be exact).
# --------------------------------------------------------------------------
def chunk_page(pages: list[PageContent], overlap_chars: int = 200) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for idx, page in enumerate(pages):
        text = page.text.strip()
        if not text:
            continue
        prefix = ""
        if idx > 0 and pages[idx - 1].text.strip():
            prefix = pages[idx - 1].text.strip()[-overlap_chars:] + "\n"
        chunks.append(RawChunk(text=prefix + text, page_number=page.page_number, char_offset=0, chunk_index=idx))
    return chunks


# --------------------------------------------------------------------------
# Strategy: section/heading-aware. Detects heading-like lines (all-caps,
# numbered, or Title Case short lines followed by a blank line) and splits
# there first; any resulting section still too large is recursively split.
# --------------------------------------------------------------------------
_HEADING_RE = re.compile(
    r"^(?:\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,\-:]{2,80}$"
)


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.isupper() and len(line.split()) <= 12:
        return True
    return bool(_HEADING_RE.match(line))


def chunk_section(pages: list[PageContent], size_tokens: int, overlap_tokens: int) -> list[RawChunk]:
    sub_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=size_tokens, chunk_overlap=overlap_tokens
    )

    chunks: list[RawChunk] = []
    idx = 0
    for page in pages:
        lines = page.text.split("\n")
        sections: list[str] = []
        current: list[str] = []
        for line in lines:
            if _looks_like_heading(line) and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current))

        offset = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if count_tokens(section) <= size_tokens:
                pieces = [section]
            else:
                pieces = sub_splitter.split_text(section)
            for piece in pieces:
                found = page.text.find(piece, offset)
                char_offset = found if found != -1 else offset
                chunks.append(RawChunk(text=piece, page_number=page.page_number, char_offset=char_offset, chunk_index=idx))
                offset = char_offset + len(piece)
                idx += 1
    return chunks


# --------------------------------------------------------------------------
# Strategy: semantic breakpoint splitting. Embeds consecutive sentences and
# splits where cosine similarity between neighbors drops sharply — i.e. at
# genuine topic shifts rather than arbitrary character counts. Costs one
# embedding call per sentence batch, so it's the most expensive strategy;
# use for high-value corpora where retrieval precision matters most.
# --------------------------------------------------------------------------
def chunk_semantic(
    pages: list[PageContent],
    embed_fn,  # Callable[[list[str]], list[list[float]]]
    max_tokens: int = 500,
    breakpoint_percentile: float = 0.85,
) -> list[RawChunk]:
    import numpy as np

    chunks: list[RawChunk] = []
    idx = 0
    for page in pages:
        sentences = re.split(r"(?<=[.!?])\s+", page.text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            continue
        if len(sentences) == 1:
            chunks.append(RawChunk(text=sentences[0], page_number=page.page_number, char_offset=0, chunk_index=idx))
            idx += 1
            continue

        embeddings = np.array(embed_fn(sentences))
        sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]) /
                  (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8))
            for i in range(len(embeddings) - 1)
        ]
        threshold = float(np.percentile(sims, (1 - breakpoint_percentile) * 100))
        breakpoints = {i for i, s in enumerate(sims) if s < threshold}

        current: list[str] = []
        current_tokens = 0
        offset = 0
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            current_tokens += count_tokens(sentence)
            should_break = i in breakpoints or current_tokens >= max_tokens
            if should_break and current:
                text = " ".join(current)
                found = page.text.find(current[0], offset)
                char_offset = found if found != -1 else offset
                chunks.append(RawChunk(text=text, page_number=page.page_number, char_offset=char_offset, chunk_index=idx))
                offset = char_offset + len(text)
                idx += 1
                current, current_tokens = [], 0
        if current:
            text = " ".join(current)
            chunks.append(RawChunk(text=text, page_number=page.page_number, char_offset=offset, chunk_index=idx))
            idx += 1
    return chunks


STRATEGIES = {
    "recursive": chunk_recursive,
    "page": chunk_page,
    "section": chunk_section,
    "semantic": chunk_semantic,  # requires embed_fn, wired specially in pipeline.py
}

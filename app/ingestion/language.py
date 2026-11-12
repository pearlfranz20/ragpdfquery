from langdetect import DetectorFactory, LangDetectException, detect

# Make detection deterministic (langdetect is otherwise seeded randomly)
DetectorFactory.seed = 0


def detect_language(text: str, default: str = "en") -> str:
    text = text.strip()
    if len(text) < 20:
        return default
    try:
        return detect(text)
    except LangDetectException:
        return default


def detect_document_language(page_texts: list[str]) -> str:
    """Concatenate a sample of pages and detect the dominant language."""
    sample = "\n".join(page_texts[:5])[:5000]
    return detect_language(sample)

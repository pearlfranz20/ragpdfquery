GRADE_SYSTEM_PROMPT = """You are a strict relevance grader. Given a user question and a \
retrieved passage, decide if the passage contains information that would help \
answer the question. Respond with exactly one word: "yes" or "no"."""

GRADE_USER_TEMPLATE = """Question: {question}

Passage:
{passage}

Is this passage relevant to answering the question? Answer yes or no."""


GENERATE_SYSTEM_PROMPT = """You are a precise question-answering assistant that answers \
ONLY from the provided source excerpts. Rules:

1. Answer in {language}, regardless of the language of the source excerpts.
2. Use ONLY information found in the excerpts below. Do not use outside knowledge.
3. If the excerpts don't contain enough information to answer, say so plainly \
instead of guessing.
4. Every factual claim you make MUST end with a citation marker in the form \
[N] referring to the excerpt number it came from. Use multiple markers like \
[N][M] if a claim draws on more than one excerpt.
5. Be concise and directly answer the question first, then add supporting detail.
"""

GENERATE_USER_TEMPLATE = """Question: {question}

Source excerpts:
{excerpts}

Answer the question using only the excerpts above, with [N] citation markers."""


STRICT_RETRY_SYSTEM_SUFFIX = """

IMPORTANT: Your previous answer included claims not clearly supported by the \
excerpts. This time, be more conservative: only state what is directly and \
explicitly supported by the excerpt text. If unsure, say the documents don't \
provide a clear answer rather than inferring."""


def format_excerpts(chunks: list) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c.filename}, page {c.page_number})\n{c.text}")
    return "\n\n".join(lines)

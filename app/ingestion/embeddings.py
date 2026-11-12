"""
Embedding provider abstraction. Both providers expose multilingual-capable
models, so the same vector space can be searched regardless of the source
document's language.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    @property
    def dimension(self) -> int:
        return settings.openai_embedding_dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI batches up to a few thousand inputs; chunk defensively.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out


class BedrockEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.bedrock_embedding_model_id

    @property
    def dimension(self) -> int:
        return settings.bedrock_embedding_dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text[:8000]})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan embeddings are invoked one text at a time; parallelize lightly.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(self._embed_one, texts))


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "bedrock":
        return BedrockEmbeddingProvider()
    return OpenAIEmbeddingProvider()

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str: ...


class OpenAIChatProvider(ChatProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15))
    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class BedrockChatProvider(ChatProvider):
    def __init__(self):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.bedrock_chat_model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15))
    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        # Anthropic-on-Bedrock message format
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        )
        response = self.client.invoke_model(
            modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
        )
        payload = json.loads(response["body"].read())
        return "".join(block["text"] for block in payload.get("content", []) if block["type"] == "text")


def get_chat_provider() -> ChatProvider:
    if settings.llm_provider == "bedrock":
        return BedrockChatProvider()
    return OpenAIChatProvider()

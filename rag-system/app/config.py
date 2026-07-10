from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # Providers
    llm_provider: str = "openai"          # "openai" | "bedrock"
    embedding_provider: str = "openai"    # "openai" | "bedrock"

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dim: int = 3072

    # Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_chat_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_embedding_dim: int = 1024

    # OCR
    use_textract: bool = False
    textract_s3_bucket: str = ""

    # Ingestion
    chunking_strategy: str = "recursive"  # recursive | section | page | semantic
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75
    ingest_max_workers: int = 8

    # Retrieval
    retrieval_top_k: int = 8
    retrieval_min_similarity: float = 0.25

    # Hallucination / eval
    enable_inline_faithfulness_check: bool = True
    faithfulness_threshold: float = 0.6
    deepeval_model: str = "gpt-4o-mini"

    log_level: str = "INFO"

    @property
    def embedding_dim(self) -> int:
        return (
            self.openai_embedding_dim
            if self.embedding_provider == "openai"
            else self.bedrock_embedding_dim
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant.apps-prod.svc.cluster.local:6333"
    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimensions: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 64
    api_token: str = ""

    model_config = {"env_prefix": "RAG_"}


settings = Settings()

COLLECTIONS = [
    "conversations",
    "research",
    "operations",
    "code_knowledge",
    "documents",
]

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant.apps-prod.svc.cluster.local:6333"
    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    embedding_provider: str = "fireworks"  # "fireworks" or "local"
    ollama_url: str = "http://ollama.apps-prod.svc.cluster.local:11434"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    ollama_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 64
    api_token: str = ""
    hybrid_search: bool = True

    model_config = {"env_prefix": "RAG_"}


settings = Settings()

COLLECTIONS = [
    "conversations",
    "research",
    "operations",
    "code_knowledge",
    "documents",
]

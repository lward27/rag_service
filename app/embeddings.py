import httpx

from app.config import settings


async def _fireworks_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.fireworks_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.fireworks_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": settings.embedding_model, "input": text},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def _fireworks_embeddings_batch(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.fireworks_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.fireworks_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": settings.embedding_model, "input": texts},
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


async def _ollama_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.ollama_model, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


async def _ollama_embeddings_batch(texts: list[str]) -> list[list[float]]:
    # Ollama's /api/embed supports batch input
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.ollama_model, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector using the configured provider."""
    if settings.embedding_provider == "local":
        return await _ollama_embedding(text)
    return await _fireworks_embedding(text)


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts using the configured provider."""
    if not texts:
        return []
    if settings.embedding_provider == "local":
        return await _ollama_embeddings_batch(texts)
    return await _fireworks_embeddings_batch(texts)

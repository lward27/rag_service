import httpx

from app.config import settings


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Fireworks API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.fireworks_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.fireworks_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single API call."""
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.fireworks_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.fireworks_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

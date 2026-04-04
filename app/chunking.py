from app.config import settings


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks by token approximation.

    Uses whitespace splitting as a rough token approximation (~1 token per word).
    Chunks overlap to preserve context across boundaries.
    """
    words = text.split()
    if len(words) <= settings.chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + settings.chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += settings.chunk_size - settings.chunk_overlap

    return chunks

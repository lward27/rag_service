from fastembed import SparseTextEmbedding

_model = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _model


def generate_sparse_embedding(text: str) -> tuple[list[int], list[float]]:
    """Generate a BM25 sparse embedding for the given text."""
    model = _get_model()
    result = list(model.embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()


def generate_sparse_embeddings_batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Generate BM25 sparse embeddings for multiple texts."""
    if not texts:
        return []
    model = _get_model()
    results = list(model.embed(texts))
    return [(r.indices.tolist(), r.values.tolist()) for r in results]

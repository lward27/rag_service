import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.chunking import chunk_text
from app.config import COLLECTIONS, settings
from app.embeddings import generate_embedding, generate_embeddings_batch
from app.models import (
    BatchIndexRequest,
    BatchIndexResponse,
    CollectionStats,
    DeleteRequest,
    IndexRequest,
    IndexResponse,
    SearchAllRequest,
    SearchAllResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatusResponse,
)
from app.sparse import generate_sparse_embedding, generate_sparse_embeddings_batch

security = HTTPBearer(auto_error=False)
qdrant: QdrantClient = None

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _init_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=10)


def _ensure_collections(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name not in existing:
            _create_hybrid_collection(client, name)
        else:
            # Check if collection already has named vectors (hybrid)
            info = client.get_collection(name)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, VectorParams):
                # Old-style single vector — migrate to hybrid
                print(f"Migrating collection '{name}' to hybrid vectors...")
                _migrate_to_hybrid(client, name)


def _create_hybrid_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )


def _migrate_to_hybrid(client: QdrantClient, name: str) -> None:
    """Migrate a single-vector collection to named dense+sparse vectors.

    Backs up all points, recreates the collection with named vectors,
    and restores points with the original vector as the 'dense' named vector.
    Sparse vectors will be added on next re-index.
    """
    # Scroll all points
    all_points = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=name, limit=100, offset=offset,
            with_payload=True, with_vectors=True,
        )
        all_points.extend(results)
        if offset is None:
            break

    # Delete and recreate
    client.delete_collection(name)
    _create_hybrid_collection(client, name)

    # Restore points with named dense vector
    if all_points:
        points = []
        for p in all_points:
            vector = p.vector
            # Old points have a plain list vector; new ones have a dict
            if isinstance(vector, list):
                dense_vec = vector
            elif isinstance(vector, dict) and DENSE_VECTOR_NAME in vector:
                dense_vec = vector[DENSE_VECTOR_NAME]
            else:
                continue
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={DENSE_VECTOR_NAME: dense_vec},
                payload=p.payload,
            ))
        if points:
            client.upsert(collection_name=name, points=points)
    print(f"Migrated '{name}': {len(all_points)} points")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant
    qdrant = _init_qdrant()
    _ensure_collections(qdrant)
    yield
    qdrant.close()


app = FastAPI(
    title="RAG Service",
    description="Vector memory API for OpenClaw agents and Claude Code",
    version="2.0.0",
    lifespan=lifespan,
)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not settings.api_token:
        return
    if credentials is None or credentials.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _validate_collection(name: str) -> None:
    if name not in COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection '{name}'. Must be one of: {COLLECTIONS}",
        )


def _build_point(chunk: str, dense_vec: list[float], sparse_indices: list[int],
                 sparse_values: list[float], payload: dict) -> PointStruct:
    """Build a point with both dense and sparse vectors."""
    vectors = {DENSE_VECTOR_NAME: dense_vec}
    sparse = {}
    if sparse_indices:
        sparse[SPARSE_VECTOR_NAME] = SparseVector(
            indices=sparse_indices, values=sparse_values,
        )
    return PointStruct(
        id=str(uuid.uuid4()),
        vector={**vectors, **sparse},
        payload=payload,
    )


@app.post("/index", response_model=IndexResponse)
async def index_document(req: IndexRequest, _=Security(verify_token)):
    """Index a document by chunking, embedding, and storing in Qdrant."""
    _validate_collection(req.collection)

    chunks = chunk_text(req.content)
    dense_embeddings = await generate_embeddings_batch(chunks)
    sparse_embeddings = generate_sparse_embeddings_batch(chunks)

    doc_id = req.document_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    points = []
    for i, (chunk, dense, (sp_idx, sp_val)) in enumerate(
        zip(chunks, dense_embeddings, sparse_embeddings)
    ):
        payload = {
            "content": chunk,
            "document_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "indexed_at": now,
            **req.metadata,
        }
        points.append(_build_point(chunk, dense, sp_idx, sp_val, payload))

    qdrant.upsert(collection_name=req.collection, points=points)

    return IndexResponse(
        document_id=doc_id,
        collection=req.collection,
        chunks=len(chunks),
        message=f"Indexed {len(chunks)} chunk(s) (hybrid: dense+sparse)",
    )


def _hybrid_search(collection_name: str, query_dense: list[float],
                   query_sparse_indices: list[int], query_sparse_values: list[float],
                   query_filter, limit: int, score_threshold: float):
    """Perform hybrid search using Prefetch + RRF fusion."""
    prefetch = [
        Prefetch(
            query=query_dense,
            using=DENSE_VECTOR_NAME,
            limit=limit * 2,
            filter=query_filter,
        ),
    ]
    if query_sparse_indices:
        prefetch.append(
            Prefetch(
                query=SparseVector(indices=query_sparse_indices, values=query_sparse_values),
                using=SPARSE_VECTOR_NAME,
                limit=limit * 2,
                filter=query_filter,
            ),
        )

    return qdrant.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=Fusion.RRF,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )


def _dense_only_search(collection_name: str, query_dense: list[float],
                       query_filter, limit: int, score_threshold: float):
    """Fallback: dense-only search using named vector."""
    return qdrant.query_points(
        collection_name=collection_name,
        query=query_dense,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )


def _search_collection(collection_name: str, query_dense: list[float],
                       query_sparse, query_filter, limit: int,
                       score_threshold: float):
    """Search a collection with hybrid or dense-only depending on config."""
    if settings.hybrid_search and query_sparse:
        sp_idx, sp_val = query_sparse
        try:
            return _hybrid_search(
                collection_name, query_dense, sp_idx, sp_val,
                query_filter, limit, score_threshold,
            )
        except Exception:
            # Fallback if sparse index not ready (migrated points without sparse)
            return _dense_only_search(
                collection_name, query_dense, query_filter, limit, score_threshold,
            )
    return _dense_only_search(
        collection_name, query_dense, query_filter, limit, score_threshold,
    )


def _points_to_results(points, collection: str = "") -> list[SearchResult]:
    results = []
    for point in points:
        results.append(
            SearchResult(
                content=point.payload.get("content", ""),
                score=point.score,
                metadata={
                    k: v for k, v in point.payload.items()
                    if k not in ("content",)
                },
                document_id=point.payload.get("document_id", ""),
                collection=collection,
            )
        )
    return results


@app.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest, _=Security(verify_token)):
    """Hybrid semantic + keyword search across a collection."""
    _validate_collection(req.collection)

    query_dense = await generate_embedding(req.query)
    query_sparse = generate_sparse_embedding(req.query) if settings.hybrid_search else None

    query_filter = None
    if req.filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in req.filters.items()
        ]
        query_filter = Filter(must=conditions)

    results = _search_collection(
        req.collection, query_dense, query_sparse,
        query_filter, req.limit, req.score_threshold,
    )

    return SearchResponse(
        results=_points_to_results(results.points, req.collection),
        query=req.query,
        collection=req.collection,
        total=len(results.points),
    )


@app.post("/search-all", response_model=SearchAllResponse)
async def search_all_collections(req: SearchAllRequest, _=Security(verify_token)):
    """Hybrid search across all collections, merged by score."""
    query_dense = await generate_embedding(req.query)
    query_sparse = generate_sparse_embedding(req.query) if settings.hybrid_search else None

    query_filter = None
    if req.filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in req.filters.items()
        ]
        query_filter = Filter(must=conditions)

    all_results = []
    for collection_name in COLLECTIONS:
        try:
            results = _search_collection(
                collection_name, query_dense, query_sparse,
                query_filter, req.limit, req.score_threshold,
            )
            all_results.extend(_points_to_results(results.points, collection_name))
        except Exception:
            continue

    all_results.sort(key=lambda r: r.score, reverse=True)
    seen = set()
    deduped = []
    for r in all_results:
        if r.document_id not in seen:
            seen.add(r.document_id)
            deduped.append(r)

    return SearchAllResponse(
        results=deduped[: req.limit],
        query=req.query,
        collections_searched=COLLECTIONS,
        total=len(deduped[: req.limit]),
    )


@app.post("/batch", response_model=BatchIndexResponse)
async def batch_index(req: BatchIndexRequest, _=Security(verify_token)):
    """Bulk index multiple documents with hybrid vectors."""
    indexed = 0
    total_chunks = 0
    errors = []

    for i, doc in enumerate(req.documents):
        try:
            _validate_collection(doc.collection)
            chunks = chunk_text(doc.content)
            dense_embeddings = await generate_embeddings_batch(chunks)
            sparse_embeddings = generate_sparse_embeddings_batch(chunks)

            doc_id = doc.document_id or str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            points = []
            for j, (chunk, dense, (sp_idx, sp_val)) in enumerate(
                zip(chunks, dense_embeddings, sparse_embeddings)
            ):
                payload = {
                    "content": chunk,
                    "document_id": doc_id,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "indexed_at": now,
                    **doc.metadata,
                }
                points.append(_build_point(chunk, dense, sp_idx, sp_val, payload))

            qdrant.upsert(collection_name=doc.collection, points=points)
            indexed += 1
            total_chunks += len(chunks)
        except Exception as e:
            errors.append(f"Document {i}: {str(e)}")

    return BatchIndexResponse(
        indexed=indexed,
        total_chunks=total_chunks,
        errors=errors,
    )


@app.delete("/documents")
async def delete_document(req: DeleteRequest, _=Security(verify_token)):
    """Delete all chunks for a given document_id from a collection."""
    _validate_collection(req.collection)

    qdrant.delete(
        collection_name=req.collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=req.document_id)
                )
            ]
        ),
    )
    return {"message": f"Deleted document {req.document_id} from {req.collection}"}


@app.get("/indexed/{collection}")
async def list_indexed(collection: str, _=Security(verify_token)):
    """List all unique document_ids in a collection."""
    _validate_collection(collection)

    doc_ids = set()
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=["document_id"],
        )
        for point in results:
            doc_id = point.payload.get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
        if offset is None:
            break

    return {"collection": collection, "document_ids": sorted(doc_ids), "total": len(doc_ids)}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Health check and collection statistics."""
    try:
        collections_info = []
        for name in COLLECTIONS:
            try:
                info = qdrant.get_collection(name)
                collections_info.append(
                    CollectionStats(
                        name=name,
                        vectors_count=info.vectors_count or 0,
                        points_count=info.points_count or 0,
                    )
                )
            except Exception:
                collections_info.append(
                    CollectionStats(name=name, vectors_count=0, points_count=0)
                )

        return StatusResponse(
            status="healthy",
            qdrant_connected=True,
            collections=collections_info,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
        )
    except Exception:
        return StatusResponse(
            status="degraded",
            qdrant_connected=False,
            collections=[],
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}

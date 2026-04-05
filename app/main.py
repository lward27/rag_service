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
    MatchValue,
    PointStruct,
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

security = HTTPBearer(auto_error=False)
qdrant: QdrantClient = None


def _init_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=10)


def _ensure_collections(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )


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
    version="1.0.0",
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


@app.post("/index", response_model=IndexResponse)
async def index_document(req: IndexRequest, _=Security(verify_token)):
    """Index a document by chunking, embedding, and storing in Qdrant."""
    _validate_collection(req.collection)

    chunks = chunk_text(req.content)
    embeddings = await generate_embeddings_batch(chunks)

    doc_id = req.document_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        payload = {
            "content": chunk,
            "document_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "indexed_at": now,
            **req.metadata,
        }
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    qdrant.upsert(collection_name=req.collection, points=points)

    return IndexResponse(
        document_id=doc_id,
        collection=req.collection,
        chunks=len(chunks),
        message=f"Indexed {len(chunks)} chunk(s)",
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest, _=Security(verify_token)):
    """Semantic search across a collection."""
    _validate_collection(req.collection)

    query_embedding = await generate_embedding(req.query)

    query_filter = None
    if req.filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in req.filters.items()
        ]
        query_filter = Filter(must=conditions)

    results = qdrant.query_points(
        collection_name=req.collection,
        query=query_embedding,
        query_filter=query_filter,
        limit=req.limit,
        score_threshold=req.score_threshold,
        with_payload=True,
    )

    search_results = []
    for point in results.points:
        search_results.append(
            SearchResult(
                content=point.payload.get("content", ""),
                score=point.score,
                metadata={
                    k: v
                    for k, v in point.payload.items()
                    if k not in ("content",)
                },
                document_id=point.payload.get("document_id", ""),
            )
        )

    return SearchResponse(
        results=search_results,
        query=req.query,
        collection=req.collection,
        total=len(search_results),
    )


@app.post("/search-all", response_model=SearchAllResponse)
async def search_all_collections(req: SearchAllRequest, _=Security(verify_token)):
    """Search across all collections and return merged results sorted by score."""
    query_embedding = await generate_embedding(req.query)

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
            results = qdrant.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=req.limit,
                score_threshold=req.score_threshold,
                with_payload=True,
            )
            for point in results.points:
                all_results.append(
                    SearchResult(
                        content=point.payload.get("content", ""),
                        score=point.score,
                        metadata={
                            k: v
                            for k, v in point.payload.items()
                            if k not in ("content",)
                        },
                        document_id=point.payload.get("document_id", ""),
                        collection=collection_name,
                    )
                )
        except Exception:
            continue

    all_results.sort(key=lambda r: r.score, reverse=True)
    # Deduplicate by document_id, keeping highest score
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
    """Bulk index multiple documents."""
    indexed = 0
    total_chunks = 0
    errors = []

    for i, doc in enumerate(req.documents):
        try:
            _validate_collection(doc.collection)
            chunks = chunk_text(doc.content)
            embeddings = await generate_embeddings_batch(chunks)

            doc_id = doc.document_id or str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            points = []
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                payload = {
                    "content": chunk,
                    "document_id": doc_id,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "indexed_at": now,
                    **doc.metadata,
                }
                points.append(
                    PointStruct(id=point_id, vector=embedding, payload=payload)
                )

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
    """List all unique document_ids in a collection. Used for diff-based indexing."""
    _validate_collection(collection)

    # Scroll through all points and collect unique document_ids
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

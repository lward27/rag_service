from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class IndexRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Text content to index")
    collection: str = Field(..., description="Target collection name")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    document_id: Optional[str] = Field(None, description="Optional ID for upsert behavior")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection: str = Field(..., description="Collection to search")
    limit: int = Field(default=5, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: dict = Field(default_factory=dict, description="Metadata filters")


class BatchIndexRequest(BaseModel):
    documents: list[IndexRequest] = Field(..., min_length=1, description="Documents to index")


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict
    document_id: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    collection: str
    total: int


class IndexResponse(BaseModel):
    document_id: str
    collection: str
    chunks: int
    message: str


class BatchIndexResponse(BaseModel):
    indexed: int
    total_chunks: int
    errors: list[str]


class CollectionStats(BaseModel):
    name: str
    vectors_count: int
    points_count: int


class StatusResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collections: list[CollectionStats]
    embedding_model: str


class DeleteRequest(BaseModel):
    document_id: str
    collection: str

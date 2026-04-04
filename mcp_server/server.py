"""MCP server that provides RAG vector memory tools for Claude Code."""

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

RAG_API_URL = os.environ.get("RAG_API_URL", "https://rag.lucas.engineering")
RAG_API_TOKEN = os.environ.get("RAG_API_TOKEN", "")

mcp = FastMCP("rag-memory")


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if RAG_API_TOKEN:
        h["Authorization"] = f"Bearer {RAG_API_TOKEN}"
    return h


def _url(path: str) -> str:
    return f"{RAG_API_URL.rstrip('/')}{path}"


@mcp.tool()
async def rag_search(
    query: str,
    collection: str = "research",
    limit: int = 5,
    score_threshold: float = 0.3,
    filters: dict | None = None,
) -> str:
    """Search the RAG vector memory for semantically similar content.

    Collections: conversations, research, operations, code_knowledge, documents

    Args:
        query: Natural language search query
        collection: Which collection to search (default: research)
        limit: Max results to return (1-50, default: 5)
        score_threshold: Minimum similarity score (0.0-1.0, default: 0.3)
        filters: Optional metadata filters, e.g. {"agent_id": "scout", "topic": "qdrant"}
    """
    body = {
        "query": query,
        "collection": collection,
        "limit": limit,
        "score_threshold": score_threshold,
    }
    if filters:
        body["filters"] = filters

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_url("/search"), headers=_headers(), json=body)
        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"
        data = resp.json()

    if not data.get("results"):
        return f"No results found in '{collection}' for: {query}"

    lines = [f"Found {data['total']} result(s) in '{collection}':\n"]
    for i, r in enumerate(data["results"], 1):
        score = f"{r['score']:.3f}"
        meta = r.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k not in ("content", "document_id", "chunk_index", "total_chunks"))
        lines.append(f"--- Result {i} (score: {score}) ---")
        if meta_str:
            lines.append(f"Metadata: {meta_str}")
        lines.append(r["content"])
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def rag_index(
    content: str,
    collection: str = "conversations",
    topic: str = "",
    source: str = "claude-code",
    document_id: str | None = None,
) -> str:
    """Store content in the RAG vector memory for future retrieval.

    Collections: conversations, research, operations, code_knowledge, documents

    Args:
        content: The text content to store (will be chunked and embedded automatically)
        collection: Which collection to store in (default: conversations)
        topic: Topic tag for metadata (e.g. "qdrant-setup", "auth-debugging")
        source: Source identifier (default: claude-code)
        document_id: Optional unique ID for upsert behavior
    """
    metadata = {
        "agent_id": "claude-code",
        "source": source,
    }
    if topic:
        metadata["topic"] = topic

    body = {
        "content": content,
        "collection": collection,
        "metadata": metadata,
    }
    if document_id:
        body["document_id"] = document_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_url("/index"), headers=_headers(), json=body)
        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"
        data = resp.json()

    return f"Indexed to '{collection}': {data['chunks']} chunk(s), doc_id={data['document_id']}"


@mcp.tool()
async def rag_search_all(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.3,
    filters: dict | None = None,
) -> str:
    """Search ALL RAG collections at once and return merged results sorted by relevance.

    This is the best tool for broad questions like "what do we know about X?" when you
    don't know which collection has the answer.

    Args:
        query: Natural language search query
        limit: Max total results to return (1-50, default: 5)
        score_threshold: Minimum similarity score (0.0-1.0, default: 0.3)
        filters: Optional metadata filters, e.g. {"agent_id": "scout"}
    """
    body = {
        "query": query,
        "limit": limit,
        "score_threshold": score_threshold,
    }
    if filters:
        body["filters"] = filters

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_url("/search-all"), headers=_headers(), json=body)
        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"
        data = resp.json()

    if not data.get("results"):
        return f"No results found across any collection for: {query}"

    lines = [f"Found {data['total']} result(s) across {', '.join(data['collections_searched'])}:\n"]
    for i, r in enumerate(data["results"], 1):
        score = f"{r['score']:.3f}"
        collection = r.get("collection", "unknown")
        meta = r.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k not in ("content", "document_id", "chunk_index", "total_chunks"))
        lines.append(f"--- Result {i} [{collection}] (score: {score}) ---")
        if meta_str:
            lines.append(f"Metadata: {meta_str}")
        lines.append(r["content"])
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def rag_status() -> str:
    """Check RAG system health and collection statistics."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(_url("/status"), headers=_headers())
        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"
        data = resp.json()

    lines = [
        f"Status: {data['status']}",
        f"Qdrant: {'connected' if data['qdrant_connected'] else 'disconnected'}",
        f"Embedding model: {data['embedding_model']} ({data.get('embedding_provider', 'unknown')})",
        "",
        "Collections:",
    ]
    for c in data.get("collections", []):
        lines.append(f"  {c['name']}: {c['points_count']} points, {c['vectors_count']} vectors")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()

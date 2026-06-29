"""Database Models for RAG Explorer

These SQLModel tables are the source of truth for the relational metadata that
backs the RAG platform. The actual vectors live in ChromaDB; here we keep the
configuration (providers, chunking strategies, collections) and bookkeeping
(documents, chunks, queries, experiments, crawl jobs).

Note: ``metadata`` is reserved by SQLAlchemy's declarative base, so JSON
metadata columns are named ``*_metadata`` / ``*_json`` instead.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field


class EmbeddingProvider(SQLModel, table=True):
    """Configuration for an embedding model provider."""
    id: Optional[int] = Field(default=None, primary_key=True)
    key: Optional[str] = Field(default=None)
    name: str
    provider_type: str
    model_name: str
    dimension: int
    description: Optional[str] = None
    config_json: Optional[str] = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkingStrategy(SQLModel, table=True):
    """A configurable chunking strategy used during ingestion."""
    id: Optional[int] = Field(default=None, primary_key=True)
    key: Optional[str] = Field(default=None)
    name: str
    strategy_type: str
    chunk_size: int = 500
    overlap: int = 50
    description: Optional[str] = None
    config_json: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VectorCollection(SQLModel, table=True):
    """A vector database collection (knowledge base).

    The collection ``name`` is used directly as the ChromaDB collection name.
    Embedding/chunking choices are made per-ingestion, so the references here
    are optional defaults rather than hard requirements.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    description: Optional[str] = None
    embedding_provider_id: Optional[int] = Field(default=None, foreign_key="embeddingprovider.id")
    chunking_strategy_id: Optional[int] = Field(default=None, foreign_key="chunkingstrategy.id")
    chroma_collection_name: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    security_enabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class IndexedDocument(SQLModel, table=True):
    """A document that has been ingested into a vector collection."""
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="vectorcollection.id")
    embedding_provider_id: Optional[int] = Field(default=None, foreign_key="embeddingprovider.id")
    chunking_strategy_id: Optional[int] = Field(default=None, foreign_key="chunkingstrategy.id")
    filename: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    content_hash: Optional[str] = None
    source_type: str = "upload"
    source_url: Optional[str] = None
    status: str = "pending"
    chunk_count: int = 0
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(SQLModel, table=True):
    """An individual chunk of a document. The embedding lives in ChromaDB."""
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="indexeddocument.id")
    chunk_index: int
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None
    chroma_id: str
    chunk_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RAGExperiment(SQLModel, table=True):
    """An A/B test experiment comparing two RAG configurations."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    collection_id: int = Field(foreign_key="vectorcollection.id")
    config_a: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    config_b: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    status: str = "active"
    test_query_count: int = 0
    avg_mrr: Optional[float] = None
    avg_ndcg: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class RAGQuery(SQLModel, table=True):
    """A search query and its results, used for quality measurement."""
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: Optional[int] = Field(default=None, foreign_key="ragexperiment.id")
    collection_id: int = Field(foreign_key="vectorcollection.id")
    embedding_provider_id: Optional[int] = Field(default=None, foreign_key="embeddingprovider.id")
    query_text: str
    search_type: str = "semantic"
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    results: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    result_count: int = 0
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CrawlJob(SQLModel, table=True):
    """A web crawling job."""
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: Optional[int] = Field(default=None, foreign_key="vectorcollection.id")
    url: str
    crawl_type: str = "single"
    max_depth: int = 2
    max_pages: int = 100
    respect_robots: bool = True
    status: str = "queued"
    pages_crawled: int = 0
    pages_failed: int = 0
    result_summary: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

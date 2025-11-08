"""Database Models for RAG Explorer"""
from __future__ import annotations
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class EmbeddingProvider(SQLModel, table=True):
    """Configuration for embedding model providers"""
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True)
    name: str
    provider_type: str
    model_name: str
    dimension: int
    config_json: Optional[str] = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkingStrategy(SQLModel, table=True):
    """Configurable chunking strategies for experimentation"""
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True)
    name: str
    strategy_type: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    config_json: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VectorCollection(SQLModel, table=True):
    """Vector database collections (knowledge bases)"""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    description: Optional[str] = None
    embedding_provider_id: int = Field(foreign_key="embeddingprovider.id")
    chunking_strategy_id: int = Field(foreign_key="chunkingstrategy.id")
    chroma_collection_name: str
    document_count: int = 0
    chunk_count: int = 0
    metadata_schema: Optional[str] = None
    security_enabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class IndexedDocument(SQLModel, table=True):
    """Documents ingested into vector collections"""
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="vectorcollection.id")
    source_type: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    title: str
    content_hash: str
    chunk_count: int = 0
    metadata_json: Optional[str] = None
    security_level: Optional[str] = None
    tags: Optional[str] = None
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(SQLModel, table=True):
    """Individual chunks with embeddings stored in ChromaDB"""
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="indexeddocument.id")
    collection_id: int = Field(foreign_key="vectorcollection.id")
    chroma_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata_json: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RAGExperiment(SQLModel, table=True):
    """Experiment configurations for A/B testing"""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    collection_id: int = Field(foreign_key="vectorcollection.id")
    embedding_provider_id: int = Field(foreign_key="embeddingprovider.id")
    chunking_strategy_id: int = Field(foreign_key="chunkingstrategy.id")
    top_k: int = 5
    enable_reranking: bool = False
    reranking_model: Optional[str] = None
    enable_hybrid_search: bool = False
    hybrid_alpha: float = 0.5
    test_query_count: int = 0
    avg_mrr: Optional[float] = None
    avg_ndcg: Optional[float] = None
    avg_precision_at_k: Optional[float] = None
    avg_recall_at_k: Optional[float] = None
    config_json: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class RAGQuery(SQLModel, table=True):
    """Search queries and results for quality measurement"""
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: Optional[int] = Field(default=None, foreign_key="ragexperiment.id")
    collection_id: int = Field(foreign_key="vectorcollection.id")
    query_text: str
    results_json: Optional[str] = None
    result_count: int = 0
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    latency_ms: Optional[int] = None
    relevant_doc_ids: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CrawlJob(SQLModel, table=True):
    """Web crawling jobs"""
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="vectorcollection.id")
    start_url: str
    crawl_type: str
    max_depth: int = 1
    max_pages: int = 100
    respect_robots: bool = True
    status: str = "queued"
    pages_crawled: int = 0
    pages_indexed: int = 0
    error: Optional[str] = None
    is_recurring: bool = False
    cron_schedule: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

"""Database Configuration"""
from sqlmodel import SQLModel, create_engine, Session
from .models import (
    EmbeddingProvider,
    ChunkingStrategy,
    VectorCollection,
    IndexedDocument,
    DocumentChunk,
    RAGExperiment,
    RAGQuery,
    CrawlJob
)

# SQLite database
DATABASE_URL = "sqlite:///./rag_explorer.db"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialize database tables"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get database session for dependency injection"""
    with Session(engine) as session:
        yield session

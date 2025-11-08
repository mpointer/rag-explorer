"""
Seed database with default configurations
"""
from sqlmodel import Session, select
from .db import engine
from .models import EmbeddingProvider, ChunkingStrategy, VectorCollection
from .rag_services.chroma_service import chroma_service
import logging

logger = logging.getLogger(__name__)


def seed_database():
    """Seed database with default configurations"""
    
    with Session(engine) as session:
        logger.info("Seeding database...")
        
        # Check if already seeded
        existing_providers = session.exec(select(EmbeddingProvider)).first()
        if existing_providers:
            logger.info("Database already seeded")
            return
        
        # Seed embedding providers
        providers = [
            EmbeddingProvider(
                name="OpenAI text-embedding-3-small",
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
                description="Fast and efficient OpenAI embeddings"
            ),
            EmbeddingProvider(
                name="OpenAI text-embedding-3-large",
                provider_type="openai",
                model_name="text-embedding-3-large",
                dimension=3072,
                description="High-quality OpenAI embeddings"
            ),
            EmbeddingProvider(
                name="Cohere embed-english-v3.0",
                provider_type="cohere",
                model_name="embed-english-v3.0",
                dimension=1024,
                description="Cohere embeddings optimized for English"
            ),
            EmbeddingProvider(
                name="HuggingFace all-MiniLM-L6-v2",
                provider_type="huggingface",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                description="Fast local embeddings using sentence-transformers"
            ),
            EmbeddingProvider(
                name="Ollama nomic-embed-text",
                provider_type="ollama",
                model_name="nomic-embed-text",
                dimension=768,
                description="Local embeddings via Ollama"
            ),
        ]
        
        for provider in providers:
            session.add(provider)
        
        logger.info(f"Added {len(providers)} embedding providers")
        
        # Seed chunking strategies
        strategies = [
            ChunkingStrategy(
                name="Fixed 500 chars",
                strategy_type="fixed_size",
                chunk_size=500,
                overlap=50,
                description="Simple fixed-size chunks with 50 char overlap"
            ),
            ChunkingStrategy(
                name="Fixed 1000 chars",
                strategy_type="fixed_size",
                chunk_size=1000,
                overlap=100,
                description="Larger fixed-size chunks with 100 char overlap"
            ),
            ChunkingStrategy(
                name="Recursive Character",
                strategy_type="recursive_character",
                chunk_size=500,
                overlap=50,
                description="Respects paragraph and sentence boundaries"
            ),
            ChunkingStrategy(
                name="Semantic",
                strategy_type="semantic",
                chunk_size=500,
                overlap=50,
                description="Sentence-boundary aware chunking"
            ),
            ChunkingStrategy(
                name="Document Aware",
                strategy_type="document_aware",
                chunk_size=500,
                overlap=50,
                description="Respects document structure (headers, sections)"
            ),
        ]
        
        for strategy in strategies:
            session.add(strategy)
        
        logger.info(f"Added {len(strategies)} chunking strategies")
        
        # Create default collection
        default_collection = VectorCollection(
            name="default",
            description="Default vector collection"
        )
        session.add(default_collection)
        
        session.commit()
        
        # Create collection in ChromaDB
        chroma_service.create_collection(
            collection_name="default",
            metadata={"description": "Default vector collection"}
        )
        
        logger.info("Database seeded successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed_database()

"""
Ingestion Pipeline - Orchestrates document processing and indexing
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

from sqlmodel import Session, select

from ..models import (
    IndexedDocument, DocumentChunk, VectorCollection,
    EmbeddingProvider, ChunkingStrategy
)
from ..utils.extract_text import extract_text
from .chunking_service import ChunkingStrategyFactory
from .embedding_providers import EmbeddingProviderFactory
from .chroma_service import chroma_service

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents into the RAG system"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def ingest_document(
        self,
        file_path: str,
        collection_id: int,
        embedding_provider_id: int,
        chunking_strategy_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IndexedDocument:
        """
        Ingest a single document through the complete pipeline:
        1. Extract text
        2. Chunk text
        3. Generate embeddings
        4. Store in ChromaDB
        5. Record in database
        """
        logger.info(f"Starting ingestion for: {file_path}")
        
        # Get configuration
        collection = self.session.get(VectorCollection, collection_id)
        embedding_config = self.session.get(EmbeddingProvider, embedding_provider_id)
        chunking_config = self.session.get(ChunkingStrategy, chunking_strategy_id)
        
        if not all([collection, embedding_config, chunking_config]):
            raise ValueError("Invalid configuration IDs")
        
        # Step 1: Extract text
        logger.info("Extracting text...")
        text = extract_text(file_path)
        if not text:
            raise ValueError(f"No text extracted from {file_path}")
        
        # Step 2: Create document record
        file_path_obj = Path(file_path)
        content_hash = self._compute_hash(text)
        
        document = IndexedDocument(
            filename=file_path_obj.name,
            file_path=str(file_path_obj.absolute()),
            file_size=file_path_obj.stat().st_size,
            content_hash=content_hash,
            collection_id=collection_id,
            embedding_provider_id=embedding_provider_id,
            chunking_strategy_id=chunking_strategy_id,
            metadata=metadata or {},
            status="processing"
        )
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        
        try:
            # Step 3: Chunk text
            logger.info("Chunking text...")
            chunker = ChunkingStrategyFactory.create_strategy(
                strategy_type=chunking_config.strategy_type,
                chunk_size=chunking_config.chunk_size,
                overlap=chunking_config.overlap
            )
            
            chunks = chunker.chunk(
                text,
                metadata={
                    "document_id": document.id,
                    "filename": document.filename,
                    **(metadata or {})
                }
            )
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            logger.info("Generating embeddings...")
            embedding_provider = EmbeddingProviderFactory.create_provider(
                provider_type=embedding_config.provider_type,
                model_name=embedding_config.model_name
            )
            
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_provider.embed_documents(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 5: Store in ChromaDB
            logger.info("Storing in ChromaDB...")
            chunk_ids = [
                f"doc_{document.id}_chunk_{i}"
                for i in range(len(chunks))
            ]
            
            chunk_metadatas = [
                {
                    **chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "document_id": document.id,
                    "collection_id": collection_id
                }
                for chunk in chunks
            ]
            
            chroma_service.add_documents(
                collection_name=collection.name,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Step 6: Record chunks in database
            logger.info("Recording chunks in database...")
            for i, chunk in enumerate(chunks):
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chroma_id=chunk_ids[i],
                    metadata=chunk.metadata
                )
                self.session.add(chunk_record)
            
            # Update document status
            document.status = "completed"
            document.chunk_count = len(chunks)
            self.session.commit()
            
            logger.info(f"Successfully ingested document: {document.filename}")
            return document
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            document.status = "failed"
            document.metadata["error"] = str(e)
            self.session.commit()
            raise
    
    def ingest_crawled_content(
        self,
        crawl_results: List[Dict[str, Any]],
        collection_id: int,
        embedding_provider_id: int,
        chunking_strategy_id: int
    ) -> List[IndexedDocument]:
        """Ingest multiple crawled web pages"""
        documents = []
        
        for result in crawl_results:
            if result["status"] != "success":
                continue
            
            try:
                # Create temporary text content
                content = result["content"]
                url = result["url"]
                
                # Create a pseudo-file path for web content
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                temp_filename = f"web_{url_hash}.txt"
                
                # Write to temp file
                temp_path = Path("/tmp") / temp_filename
                temp_path.write_text(content)
                
                # Ingest using standard pipeline
                metadata = {
                    **result.get("metadata", {}),
                    "source_type": "web_crawl",
                    "source_url": url
                }
                
                document = self.ingest_document(
                    file_path=str(temp_path),
                    collection_id=collection_id,
                    embedding_provider_id=embedding_provider_id,
                    chunking_strategy_id=chunking_strategy_id,
                    metadata=metadata
                )
                
                documents.append(document)
                
                # Clean up temp file
                temp_path.unlink()
                
            except Exception as e:
                logger.error(f"Error ingesting crawled content from {result.get('url')}: {e}")
        
        logger.info(f"Ingested {len(documents)} crawled pages")
        return documents
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks"""
        try:
            document = self.session.get(IndexedDocument, document_id)
            if not document:
                return False
            
            # Get collection name
            collection = self.session.get(VectorCollection, document.collection_id)
            
            # Delete from ChromaDB
            chunks = self.session.exec(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            ).all()
            
            if chunks:
                chunk_ids = [chunk.chroma_id for chunk in chunks]
                chroma_service.delete_documents(collection.name, chunk_ids)
            
            # Delete from database (cascades to chunks)
            self.session.delete(document)
            self.session.commit()
            
            logger.info(f"Deleted document: {document.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def reindex_document(self, document_id: int) -> IndexedDocument:
        """Re-process and re-index an existing document"""
        document = self.session.get(IndexedDocument, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Delete existing chunks
        self.delete_document(document_id)
        
        # Re-ingest
        return self.ingest_document(
            file_path=document.file_path,
            collection_id=document.collection_id,
            embedding_provider_id=document.embedding_provider_id,
            chunking_strategy_id=document.chunking_strategy_id,
            metadata=document.metadata
        )
    
    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

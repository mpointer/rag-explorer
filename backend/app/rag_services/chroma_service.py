"""
ChromaDB Service - Vector database operations
"""
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for interacting with ChromaDB"""
    
    def __init__(self):
        """Initialize ChromaDB client"""
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"ChromaDB initialized at {settings.chroma_db_dir}")
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """Create or get a collection"""
        try:
            # Check if collection exists
            existing = self.client.list_collections()
            collection_names = [c.name for c in existing]
            
            if collection_name in collection_names:
                logger.info(f"Collection '{collection_name}' already exists, retrieving...")
                return self.client.get_collection(name=collection_name)
            
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            logger.info(f"Created collection: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
        """Get an existing collection"""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error getting collection '{collection_name}': {e}")
            return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents to a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Add timestamp to metadata
            for metadata in metadatas:
                metadata["indexed_at"] = datetime.utcnow().isoformat()
            
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def update_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Update existing documents in a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection.update(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Updated {len(documents)} documents in '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def search(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching collection: {e}")
            raise
    
    def get_documents_by_ids(
        self,
        collection_name: str,
        ids: List[str]
    ) -> Dict[str, Any]:
        """Retrieve documents by their IDs"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            results = collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def delete_documents(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete documents by IDs"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {"error": f"Collection '{collection_name}' not found"}
            
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [
                {
                    "name": c.name,
                    "count": c.count(),
                    "metadata": c.metadata
                }
                for c in collections
            ]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def reset_database(self) -> bool:
        """Reset entire database (use with caution!)"""
        try:
            self.client.reset()
            logger.warning("Database reset performed")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False


# Singleton instance
chroma_service = ChromaService()

"""
Search Service - Semantic search, hybrid search, and reranking
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from sqlmodel import Session, select

from ..models import VectorCollection, EmbeddingProvider, RAGQuery
from .embedding_providers import EmbeddingProviderFactory
from .chroma_service import chroma_service

logger = logging.getLogger(__name__)


class SearchService:
    """Service for searching documents in the RAG system"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def search(
        self,
        query: str,
        collection_id: int,
        embedding_provider_id: int,
        top_k: int = 10,
        search_type: str = "semantic",
        rerank: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant documents
        
        Args:
            query: Search query text
            collection_id: ID of the collection to search
            embedding_provider_id: ID of embedding provider to use
            top_k: Number of results to return
            search_type: "semantic", "hybrid", or "keyword"
            rerank: Whether to apply reranking
            filters: Optional metadata filters
        """
        logger.info(f"Searching: '{query}' in collection {collection_id}")
        
        # Get configuration
        collection = self.session.get(VectorCollection, collection_id)
        embedding_config = self.session.get(EmbeddingProvider, embedding_provider_id)
        
        if not collection or not embedding_config:
            raise ValueError("Invalid collection or embedding provider ID")
        
        # Create query record
        query_record = RAGQuery(
            query_text=query,
            collection_id=collection_id,
            embedding_provider_id=embedding_provider_id,
            search_type=search_type,
            top_k=top_k,
            filters=filters or {},
            results=[]
        )
        self.session.add(query_record)
        self.session.commit()
        self.session.refresh(query_record)
        
        try:
            # Generate query embedding
            embedding_provider = EmbeddingProviderFactory.create_provider(
                provider_type=embedding_config.provider_type,
                model_name=embedding_config.model_name
            )
            
            query_embedding = embedding_provider.embed_query(query)
            
            # Perform search based on type
            if search_type == "semantic":
                results = self._semantic_search(
                    collection.name,
                    query_embedding,
                    top_k,
                    filters
                )
            elif search_type == "hybrid":
                results = self._hybrid_search(
                    collection.name,
                    query,
                    query_embedding,
                    top_k,
                    filters
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Rerank if requested
            if rerank and len(results) > 0:
                results = self._rerank_results(query, results, top_k)
            
            # Update query record
            query_record.results = results
            query_record.result_count = len(results)
            self.session.commit()
            
            return {
                "query_id": query_record.id,
                "query": query,
                "results": results,
                "count": len(results),
                "search_type": search_type,
                "reranked": rerank
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def _semantic_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform pure semantic (vector) search"""
        
        results = chroma_service.search(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted_results = []
        
        if results and "documents" in results:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": float(results["distances"][0][i]),
                    "score": float(1 / (1 + results["distances"][0][i])),  # Convert distance to similarity
                    "rank": i + 1,
                    "search_method": "semantic"
                })
        
        return formatted_results
    
    def _hybrid_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching
        Uses reciprocal rank fusion (RRF) to combine results
        """
        
        # Get semantic search results (more results for fusion)
        semantic_results = self._semantic_search(
            collection_name,
            query_embedding,
            top_k * 2,  # Get more results for fusion
            filters
        )
        
        # Simple keyword search using ChromaDB's where_document
        # This is a basic implementation - could be enhanced with BM25
        keyword_results = chroma_service.search(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            where=filters,
            where_document={"$contains": query.lower()}
        )
        
        # Format keyword results
        keyword_formatted = []
        if keyword_results and "documents" in keyword_results:
            for i in range(len(keyword_results["documents"][0])):
                keyword_formatted.append({
                    "text": keyword_results["documents"][0][i],
                    "metadata": keyword_results["metadatas"][0][i],
                    "distance": float(keyword_results["distances"][0][i]),
                    "score": float(1 / (1 + keyword_results["distances"][0][i])),
                    "rank": i + 1,
                    "search_method": "keyword"
                })
        
        # Apply reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_formatted,
            top_k
        )
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion
        RRF score = sum(1 / (k + rank)) for each result list
        """
        
        # Create a dictionary to store combined scores
        combined_scores: Dict[str, Dict[str, Any]] = {}
        
        # Add semantic results
        for result in semantic_results:
            text = result["text"]
            if text not in combined_scores:
                combined_scores[text] = {
                    **result,
                    "rrf_score": 0,
                    "search_method": "hybrid"
                }
            combined_scores[text]["rrf_score"] += 1 / (k + result["rank"])
        
        # Add keyword results
        for result in keyword_results:
            text = result["text"]
            if text not in combined_scores:
                combined_scores[text] = {
                    **result,
                    "rrf_score": 0,
                    "search_method": "hybrid"
                }
            combined_scores[text]["rrf_score"] += 1 / (k + result["rank"])
        
        # Sort by RRF score and return top_k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]
        
        # Update ranks
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
            result["score"] = result["rrf_score"]
        
        return sorted_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder or relevance model
        This is a simplified implementation - in production, use Cohere rerank API
        """
        
        try:
            # Try using Cohere rerank if available
            import cohere
            import os
            
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                logger.warning("Cohere API key not found, skipping rerank")
                return results
            
            co = cohere.Client(api_key)
            
            # Prepare documents for reranking
            documents = [r["text"] for r in results]
            
            # Call Cohere rerank
            rerank_response = co.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0"
            )
            
            # Map reranked results back
            reranked_results = []
            for idx, rerank_result in enumerate(rerank_response.results):
                original_result = results[rerank_result.index]
                reranked_results.append({
                    **original_result,
                    "rank": idx + 1,
                    "rerank_score": rerank_result.relevance_score,
                    "original_rank": original_result["rank"]
                })
            
            logger.info("Reranked results using Cohere")
            return reranked_results
            
        except ImportError:
            logger.warning("Cohere library not available, skipping rerank")
            return results
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            return results
    
    def calculate_metrics(
        self,
        query_id: int,
        relevant_doc_ids: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for a query
        
        Metrics:
        - MRR (Mean Reciprocal Rank)
        - Precision@K
        - Recall@K
        - NDCG@K (Normalized Discounted Cumulative Gain)
        """
        
        query_record = self.session.get(RAGQuery, query_id)
        if not query_record or not query_record.results:
            return {}
        
        results = query_record.results
        retrieved_ids = [r["metadata"].get("document_id") for r in results]
        
        metrics = {}
        
        # MRR - Mean Reciprocal Rank
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if str(doc_id) in relevant_doc_ids:
                mrr = 1 / (i + 1)
                break
        metrics["mrr"] = mrr
        
        # Calculate metrics at different K values
        for k in k_values:
            retrieved_at_k = retrieved_ids[:k]
            relevant_retrieved = [
                doc_id for doc_id in retrieved_at_k
                if str(doc_id) in relevant_doc_ids
            ]
            
            # Precision@K
            precision = len(relevant_retrieved) / k if k > 0 else 0
            metrics[f"precision@{k}"] = precision
            
            # Recall@K
            recall = (
                len(relevant_retrieved) / len(relevant_doc_ids)
                if len(relevant_doc_ids) > 0 else 0
            )
            metrics[f"recall@{k}"] = recall
            
            # NDCG@K
            ndcg = self._calculate_ndcg(
                retrieved_at_k,
                relevant_doc_ids,
                k
            )
            metrics[f"ndcg@{k}"] = ndcg
        
        # Update query record with metrics
        query_record.metrics = metrics
        self.session.commit()
        
        return metrics
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        
        # DCG - Discounted Cumulative Gain
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if str(doc_id) in relevant_ids:
                # Binary relevance: 1 if relevant, 0 otherwise
                rel = 1
                dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1
        
        # IDCG - Ideal DCG (if all relevant docs were at top)
        ideal_retrieved = relevant_ids[:k]
        idcg = 0
        for i in range(min(len(ideal_retrieved), k)):
            idcg += 1 / np.log2(i + 2)
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg

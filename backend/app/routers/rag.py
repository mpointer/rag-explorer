"""
RAG API Router - Endpoints for RAG operations
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import shutil
from pathlib import Path
import logging

from ..db import get_session
from ..config import settings
from ..models import (
    VectorCollection, EmbeddingProvider, ChunkingStrategy,
    IndexedDocument, RAGExperiment, RAGQuery, CrawlJob
)
from ..rag_services.ingestion_pipeline import IngestionPipeline
from ..rag_services.search_service import SearchService
from ..rag_services.crawl_service import crawl_service
from ..rag_services.chroma_service import chroma_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["RAG"])


# Pydantic models for requests/responses
class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    collection_id: int
    embedding_provider_id: int
    top_k: int = 10
    search_type: str = "semantic"
    rerank: bool = False
    filters: Optional[Dict[str, Any]] = None


class CrawlRequest(BaseModel):
    url: str
    crawl_type: str = "single"  # "single", "sitemap", "recursive"
    max_depth: int = 2
    max_pages: int = 100
    collection_id: int
    embedding_provider_id: int
    chunking_strategy_id: int


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    collection_id: int
    config_a: Dict[str, Any]
    config_b: Dict[str, Any]


class MetricsRequest(BaseModel):
    query_id: int
    relevant_doc_ids: List[str]
    k_values: Optional[List[int]] = [5, 10, 20]


# Collections endpoints
@router.get("/collections")
def list_collections(session: Session = Depends(get_session)):
    """List all vector collections"""
    collections = session.exec(select(VectorCollection)).all()
    return collections


@router.post("/collections")
def create_collection(
    collection: CollectionCreate,
    session: Session = Depends(get_session)
):
    """Create a new vector collection"""
    # Create in database
    db_collection = VectorCollection(
        name=collection.name,
        description=collection.description
    )
    session.add(db_collection)
    session.commit()
    session.refresh(db_collection)
    
    # Create in ChromaDB
    chroma_service.create_collection(
        collection_name=collection.name,
        metadata={"description": collection.description or ""}
    )
    
    return db_collection


@router.delete("/collections/{collection_id}")
def delete_collection(collection_id: int, session: Session = Depends(get_session)):
    """Delete a collection"""
    collection = session.get(VectorCollection, collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Delete from ChromaDB
    chroma_service.delete_collection(collection.name)
    
    # Delete from database
    session.delete(collection)
    session.commit()
    
    return {"message": "Collection deleted"}


# Embedding providers endpoints
@router.get("/embedding-providers")
def list_embedding_providers(session: Session = Depends(get_session)):
    """List all embedding providers"""
    providers = session.exec(select(EmbeddingProvider)).all()
    return providers


# Chunking strategies endpoints
@router.get("/chunking-strategies")
def list_chunking_strategies(session: Session = Depends(get_session)):
    """List all chunking strategies"""
    strategies = session.exec(select(ChunkingStrategy)).all()
    return strategies


# Document upload endpoint
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: int = 1,
    embedding_provider_id: int = 1,
    chunking_strategy_id: int = 1,
    session: Session = Depends(get_session)
):
    """Upload and ingest a document"""
    
    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest document
        pipeline = IngestionPipeline(session)
        document = pipeline.ingest_document(
            file_path=str(file_path),
            collection_id=collection_id,
            embedding_provider_id=embedding_provider_id,
            chunking_strategy_id=chunking_strategy_id,
            metadata={"original_filename": file.filename}
        )
        
        return document
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document management endpoints
@router.get("/documents")
def list_documents(
    collection_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """List all indexed documents"""
    query = select(IndexedDocument)
    if collection_id:
        query = query.where(IndexedDocument.collection_id == collection_id)
    
    documents = session.exec(query).all()
    return documents


@router.get("/documents/{document_id}")
def get_document(document_id: int, session: Session = Depends(get_session)):
    """Get document details"""
    document = session.get(IndexedDocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, session: Session = Depends(get_session)):
    """Delete a document"""
    pipeline = IngestionPipeline(session)
    success = pipeline.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted"}


# Web crawling endpoints
@router.post("/crawl")
async def crawl_web(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """Crawl web pages and ingest content"""
    
    # Create crawl job
    crawl_job = CrawlJob(
        url=request.url,
        crawl_type=request.crawl_type,
        status="pending",
        pages_crawled=0,
        pages_failed=0
    )
    session.add(crawl_job)
    session.commit()
    session.refresh(crawl_job)
    
    # Start crawling in background
    background_tasks.add_task(
        _crawl_and_ingest,
        crawl_job.id,
        request
    )
    
    return crawl_job


def _crawl_and_ingest(crawl_job_id: int, request: CrawlRequest):
    """Background task for crawling and ingesting"""
    from ..db import engine
    
    with Session(engine) as session:
        crawl_job = session.get(CrawlJob, crawl_job_id)
        
        try:
            crawl_job.status = "running"
            session.commit()
            
            # Perform crawl
            if request.crawl_type == "single":
                results = [crawl_service.crawl_single_page(request.url)]
            elif request.crawl_type == "sitemap":
                results = crawl_service.crawl_sitemap(request.url)
            elif request.crawl_type == "recursive":
                results = crawl_service.crawl_recursive(
                    request.url,
                    allowed_domains=None
                )
            else:
                raise ValueError(f"Unknown crawl type: {request.crawl_type}")
            
            # Count successes and failures
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]
            
            crawl_job.pages_crawled = len(successful)
            crawl_job.pages_failed = len(failed)
            
            # Ingest successful results
            pipeline = IngestionPipeline(session)
            documents = pipeline.ingest_crawled_content(
                crawl_results=successful,
                collection_id=request.collection_id,
                embedding_provider_id=request.embedding_provider_id,
                chunking_strategy_id=request.chunking_strategy_id
            )
            
            crawl_job.status = "completed"
            crawl_job.result_summary = {
                "documents_ingested": len(documents),
                "pages_crawled": len(successful),
                "pages_failed": len(failed)
            }
            
        except Exception as e:
            logger.error(f"Crawl job {crawl_job_id} failed: {e}")
            crawl_job.status = "failed"
            crawl_job.result_summary = {"error": str(e)}
        
        finally:
            session.commit()


@router.get("/crawl-jobs")
def list_crawl_jobs(session: Session = Depends(get_session)):
    """List all crawl jobs"""
    jobs = session.exec(select(CrawlJob)).all()
    return jobs


@router.get("/crawl-jobs/{job_id}")
def get_crawl_job(job_id: int, session: Session = Depends(get_session)):
    """Get crawl job status"""
    job = session.get(CrawlJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Crawl job not found")
    return job


# Search endpoints
@router.post("/search")
def search(request: SearchRequest, session: Session = Depends(get_session)):
    """Search for relevant documents"""
    search_service = SearchService(session)
    
    try:
        results = search_service.search(
            query=request.query,
            collection_id=request.collection_id,
            embedding_provider_id=request.embedding_provider_id,
            top_k=request.top_k,
            search_type=request.search_type,
            rerank=request.rerank,
            filters=request.filters
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queries")
def list_queries(
    collection_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    """List search queries"""
    query = select(RAGQuery).order_by(RAGQuery.created_at.desc())
    if collection_id:
        query = query.where(RAGQuery.collection_id == collection_id)
    
    queries = session.exec(query.limit(100)).all()
    return queries


@router.post("/metrics")
def calculate_metrics(
    request: MetricsRequest,
    session: Session = Depends(get_session)
):
    """Calculate quality metrics for a query"""
    search_service = SearchService(session)
    
    try:
        metrics = search_service.calculate_metrics(
            query_id=request.query_id,
            relevant_doc_ids=request.relevant_doc_ids,
            k_values=request.k_values
        )
        return metrics
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Experiments endpoints
@router.post("/experiments")
def create_experiment(
    experiment: ExperimentCreate,
    session: Session = Depends(get_session)
):
    """Create an A/B test experiment"""
    db_experiment = RAGExperiment(
        name=experiment.name,
        description=experiment.description,
        collection_id=experiment.collection_id,
        config_a=experiment.config_a,
        config_b=experiment.config_b,
        status="active"
    )
    session.add(db_experiment)
    session.commit()
    session.refresh(db_experiment)
    
    return db_experiment


@router.get("/experiments")
def list_experiments(session: Session = Depends(get_session)):
    """List all experiments"""
    experiments = session.exec(select(RAGExperiment)).all()
    return experiments


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Get experiment details"""
    experiment = session.get(RAGExperiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


# Health check
@router.get("/health")
def health_check():
    """Check RAG system health"""
    return {
        "status": "healthy",
        "chroma_collections": len(chroma_service.list_collections())
    }

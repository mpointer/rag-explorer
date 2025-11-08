"""
RAG Explorer - FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from .routers import rag
from .config import settings
from .db import init_db

app = FastAPI(
    title="RAG Explorer API",
    version="1.0.0",
    description="Experimentation platform for RAG systems"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rag.router)

@app.get("/api/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "RAG Explorer"}


@app.on_event("startup")
def on_startup():
    """Initialize database on startup"""
    logger.info("Initializing database...")
    init_db()
    logger.info("RAG Explorer API started successfully")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "RAG Explorer API",
        "docs": "/docs",
        "health": "/api/health"
    }

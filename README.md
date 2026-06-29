# RAG Explorer

An experimentation platform for **Retrieval-Augmented Generation (RAG)**. Upload
documents or crawl websites, index them into a vector store using one of several
embedding providers and chunking strategies, then run semantic and hybrid
searches and measure retrieval quality with standard IR metrics.

## Features

- **Document ingestion** — upload PDF, DOCX, PPTX, TXT, Markdown, HTML, CSV, or
  JSON files. Text is extracted, chunked, embedded, and stored in ChromaDB.
- **Web crawling** — single page, sitemap, or recursive crawl, ingested as
  documents in the background.
- **Multiple embedding providers** — OpenAI, Cohere, HuggingFace
  (sentence-transformers), Ollama, and Google. Providers are loaded lazily, so
  you only need the libraries/keys for the ones you use.
- **Multiple chunking strategies** — fixed-size, recursive-character, semantic
  (sentence-aware), and document-aware (header/structure-aware).
- **Search** — pure semantic (vector) search and hybrid search (vector +
  keyword) combined with Reciprocal Rank Fusion, with optional Cohere reranking.
- **Quality metrics** — MRR, Precision@K, Recall@K, and NDCG@K computed against a
  set of relevant document IDs.
- **A/B experiments** — store paired configurations for comparison.

## Architecture

```text
frontend/   Vue 3 + Vite + Tailwind single-page playground
backend/    FastAPI + SQLModel (SQLite) + ChromaDB
```

- **Relational metadata** (collections, providers, strategies, documents,
  chunks, queries, experiments, crawl jobs) lives in SQLite via SQLModel.
- **Vectors** live in a persistent ChromaDB store on disk.
- The frontend talks to the backend under `/api/rag/*`; in dev, Vite proxies
  `/api` to the backend.

## Quick start (Docker)

```bash
cp .env.example .env   # add API keys for any providers you want to use
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs

On first start the backend seeds a default collection plus a set of embedding
providers and chunking strategies so the UI is usable immediately.

## Running locally without Docker

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API keys are read from environment variables (`OPENAI_API_KEY`,
`COHERE_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`). Providers that need a
key or an unavailable library will only fail when actually used — the rest of the
app works without them. The HuggingFace provider runs locally and requires no
key.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API overview

All endpoints are under `/api/rag`:

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET    | `/collections` | List collections |
| POST   | `/collections` | Create a collection |
| DELETE | `/collections/{id}` | Delete a collection |
| GET    | `/embedding-providers` | List embedding providers |
| GET    | `/chunking-strategies` | List chunking strategies |
| POST   | `/documents/upload` | Upload and ingest a document |
| GET    | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Delete a document |
| POST   | `/crawl` | Start a crawl job |
| GET    | `/crawl-jobs` | List crawl jobs |
| POST   | `/search` | Run a semantic/hybrid search |
| GET    | `/queries` | List recent queries |
| POST   | `/metrics` | Compute quality metrics for a query |
| POST   | `/experiments` | Create an A/B experiment |
| GET    | `/experiments` | List experiments |

Interactive docs are available at `/docs`.

## Notes

- The SQLite database (`rag_explorer.db`), the `uploads/` directory, and the
  `chroma_db/` vector store are created on first run and are git-ignored.
- Retrieval metrics are computed at the document level: chunk results are
  de-duplicated by `document_id` before scoring so metrics stay within `[0, 1]`.

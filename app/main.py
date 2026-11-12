import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_admin, routes_ingest, routes_query
from app.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(title="PDF RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(routes_query.router, tags=["query"])
app.include_router(routes_admin.router, prefix="/admin", tags=["admin"])


@app.get("/health")
def health():
    return {"status": "ok"}

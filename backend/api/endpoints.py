from fastapi import APIRouter, Depends, HTTPException, Request
from models.schemas import QueryRequest, QueryResponse, IndexResponse
from services.legaldoc_indexer import LegalDocIndexer
from services.rag_pipeline import run_qa
from qdrant_client import QdrantClient
from providers.s3_provider import S3Provider

router = APIRouter()

def get_qdrant_client(request: Request) -> QdrantClient:
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    return qdrant_client

def get_s3_provider(request: Request) -> S3Provider:
    provider = getattr(request.app.state, "s3_provider", None)
    if provider is None:
        raise HTTPException(status_code=500, detail="S3 provider not initialized")
    return provider

@router.post("/index-documents", response_model=IndexResponse)
async def index_documents_endpoint(
    s3_provider: S3Provider = Depends(get_s3_provider),
    qdrant_client: QdrantClient = Depends(get_qdrant_client)
):
    s3_client = s3_provider.get_s3_client()
    indexer = LegalDocIndexer(qdrant_client, s3_client)
    indexed_count = indexer.index()
    return {"indexed_count": indexed_count}

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, client: QdrantClient = Depends(get_qdrant_client)):
    answer = run_qa(request.query, client)
    return {"answer": answer}

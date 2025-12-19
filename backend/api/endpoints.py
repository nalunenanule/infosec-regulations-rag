from fastapi import APIRouter, Depends, HTTPException, Request
from models.schemas import QueryRequest, QueryResponse, IndexResponse
from utils.pdf_utilities import PDFUtilities
from services.qdrant_indexer import QdrantIndexer
from services.rag_pipeline import run_qa
from qdrant_client import QdrantClient
from providers.s3_provider import S3Provider

router = APIRouter()

def get_client(request: Request) -> QdrantClient:
    client = getattr(request.app.state, "client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    return client

def get_s3_provider(request: Request) -> S3Provider:
    provider = getattr(request.app.state, "s3_provider", None)
    if provider is None:
        raise HTTPException(status_code=500, detail="S3 provider not initialized")
    return provider

@router.post("/index-documents", response_model=IndexResponse)
async def index_documents_from_s3_endpoint(
    s3: S3Provider = Depends(get_s3_provider),
    client: QdrantClient = Depends(get_client)
):
    files_urls = s3.get_files_urls() 
    docs, doc_length = PDFUtilities.load_and_split_pdfs(files_urls)
    indexer = QdrantIndexer(client)
    indexer.build_collection(docs)
    return {"indexed_count": doc_length}

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, client: QdrantClient = Depends(get_client)):
    answer = run_qa(request.query, client)
    return {"answer": answer}

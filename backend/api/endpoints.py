from fastapi import APIRouter, Depends, HTTPException, Request
from models.schemas import QueryRequest, QueryResponse, IndexResponse
from services.loader import load_pdfs, split_documents
from services.vectorstore import build_qdrant, QdrantClient
from services.rag_pipeline import run_qa

router = APIRouter()

def get_client(request: Request) -> QdrantClient:
    client = getattr(request.app.state, "client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    return client

@router.post("/index-documents", response_model=IndexResponse)
async def index_documents_endpoint(client: QdrantClient = Depends(get_client)):
    raw_docs, doc_lenth = load_pdfs()
    docs = split_documents(raw_docs)
    build_qdrant(client, docs)
    return {"indexed_count": doc_lenth}

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, client: QdrantClient = Depends(get_client)):
    answer = run_qa(request.query, client)
    return {"answer": answer}

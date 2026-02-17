from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.endpoints import router as api_router
from qdrant_client import QdrantClient
from providers.s3_provider import S3Provider
from config import QDRANT_URL, S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False, timeout=60)
    app.state.qdrant_client = qdrant_client
    print("Qdrant client initialized.")

    s3_provider = S3Provider(
        endpoint_url=S3_URL,
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
    )
    app.state.s3_provider = s3_provider
    print("S3 provider initialized.")

    yield
    print("Application shutdown.")

app = FastAPI(title="API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)

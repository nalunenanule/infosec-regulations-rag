from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.endpoints import router as api_router
from services.vectorstore import QdrantClient
from config import QDRANT_URL

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = QdrantClient(url=QDRANT_URL)
    app.state.client = client
    print("Qdrant client initialized.")
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

import os
from dotenv import load_dotenv
from pathlib import Path

if os.getenv("RUNNING_IN_DOCKER") != "1":
    env_path = Path(".local.env")
    if env_path.exists():
        load_dotenv(env_path)

USE_GIGACHAT_EMBEDDINGS = os.getenv("USE_GIGACHAT_EMBEDDINGS")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
GIGACHAT_TOKEN = os.getenv("GIGACHAT_TOKEN")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")
S3_URL = os.getenv("S3_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
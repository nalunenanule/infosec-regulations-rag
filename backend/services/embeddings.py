from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from config import EMBEDDINGS_MODEL_NAME

def get_dense_embeddings():
    return HuggingFaceEmbeddings(
        model_name=f"/huggingface_models/{EMBEDDINGS_MODEL_NAME}",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_sparse_embeddings(model_name="Qdrant/bm25"):
    return SparseTextEmbedding(model_name=model_name)
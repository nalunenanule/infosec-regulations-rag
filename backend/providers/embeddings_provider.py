from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from config import EMBEDDINGS_MODEL_NAME

class EmbeddingProvider:
    """
    Класс-провайдер для работы с моделями эмбеддингов
    """

    def __init__(self, dense_model_name: str = EMBEDDINGS_MODEL_NAME, sparse_model_name: str = "Qdrant/bm25"):
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

    def get_dense_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=f"/huggingface_models/{self.dense_model_name}",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_sparse_embeddings(self):
        return SparseTextEmbedding(model_name=self.sparse_model_name, cache_dir="./huggingface_models/")
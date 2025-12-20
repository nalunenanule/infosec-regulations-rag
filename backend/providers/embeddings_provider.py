from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from config import DENSE_MODEL_PATH, SPARSE_MODEL_PATH, SPARSE_MODEL_NAME

class EmbeddingProvider:
    """
    Класс-провайдер для работы с моделями эмбеддингов
    """

    def __init__(self, dense_model_path: str = DENSE_MODEL_PATH, sparse_model_path: str = SPARSE_MODEL_PATH, sparse_model_name: str = SPARSE_MODEL_NAME):
        self.dense_model_path = dense_model_path
        self.sparse_model_path = sparse_model_path
        self.sparse_model_name = sparse_model_name

    def get_dense_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.dense_model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_sparse_embeddings(self):
        return SparseTextEmbedding(model_name=self.sparse_model_name, cache_dir=self.sparse_model_path)
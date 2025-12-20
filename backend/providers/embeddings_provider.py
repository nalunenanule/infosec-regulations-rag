from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from config import DENSE_MODEL_PATH, SPARSE_MODEL_PATH, SPARSE_MODEL_NAME

class EmbeddingsProvider:
    """
    Класс-провайдер для работы с моделями эмбеддингов.
    Если модель не найдена по пути dense_model_path, скачивает ее автоматически.
    """

    DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"
    DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

    def __init__(
        self,
        dense_model_path: str = None,
        sparse_model_path: str = None,
        sparse_model_name: str = None,
    ):
        self.dense_model_path = dense_model_path or DENSE_MODEL_PATH or "./huggingface_models/dense"
        self.sparse_model_path = sparse_model_path or SPARSE_MODEL_PATH or "./huggingface_models/sparse"
        self.sparse_model_name = sparse_model_name or SPARSE_MODEL_NAME or self.DEFAULT_SPARSE_MODEL

        # Проверяем и скачиваем dense модель
        if not Path(self.dense_model_path).exists():
            print(f"Dense model not found at {self.dense_model_path}, downloading...")
            Path(self.dense_model_path).mkdir(parents=True, exist_ok=True)
            SentenceTransformer(self.DEFAULT_DENSE_MODEL).save(self.dense_model_path)
            print("Dense model downloaded.")

    def get_dense_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.dense_model_path,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_sparse_embeddings(self):
        return SparseTextEmbedding(
            model_name=self.sparse_model_name,
            cache_dir=self.sparse_model_path
        )

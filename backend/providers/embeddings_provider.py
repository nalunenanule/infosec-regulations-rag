import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from fastembed import SparseTextEmbedding
from config import DENSE_MODEL_NAME, SPARSE_MODEL_NAME, GIGACHAT_TOKEN, USE_GIGACHAT_EMBEDDINGS

class EmbeddingsProvider:
    """
    Класс-провайдер для работы с моделями эмбеддингов.
    """

    def __init__(
        self,
        dense_model_name: str = None,
        sparse_model_name: str = None,
    ):
        self.dense_model_name = dense_model_name or DENSE_MODEL_NAME
        self.sparse_model_name = sparse_model_name or SPARSE_MODEL_NAME

    def get_dense_embeddings(self):
        if USE_GIGACHAT_EMBEDDINGS != "1":
            return HuggingFaceEmbeddings(
                model_name=self.dense_model_name,
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
                show_progress=True
            )
        return GigaChatEmbeddings(credentials=GIGACHAT_TOKEN, verify_ssl_certs=False)

    def get_sparse_embeddings(self):
        return SparseTextEmbedding(
            model_name=self.sparse_model_name
        )

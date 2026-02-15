from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from providers.embeddings_provider import EmbeddingsProvider
from utils.ru_text_utilities import RuTextUtilities
from config import QDRANT_COLLECTION_NAME, S3_BUCKET_NAME

class LegalDocIndexer:
    """
    Класс для загрузки, разбиения и индексирования документов в Qdrant.
    """

    def __init__(self, qdrant_client: QdrantClient, s3_client):
        self.qdrant_client = qdrant_client
        self.s3_client = s3_client
        self.embedding_provider = EmbeddingsProvider()

    def load_and_split_pdfs(self) -> List[Document]:
        """
        Загружает PDF-документы и делит их на чанки.
        """
        all_docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=[
                "\n\n\n",
                "Глава ",
                "\nСтатья ",
                "\n\n",
                "\n\d+\.\s",
                "\n\d+\)\s",
                "\n\d+\.\d+\)\s",
                "\n",
                " "
            ],
            keep_separator=True
        )

        file_urls = self.list_files()
        for url in file_urls:
            loader = PyPDFLoader(url)
            docs = loader.load()
            split_docs = splitter.split_documents(docs)
            all_docs.extend(split_docs)

        return all_docs
    
    def list_files(self, prefix: str = "") -> List[str]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        file_list = []
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                file_list.append(self.get_file_url(obj["Key"]))
        return file_list
    
    def get_file_url(self, key: str, expires_in: int = 3600) -> str:
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in
        )

    def build_collection(self, docs: List[Document]):
        """
        Создает коллекцию в Qdrant и загружает все документы.
        """
        if self.qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
            self.qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)

        self.qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )

        dense_embeddings = self.embedding_provider.get_dense_embeddings()
        sparse_model = self.embedding_provider.get_sparse_embeddings()

        points = []
        for i, doc in enumerate(docs):
            dense_vector = dense_embeddings.embed_documents([f"passage: {doc.page_content}"])[0]
            sparse_vector = list(sparse_model.embed([RuTextUtilities().preprocess(doc.page_content)]))[0]

            points.append(
                PointStruct(
                    id=i,
                    vector={"dense": dense_vector, "sparse": sparse_vector.as_object()},
                    payload={"page_content": doc.page_content, "metadata": doc.metadata}
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points[i:i+batch_size]
            )

    def index(self) -> int:
        """
        Основная функция: загружает PDF, разбивает на чанки и индексирует в Qdrant.
        Возвращает количество чанков.
        """
        docs = self.load_and_split_pdfs()
        self.build_collection(docs)
        return len(docs)

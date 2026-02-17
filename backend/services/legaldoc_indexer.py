from typing import List
import tempfile
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from providers.embeddings_provider import EmbeddingsProvider
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
        all_docs = []
        headers_to_split_on = [
            ("#", "chapter"),
            ("##", "article"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        file_keys = self.list_s3_keys()
        for key in file_keys:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
            tmp_path = tmp.name
            tmp.close()

            try:
                self.s3_client.download_file(S3_BUCKET_NAME, key, tmp_path)
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                file_content = re.sub(r'^(#+)([^ \t#\n])', r'\1 \2', file_content, flags=re.MULTILINE)
                sections = md_splitter.split_text(file_content)

                for section in sections:
                    final_chunks = recursive_splitter.split_documents([section])
                    
                    for chunk in final_chunks:
                        chunk.metadata["source"] = key
                        if "article" in chunk.metadata:
                            chunk.metadata["article_number"] = self._extract_number(chunk.metadata["article"])
                        if "chapter" in chunk.metadata:
                            chunk.metadata["chapter_number"] = self._extract_number(chunk.metadata["chapter"])
                        
                        all_docs.append(chunk)
                        
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return len(file_keys), all_docs
    
    def _extract_number(self, text: str) -> str:
        match = re.search(r"(\d+)", text)
        return match.group(1) if match else text
    
    def list_s3_keys(self, prefix: str = "") -> List[str]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".md"):
                    keys.append(obj["Key"])
        return keys

    def build_collection(self, docs: List[Document]):
        """
        Создает коллекцию в Qdrant и загружает все документы.
        """
        if self.qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
            self.qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)

        self.qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)}
        )

        dense_embeddings = self.embedding_provider.get_dense_embeddings()

        points = []
        for i, doc in enumerate(docs):
            chapter_name = doc.metadata.get("chapter", "")
            article_name = doc.metadata.get("article", "")
            full_context = f"passage: {chapter_name} {article_name} {doc.page_content}"

            dense_vector = dense_embeddings.embed_documents([full_context])[0]

            points.append(
                PointStruct(
                    id=i,
                    vector={"dense": dense_vector},
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
        docs_lenth, docs = self.load_and_split_pdfs()
        self.build_collection(docs)
        return docs_lenth

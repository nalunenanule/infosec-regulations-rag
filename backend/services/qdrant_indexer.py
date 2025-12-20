from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from langchain_core.documents import Document
from config import COLLECTION_NAME
from providers.embeddings_provider import EmbeddingsProvider
from utils.ru_text_utilities import RuTextUtilities

class QdrantIndexer:
    """
    Класс для индексирования документов в Qdrant.
    Поддерживает dense и sparse эмбеддинги.
    """
    
    def __init__(self, client: QdrantClient):
        self.client = client
        self.embedding_provider = EmbeddingsProvider()

    def build_collection(self, docs: list[Document]):
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )

        self.index_documents(docs)

    def index_documents(self, docs: list[Document]):
        dense_embeddings = self.embedding_provider.get_dense_embeddings()
        sparse_model = self.embedding_provider.get_sparse_embeddings()

        dense_vectors = dense_embeddings.embed_documents([f"passage: {d.page_content}" for d in docs])
        sparse_vectors = list(sparse_model.embed([RuTextUtilities().preprocess(d.page_content) for d in docs]))

        points = []
        for i, doc in enumerate(tqdm(docs, desc="Preparing points", unit="doc")):
            points.append(
                PointStruct(
                    id=i,
                    vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i].as_object()},
                    payload={"page_content": doc.page_content, "metadata": doc.metadata}
                )
            )

        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant", unit="batch"):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i+batch_size]
            )

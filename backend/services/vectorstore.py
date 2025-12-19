from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from langchain_core.documents import Document
from config import COLLECTION_NAME
from services.embeddings import get_dense_embeddings, get_sparse_embeddings
from services.preprocessing import preprocess_text_ru

def build_qdrant(client: QdrantClient , docs: list[Document]) -> QdrantClient :
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()}
    )

    index_documents(client, docs)
    return client

def index_documents(client: QdrantClient, docs: list[Document]):
    dense_embeddings = get_dense_embeddings()
    sparse_model = get_sparse_embeddings()

    dense_vectors = dense_embeddings.embed_documents([f"passage: {d.page_content}" for d in docs])
    sparse_vectors = list(sparse_model.embed([preprocess_text_ru(d.page_content) for d in docs]))

    # Подготовка точек
    points = []
    for i, doc in enumerate(tqdm(docs, desc="Preparing points", unit="doc")):
        points.append(
            PointStruct(
                id=i,
                vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i].as_object()},
                payload={"page_content": doc.page_content, "metadata": doc.metadata}
            )
        )

    # Апдейт в Qdrant
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant", unit="batch"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i+batch_size]
        )
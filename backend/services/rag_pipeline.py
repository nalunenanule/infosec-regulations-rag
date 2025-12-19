from langchain_core.documents import Document
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from services.vectorstore import QdrantClient
from services.embeddings import get_dense_embeddings, get_sparse_embeddings
from services.preprocessing import preprocess_text_ru
from config import GIGACHAT_TOKEN, COLLECTION_NAME
from qdrant_client import models

def hybrid_search(client: QdrantClient, query: str, k: int = 5):
    dense_query = get_dense_embeddings().embed_query(f"query: {query}")
    sparse_raw = get_sparse_embeddings().embed(preprocess_text_ru(query)).__next__()
    sparse_query = models.SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=sparse_query, using="sparse", limit=20),
            models.Prefetch(query=dense_query, using="dense", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
    )

    return result.points

def run_qa(query: str, client: QdrantClient):
    hits = hybrid_search(client, query, k=5)

    if not hits:
        return "В документах не найдено релевантной информации."

    docs = [Document(page_content=p.payload["page_content"]) for p in hits]
    context_text = "\n\n".join(d.page_content for d in docs)

    llm = GigaChat(credentials=GIGACHAT_TOKEN, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)

    system_prompt = (
        "Ты — AI-ассистент."
        "Используй только предоставленный контекст. Если информации недостаточно — честно скажи об этом."
    )

    user_message = f"Вопрос:\n{query}\n\nКонтекст:\n{context_text}\n\nОтветь строго на основе контекста."

    chat_payload = Chat(
        messages=[Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                  Messages(role=MessagesRole.USER, content=user_message)],
        temperature=0.1,
        max_tokens=3500
    )

    response = llm.chat(chat_payload)
    return response.choices[0].message.content

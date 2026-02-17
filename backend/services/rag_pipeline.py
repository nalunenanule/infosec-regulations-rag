import re
import json
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from providers.embeddings_provider import EmbeddingsProvider
from utils.ru_text_utilities import RuTextUtilities
from config import GIGACHAT_TOKEN, QDRANT_COLLECTION_NAME

llm_client = GigaChat(
    credentials=GIGACHAT_TOKEN, 
    scope="GIGACHAT_API_PERS", 
    model="GigaChat-2", 
    verify_ssl_certs=False
)

def extract_filters_with_llm(query: str) -> dict:
    """Извлекает номера статей и глав из естественного языка через LLM"""
    prompt = f"""
    Твоя задача — извлечь номер статьи и номер главы из вопроса пользователя по закону 152-ФЗ.
    Верни ответ СТРОГО в формате JSON: {{"article": "номер", "chapter": "номер"}}.
    Если что-то не указано, напиши null. Используй только арабские цифры.
    Пример: "в пятой статье" -> {{"article": "5", "chapter": null}}
    
    Вопрос: {query}
    """
    try:
        res = llm_client.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)], temperature=0.1))
        content = res.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {k: str(v) for k, v in data.items() if v is not None}
    except Exception as e:
        print(f"Ошибка парсинга фильтров: {e}")
    return {}

def hybrid_search(client: QdrantClient, query: str, k: int = 5):
    embedding_provider = EmbeddingsProvider()
    smart_filters = extract_filters_with_llm(query)
    
    conditions = []
    if smart_filters.get("article"):
        conditions.append(
            FieldCondition(key="metadata.article_number", match=MatchValue(value=smart_filters["article"]))
        )
    
    if smart_filters.get("chapter"):
        conditions.append(
            FieldCondition(key="metadata.chapter_number", match=MatchValue(value=smart_filters["chapter"]))
        )

    search_filter = Filter(must=conditions) if conditions else None
    dense_query = embedding_provider.get_dense_embeddings().embed_query(f"query: {query}")

    result = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=dense_query,
        using="dense",
        limit=k,
        query_filter=search_filter,
        with_payload=True
    )

    return result.points

def run_qa(query: str, client: QdrantClient):
    hits = hybrid_search(client, query, k=5)

    if not hits:
        return "В предоставленных документах не найдено информации по вашему запросу."

    formatted_docs = []
    for p in hits:
        meta = p.payload.get("metadata", {})
        chapter = meta.get("chapter", "Глава не указана")
        article = meta.get("article", "Статья не указана")
        content = p.payload.get("page_content", "")
        
        doc_block = f"--- ИСТОЧНИК: {chapter} | {article} ---\n{content}"
        formatted_docs.append(doc_block)

    context_text = "\n\n".join(formatted_docs)

    system_prompt = (
        "Ты — профессиональный юридический ассистент по анализу документов в сфере информационной безопасности\n"
        "Твоя задача: отвечать на вопросы, используя ТОЛЬКО предоставленный контекст.\n"
        "В ответе ОБЯЗАТЕЛЬНО указывай номера статей и глав, на которые ты ссылаешься.\n"
        "Если в контексте нет прямого ответа, честно скажи, что информации недостаточно."
    )

    user_message = f"Вопрос:\n{query}\n\nКонтекст:\n{context_text}\n\nОтветь строго на основе контекста."

    chat_payload = Chat(
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system_prompt),
            Messages(role=MessagesRole.USER, content=user_message)
        ],
        temperature=0.1,
        max_tokens=3000
    )

    response = llm_client.chat(chat_payload)
    return response.choices[0].message.content
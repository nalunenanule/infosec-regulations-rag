# RAG Stack (FastAPI + Qdrant + MinIO + Angular)

Проект представляет собой RAG-приложение со следующим стеком:

- **Backend** — FastAPI 
- **Frontend** — Angular
- **Vector Database** — Qdrant
- **S3-хранилище** — MinIO
- Поддержка эмбеддингов:
  - HuggingFace (локальная модель)
  - GigaChat (через API)

---

## Состав сервисов

| Сервис     | Порт | Описание |
|------------|------|----------|
| backend    | 8000 | FastAPI API |
| frontend   | 8080 | Angular UI |
| qdrant     | 6333 | HTTP API |
| minio      | 9000 | S3 API |
| minio      | 9001 | Web Console |

---

## Быстрый старт

### 1. Создать `.env`

Скопируйте файл:

```bash
cp .env.example .env
```

### 2. Заменить переменные
```
USE_GIGACHAT_EMBEDDINGS=1  # 0 — использовать локальную модель HuggingFace
                           # 1 — использовать модель GigaChat (Сбер)
GIGACHAT_TOKEN=Токен для можно получить на портале разработчиков Сбера: https://developers.sber.ru/
QDRANT_COLLECTION_NAME=my_collection
QDRANT_URL=http://qdrant:6333
DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
S3_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET_NAME=my-bucket
```

### 3. Запуск

```bash
docker compose up --build
```
"""FastAPI endpoints for the RAG chatbot.

HTTP and streaming routes with RESTful API design and async request handling.
Supports Server-Sent Events for real-time chat streaming.

Endpoints:
    - GET /health: Service health status
    - POST /chat: Chat completion requests
    - POST /upload: Document uploads for knowledge base
    - GET /sessions/{id}: Chat session history
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Retrieval-Augmented Generation API for intelligent document Q&A. "
        "Ingests PDF documents, generates embeddings, and provides context-aware "
        "responses through LLM orchestration. Supports streaming responses and "
        "session management."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Check service health status.

    Returns:
        Service status indicating operational readiness.
    """
    return {"status": "healthy", "service": "rag-chatbot"}

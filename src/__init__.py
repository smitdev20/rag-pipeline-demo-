"""RAG Chatbot - Retrieval-Augmented Generation for intelligent document Q&A.

Combines FastAPI for HTTP streaming, Agno for agent orchestration,
NiceGUI for visualization, and Pydantic for data validation.

Components:
    - api: HTTP endpoints and streaming responses
    - agent: LLM orchestration with memory and knowledge
    - parsing: PDF extraction and text processing
    - ui: Web interface for chat interactions
    - models: Request/response schemas
"""

__version__ = "0.1.0"

"""Agno agent logic for LLM orchestration.

Handles Retrieval-Augmented Generation with sessions, memory, and knowledge base integration.

Responsibilities:
    - Agent initialization with OpenAI models
    - Knowledge base queries from PDF embeddings
    - Conversation context and session state management
    - Streaming token generation coordination

Leverages the Agno framework for agent lifecycle management.
Maintains clean separation from the HTTP layer.
"""

"""
Database adapters for various vector databases.

This module provides unified interfaces for:
    - FAISS (Meta's similarity search library)
    - Qdrant (Rust-based vector database)
    - Milvus (Distributed vector database)
    - LanceDB (Columnar vector database)
    - Weaviate (AI-native vector database)
    - Chroma (Embedding database)
    - pgvector (PostgreSQL extension)
"""

from src.databases.factory import (
    DatabaseFactory,
    get_database,
    list_available_databases,
    register_database,
)

__all__ = [
    "DatabaseFactory",
    "get_database",
    "list_available_databases",
    "register_database",
]

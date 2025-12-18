"""
Factory pattern for creating vector database adapters.

This module provides a centralized way to instantiate database adapters
based on configuration, enabling easy switching between different backends.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from src.core.base import VectorDBInterface
from src.core.config import load_database_config


# Registry of available database adapters
_DATABASE_REGISTRY: Dict[str, Type[VectorDBInterface]] = {}


def register_database(name: str) -> Callable:
    """
    Decorator to register a database adapter class.

    Args:
        name: Name to register the database under

    Returns:
        Decorator function

    Example:
        @register_database("faiss")
        class FAISSAdapter(VectorDBInterface):
            ...
    """

    def decorator(cls: Type[VectorDBInterface]) -> Type[VectorDBInterface]:
        _DATABASE_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_database(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> VectorDBInterface:
    """
    Get a database adapter instance.

    Args:
        name: Name of the database (faiss, qdrant, milvus, etc.)
        config: Optional configuration dictionary (overrides file config)
        config_dir: Optional path to configuration directory

    Returns:
        Instance of the database adapter

    Raises:
        ValueError: If database is not registered
    """
    name_lower = name.lower()

    if name_lower not in _DATABASE_REGISTRY:
        # Try to import the adapter module
        _import_adapter(name_lower)

    if name_lower not in _DATABASE_REGISTRY:
        available = ", ".join(list_available_databases())
        raise ValueError(
            f"Unknown database: {name}. Available databases: {available}"
        )

    # Load configuration from file if not provided
    if config is None:
        try:
            config = load_database_config(name_lower, config_dir)
        except FileNotFoundError:
            config = {}

    # Create and return adapter instance
    adapter_class = _DATABASE_REGISTRY[name_lower]
    return adapter_class(config)


def list_available_databases() -> List[str]:
    """
    List all registered database adapters.

    Returns:
        List of database names
    """
    # Ensure all adapters are imported
    _import_all_adapters()
    return sorted(_DATABASE_REGISTRY.keys())


def _import_adapter(name: str) -> None:
    """
    Import a database adapter module by name.

    Args:
        name: Name of the database
    """
    adapter_modules = {
        "faiss": "src.databases.faiss_adapter",
        "qdrant": "src.databases.qdrant_adapter",
        "milvus": "src.databases.milvus_adapter",
        "lancedb": "src.databases.lancedb_adapter",
        "weaviate": "src.databases.weaviate_adapter",
        "chroma": "src.databases.chroma_adapter",
        "pgvector": "src.databases.pgvector_adapter",
    }

    module_name = adapter_modules.get(name.lower())
    if module_name:
        try:
            __import__(module_name)
        except ImportError as e:
            # Log warning but don't fail - the database may not be installed
            import warnings

            warnings.warn(f"Could not import {name} adapter: {e}")


def _import_all_adapters() -> None:
    """Import all available database adapter modules."""
    for name in ["faiss", "qdrant", "milvus", "lancedb", "weaviate", "chroma", "pgvector"]:
        _import_adapter(name)


class DatabaseFactory:
    """
    Factory class for creating database adapters.

    This provides an object-oriented interface to the factory functions,
    useful for dependency injection patterns.

    Example:
        factory = DatabaseFactory(config_dir="./config")
        db = factory.create("faiss")
        with db:
            db.create_index(vectors, config)
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the factory.

        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = config_dir
        self._instances: Dict[str, VectorDBInterface] = {}

    def create(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        cache: bool = False,
    ) -> VectorDBInterface:
        """
        Create a database adapter instance.

        Args:
            name: Name of the database
            config: Optional configuration override
            cache: Whether to cache and reuse the instance

        Returns:
            Database adapter instance
        """
        if cache and name in self._instances:
            return self._instances[name]

        instance = get_database(name, config, self.config_dir)

        if cache:
            self._instances[name] = instance

        return instance

    def get_cached(self, name: str) -> Optional[VectorDBInterface]:
        """
        Get a cached database instance.

        Args:
            name: Name of the database

        Returns:
            Cached instance or None
        """
        return self._instances.get(name)

    def clear_cache(self) -> None:
        """Clear all cached database instances."""
        for instance in self._instances.values():
            if instance.is_connected:
                instance.disconnect()
        self._instances.clear()

    @staticmethod
    def list_available() -> List[str]:
        """List available database adapters."""
        return list_available_databases()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - clean up cached instances."""
        self.clear_cache()
        return False

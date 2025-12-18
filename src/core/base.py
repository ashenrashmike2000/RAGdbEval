"""
Abstract base class for vector database interfaces.

This module defines the common interface that all vector database adapters must implement,
enabling consistent benchmarking across different databases.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import (
    DatabaseInfo,
    DistanceMetric,
    FilterCondition,
    IndexConfig,
    SearchResult,
)


class VectorDBInterface(ABC):
    """
    Abstract base class for vector database adapters.

    All vector database implementations must inherit from this class and implement
    the required abstract methods. This ensures consistent behavior across different
    databases and enables the benchmarking framework to work with any supported database.

    Attributes:
        name: Name of the database
        config: Database-specific configuration
        is_connected: Whether the database connection is active
        dimensions: Dimensionality of vectors (set after index creation)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database adapter.

        Args:
            config: Database-specific configuration dictionary
        """
        self.config = config
        self._is_connected = False
        self._dimensions: Optional[int] = None
        self._index_config: Optional[IndexConfig] = None
        self._distance_metric: Optional[DistanceMetric] = None
        self._num_vectors: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the database."""
        pass

    @property
    @abstractmethod
    def info(self) -> DatabaseInfo:
        """Return database information."""
        pass

    @property
    def is_connected(self) -> bool:
        """Return whether the database connection is active."""
        return self._is_connected

    @property
    def dimensions(self) -> Optional[int]:
        """Return the dimensionality of indexed vectors."""
        return self._dimensions

    @property
    def num_vectors(self) -> int:
        """Return the number of indexed vectors."""
        return self._num_vectors

    # =========================================================================
    # Connection Management
    # =========================================================================

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the database.

        For in-process libraries (like FAISS), this may be a no-op.
        For client-server databases, this establishes the connection.

        Raises:
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection.

        Should release all resources and clean up.
        """
        pass

    @contextmanager
    def connection(self) -> Generator[None, None, None]:
        """
        Context manager for database connection.

        Example:
            with db.connection():
                db.create_index(vectors, config)
                results = db.search(queries, k=10)
        """
        try:
            self.connect()
            yield
        finally:
            self.disconnect()

    # =========================================================================
    # Index Management
    # =========================================================================

    @abstractmethod
    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """
        Create an index from vectors.

        Args:
            vectors: Numpy array of shape (n, d) containing vectors
            index_config: Configuration for the index type
            distance_metric: Distance metric to use
            metadata: Optional list of metadata dicts for each vector
            ids: Optional list of vector IDs (default: 0 to n-1)

        Returns:
            Index build time in seconds

        Raises:
            ValueError: If vectors shape is invalid
            RuntimeError: If index creation fails
        """
        pass

    @abstractmethod
    def delete_index(self) -> None:
        """
        Delete the current index and free resources.
        """
        pass

    def index_exists(self) -> bool:
        """
        Check if an index currently exists.

        Returns:
            True if an index exists, False otherwise
        """
        return self._num_vectors > 0

    @abstractmethod
    def save_index(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path: Path to save the index

        Raises:
            RuntimeError: If save fails
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> float:
        """
        Load an index from disk.

        Args:
            path: Path to load the index from

        Returns:
            Load time in seconds

        Raises:
            FileNotFoundError: If index file doesn't exist
            RuntimeError: If load fails
        """
        pass

    # =========================================================================
    # Search Operations
    # =========================================================================

    @abstractmethod
    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """
        Search for k nearest neighbors.

        Args:
            queries: Numpy array of shape (n_queries, d) containing query vectors
            k: Number of nearest neighbors to return
            search_params: Optional search parameters (e.g., nprobe, ef)
            filters: Optional metadata filters

        Returns:
            Tuple of:
                - indices: Array of shape (n_queries, k) with neighbor indices
                - distances: Array of shape (n_queries, k) with distances
                - latencies: List of per-query latencies in milliseconds

        Raises:
            ValueError: If queries shape is invalid or k <= 0
            RuntimeError: If search fails
        """
        pass

    def search_single(
        self,
        query: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> SearchResult:
        """
        Search for k nearest neighbors of a single query.

        Args:
            query: 1D numpy array of shape (d,) containing the query vector
            k: Number of nearest neighbors to return
            search_params: Optional search parameters
            filters: Optional metadata filters

        Returns:
            SearchResult with indices, distances, and latency
        """
        query_2d = query.reshape(1, -1)
        indices, distances, latencies = self.search(query_2d, k, search_params, filters)
        return SearchResult(
            query_id=0,
            indices=indices[0],
            distances=distances[0],
            latency_ms=latencies[0],
        )

    # =========================================================================
    # CRUD Operations (for operational metrics)
    # =========================================================================

    @abstractmethod
    def insert(
        self,
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """
        Insert vectors into an existing index.

        Args:
            vectors: Numpy array of shape (n, d) containing vectors to insert
            metadata: Optional list of metadata dicts
            ids: Optional list of vector IDs

        Returns:
            Insert time in seconds

        Raises:
            RuntimeError: If index doesn't exist or insert fails
        """
        pass

    @abstractmethod
    def update(
        self,
        ids: List[int],
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Update existing vectors.

        Args:
            ids: List of vector IDs to update
            vectors: New vectors
            metadata: Optional new metadata

        Returns:
            Update time in seconds

        Raises:
            RuntimeError: If update fails
        """
        pass

    @abstractmethod
    def delete(self, ids: List[int]) -> float:
        """
        Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Delete time in seconds

        Raises:
            RuntimeError: If delete fails
        """
        pass

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    @abstractmethod
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary containing:
                - num_vectors: Number of indexed vectors
                - dimensions: Vector dimensionality
                - index_size_bytes: Size of index in memory
                - disk_size_bytes: Size of index on disk (if applicable)
                - index_type: Type of index
                - distance_metric: Distance metric used
        """
        pass

    def get_memory_usage(self) -> int:
        """
        Get current memory usage of the index in bytes.

        Returns:
            Memory usage in bytes
        """
        stats = self.get_index_stats()
        return stats.get("index_size_bytes", 0)

    # =========================================================================
    # Configuration and Tuning
    # =========================================================================

    @abstractmethod
    def set_search_params(self, params: Dict[str, Any]) -> None:
        """
        Set search parameters for subsequent queries.

        Args:
            params: Database-specific search parameters
        """
        pass

    @abstractmethod
    def get_search_params(self) -> Dict[str, Any]:
        """
        Get current search parameters.

        Returns:
            Dictionary of current search parameters
        """
        pass

    def get_optimal_params_for_recall(
        self,
        target_recall: float,
        queries: NDArray[np.float32],
        ground_truth: List[List[int]],
        k: int = 10,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Find optimal search parameters to achieve target recall.

        This performs a parameter search to find settings that achieve
        the target recall with minimum latency.

        Args:
            target_recall: Target recall value (0.0-1.0)
            queries: Query vectors for evaluation
            ground_truth: Ground truth neighbors for queries
            k: Number of neighbors
            param_grid: Grid of parameters to search

        Returns:
            Tuple of (optimal_params, achieved_recall)
        """
        raise NotImplementedError(
            f"{self.name} does not implement automatic parameter tuning"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __enter__(self):
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.disconnect()
        return False

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"connected={self.is_connected}, "
            f"num_vectors={self.num_vectors})"
        )

    def validate_vectors(self, vectors: NDArray[np.float32]) -> None:
        """
        Validate vector array shape and type.

        Args:
            vectors: Vector array to validate

        Raises:
            ValueError: If vectors are invalid
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be a numpy array")

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D, got {vectors.ndim}D")

        if vectors.dtype != np.float32:
            raise ValueError(f"Vectors must be float32, got {vectors.dtype}")

        if self._dimensions is not None and vectors.shape[1] != self._dimensions:
            raise ValueError(
                f"Vector dimensions mismatch: expected {self._dimensions}, "
                f"got {vectors.shape[1]}"
            )

    def normalize_vectors(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        L2-normalize vectors for cosine similarity.

        Args:
            vectors: Vector array to normalize

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

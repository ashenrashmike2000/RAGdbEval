"""
Chroma vector database adapter.

Chroma is an AI-native open-source embedding database designed for
simplicity and ease of use with LLM applications.

Documentation: https://docs.trychroma.com/
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.base import VectorDBInterface
from src.core.types import (
    DatabaseInfo,
    DatabaseType,
    DistanceMetric,
    FilterCondition,
    IndexConfig,
    IndexType,
)
from src.databases.factory import register_database

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None


@register_database("chroma")
class ChromaAdapter(VectorDBInterface):
    """Chroma vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not installed. Install with: pip install chromadb")
        super().__init__(config)
        self._client = None
        self._collection = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "chroma"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="chroma",
            display_name="Chroma",
            version="0.4.0",
            type=DatabaseType.DATABASE,
            language="Python",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=False,
            supports_gpu=False,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Connect to Chroma."""
        conn_config = self.config.get("connection", {})
        mode = conn_config.get("mode", "persistent")

        if mode == "ephemeral":
            self._client = chromadb.Client()
        elif mode == "persistent":
            path = conn_config.get("persistent", {}).get("path", "./data/chroma")
            self._client = chromadb.PersistentClient(path=path)
        else:  # http mode
            http_config = conn_config.get("http", {})
            self._client = chromadb.HttpClient(
                host=http_config.get("host", "localhost"),
                port=http_config.get("port", 8000),
            )
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Chroma."""
        self._client = None
        self._collection = None
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create Chroma collection and add vectors."""
        self.validate_vectors(vectors)
        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        start_time = time.perf_counter()

        # Map distance metric
        distance_map = {
            DistanceMetric.L2: "l2",
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.IP: "ip",
        }

        # HNSW parameters
        params = index_config.params
        hnsw_params = {
            "hnsw:space": params.get("space", distance_map.get(distance_metric, "l2")),
            "hnsw:construction_ef": params.get("construction_ef", 100),
            "hnsw:M": params.get("M", 16),
            "hnsw:search_ef": params.get("search_ef", 100),
        }

        # Create collection
        self._collection_name = f"benchmark_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata=hnsw_params,
        )

        # Prepare IDs
        if ids is None:
            ids = list(range(n_vectors))
        str_ids = [str(i) for i in ids]

        # Add vectors in batches
        batch_size = self.config.get("batch", {}).get("insert_batch_size", 5000)
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch_ids = str_ids[i:end_idx]
            batch_embeddings = vectors[i:end_idx].tolist()
            batch_metadata = metadata[i:end_idx] if metadata else None

            self._collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
            )

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        """Delete the collection."""
        if self._client and self._collection_name:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._num_vectors = 0

    def save_index(self, path: str) -> None:
        """Chroma persists automatically in persistent mode."""
        pass

    def load_index(self, path: str) -> float:
        """Load existing collection."""
        start_time = time.perf_counter()
        self._collection = self._client.get_collection(path)
        self._num_vectors = self._collection.count()
        return time.perf_counter() - start_time

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """Search for nearest neighbors."""
        if self._collection is None:
            raise RuntimeError("Collection not created")

        self.validate_vectors(queries)

        all_indices = []
        all_distances = []
        latencies = []

        # Build where filter
        where_filter = self._build_filter(filters) if filters else None

        for query in queries:
            start_time = time.perf_counter()

            results = self._collection.query(
                query_embeddings=[query.tolist()],
                n_results=k,
                where=where_filter,
                include=["distances"],
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            # Extract results
            result_ids = results["ids"][0] if results["ids"] else []
            result_distances = results["distances"][0] if results["distances"] else []

            indices = [int(i) for i in result_ids]
            distances = list(result_distances)

            while len(indices) < k:
                indices.append(-1)
                distances.append(float("inf"))

            all_indices.append(indices[:k])
            all_distances.append(distances[:k])

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies,
        )

    def _build_filter(self, filters: List[FilterCondition]) -> Optional[Dict]:
        """Build Chroma where filter."""
        if not filters:
            return None

        conditions = []
        for f in filters:
            op_map = {
                "eq": "$eq",
                "ne": "$ne",
                "gt": "$gt",
                "gte": "$gte",
                "lt": "$lt",
                "lte": "$lte",
                "in": "$in",
            }
            chroma_op = op_map.get(f.operator.lower())
            if chroma_op:
                conditions.append({f.field: {chroma_op: f.value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def insert(
        self,
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Insert vectors."""
        if ids is None:
            ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        str_ids = [str(i) for i in ids]

        start_time = time.perf_counter()
        self._collection.add(
            ids=str_ids,
            embeddings=vectors.tolist(),
            metadatas=metadata,
        )
        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(
        self,
        ids: List[int],
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Update vectors."""
        str_ids = [str(i) for i in ids]
        start_time = time.perf_counter()
        self._collection.update(
            ids=str_ids,
            embeddings=vectors.tolist(),
            metadatas=metadata,
        )
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        """Delete vectors."""
        str_ids = [str(i) for i in ids]
        start_time = time.perf_counter()
        self._collection.delete(ids=str_ids)
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self._collection is None:
            return {}
        return {
            "num_vectors": self._collection.count(),
            "dimensions": self._dimensions,
            "index_type": "HNSW",
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

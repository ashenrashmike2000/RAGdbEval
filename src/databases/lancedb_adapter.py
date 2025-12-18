"""
LanceDB vector database adapter.

LanceDB is a serverless vector database built on the Lance columnar format,
offering efficient disk-based storage and fast vector search.

Documentation: https://lancedb.github.io/lancedb/
"""

import os
import time
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
    import lancedb
    import pyarrow as pa

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None


@register_database("lancedb")
class LanceDBAdapter(VectorDBInterface):
    """LanceDB vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not installed. Install with: pip install lancedb")
        super().__init__(config)
        self._db = None
        self._table = None
        self._table_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="lancedb",
            display_name="LanceDB",
            version="0.4.0",
            type=DatabaseType.DATABASE,
            language="Rust",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.IVF, IndexType.IVFPQ],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=False,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Connect to LanceDB."""
        conn_config = self.config.get("connection", {})
        uri = conn_config.get("uri", "./data/lancedb")
        os.makedirs(uri, exist_ok=True)
        self._db = lancedb.connect(uri)
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from LanceDB."""
        self._db = None
        self._table = None
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create LanceDB table and index."""
        self.validate_vectors(vectors)
        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        start_time = time.perf_counter()

        # Prepare data
        if ids is None:
            ids = list(range(n_vectors))

        data = [{"id": i, "vector": v.tolist()} for i, v in zip(ids, vectors)]

        # Add metadata
        if metadata:
            for i, m in enumerate(metadata):
                data[i].update(m)

        # Create table
        table_prefix = self.config.get("table", {}).get("name_prefix", "benchmark")
        self._table_name = f"{table_prefix}_{int(time.time())}"
        self._table = self._db.create_table(self._table_name, data, mode="overwrite")

        # Create index if not flat
        params = index_config.params
        index_type = index_config.type.upper()

        if index_type not in ("FLAT", "NONE"):
            metric_map = {
                DistanceMetric.L2: "L2",
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.IP: "dot",
            }

            if index_type in ("IVF_PQ", "IVFPQ"):
                self._table.create_index(
                    metric=metric_map.get(distance_metric, "L2"),
                    num_partitions=params.get("num_partitions", 256),
                    num_sub_vectors=params.get("num_sub_vectors", 16),
                )

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        """Delete the table."""
        if self._db and self._table_name:
            try:
                self._db.drop_table(self._table_name)
            except Exception:
                pass
        self._table = None
        self._num_vectors = 0

    def save_index(self, path: str) -> None:
        """LanceDB persists automatically."""
        pass

    def load_index(self, path: str) -> float:
        """Load existing table."""
        start_time = time.perf_counter()
        self._table = self._db.open_table(path)
        self._num_vectors = len(self._table)
        return time.perf_counter() - start_time

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """Search for nearest neighbors."""
        if self._table is None:
            raise RuntimeError("Table not created")

        self.validate_vectors(queries)
        params = search_params or {}

        all_indices = []
        all_distances = []
        latencies = []

        for query in queries:
            start_time = time.perf_counter()

            search_query = self._table.search(query.tolist())

            if params.get("nprobes"):
                search_query = search_query.nprobes(params["nprobes"])

            if params.get("refine_factor"):
                search_query = search_query.refine_factor(params["refine_factor"])

            if filters:
                where_clause = self._build_filter(filters)
                if where_clause:
                    search_query = search_query.where(where_clause)

            results = search_query.limit(k).to_pandas()

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            indices = results["id"].tolist() if "id" in results.columns else list(range(len(results)))
            distances = results["_distance"].tolist() if "_distance" in results.columns else [0.0] * len(results)

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

    def _build_filter(self, filters: List[FilterCondition]) -> Optional[str]:
        """Build SQL-like filter string."""
        if not filters:
            return None

        conditions = []
        for f in filters:
            if f.operator == "eq":
                if isinstance(f.value, str):
                    conditions.append(f"{f.field} = '{f.value}'")
                else:
                    conditions.append(f"{f.field} = {f.value}")
            elif f.operator == "gt":
                conditions.append(f"{f.field} > {f.value}")
            elif f.operator == "gte":
                conditions.append(f"{f.field} >= {f.value}")
            elif f.operator == "lt":
                conditions.append(f"{f.field} < {f.value}")
            elif f.operator == "lte":
                conditions.append(f"{f.field} <= {f.value}")

        return " AND ".join(conditions) if conditions else None

    def insert(
        self,
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Insert vectors."""
        if ids is None:
            ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))

        data = [{"id": i, "vector": v.tolist()} for i, v in zip(ids, vectors)]
        if metadata:
            for i, m in enumerate(metadata):
                data[i].update(m)

        start_time = time.perf_counter()
        self._table.add(data)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(
        self,
        ids: List[int],
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Update vectors (delete + insert)."""
        start_time = time.perf_counter()
        self.delete(ids)
        self.insert(vectors, metadata, ids)
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        """Delete vectors by ID."""
        start_time = time.perf_counter()
        id_list = ", ".join(str(i) for i in ids)
        self._table.delete(f"id IN ({id_list})")
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        """Get table statistics."""
        if self._table is None:
            return {}
        return {
            "num_vectors": len(self._table),
            "dimensions": self._dimensions,
            "index_type": self._index_config.type if self._index_config else None,
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

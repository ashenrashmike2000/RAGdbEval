"""
Milvus vector database adapter.

Milvus is a distributed vector database designed for scalable similarity search,
supporting multiple index types and GPU acceleration.

Documentation: https://milvus.io/docs
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
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
        connections,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


@register_database("milvus")
class MilvusAdapter(VectorDBInterface):
    """Milvus vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus not installed. Install with: pip install pymilvus")
        super().__init__(config)
        self._collection: Optional[Collection] = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "milvus"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="milvus",
            display_name="Milvus",
            version="2.3.0",
            type=DatabaseType.DATABASE,
            language="Go/C++",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[
                IndexType.FLAT, IndexType.IVF, IndexType.IVFPQ,
                IndexType.IVFSQ, IndexType.HNSW, IndexType.DISKANN
            ],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=True,
            is_distributed=True,
        )

    def connect(self) -> None:
        """Connect to Milvus server."""
        conn_config = self.config.get("connection", {})
        connections.connect(
            alias="default",
            host=conn_config.get("host", "localhost"),
            port=conn_config.get("port", 19530),
            user=conn_config.get("user", ""),
            password=conn_config.get("password", ""),
        )
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        connections.disconnect("default")
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create Milvus collection and index."""
        self.validate_vectors(vectors)
        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        start_time = time.perf_counter()

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
        ]

        # Add metadata fields
        schema_config = self.config.get("schema", {}).get("metadata_fields", [])
        for field_config in schema_config:
            dtype_map = {
                "VARCHAR": DataType.VARCHAR,
                "FLOAT": DataType.FLOAT,
                "INT64": DataType.INT64,
                "BOOL": DataType.BOOL,
            }
            dtype = dtype_map.get(field_config.get("type", "VARCHAR"), DataType.VARCHAR)
            field_params = {"max_length": 64} if dtype == DataType.VARCHAR else {}
            fields.append(FieldSchema(
                name=field_config["name"],
                dtype=dtype,
                **field_params
            ))

        schema = CollectionSchema(fields=fields, description="Benchmark collection")

        # Create collection
        self._collection_name = f"benchmark_{uuid.uuid4().hex[:8]}"
        self._collection = Collection(name=self._collection_name, schema=schema)

        # Prepare data
        if ids is None:
            ids = list(range(n_vectors))

        insert_data = [ids, vectors.tolist()]

        # Add metadata columns
        if metadata and schema_config:
            for field_config in schema_config:
                field_name = field_config["name"]
                field_values = [m.get(field_name, "") for m in metadata]
                insert_data.append(field_values)

        # Insert data
        batch_size = self.config.get("batch", {}).get("insert_batch_size", 10000)
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch_data = [d[i:end_idx] if isinstance(d, list) else d[i:end_idx].tolist() for d in insert_data]
            self._collection.insert(batch_data)

        # Create index
        metric_map = {
            DistanceMetric.L2: "L2",
            DistanceMetric.COSINE: "COSINE",
            DistanceMetric.IP: "IP",
        }

        params = index_config.params
        index_type = index_config.type.upper()

        index_params = {
            "index_type": index_type,
            "metric_type": metric_map.get(distance_metric, "L2"),
            "params": params,
        }

        self._collection.create_index(field_name="vector", index_params=index_params)
        self._collection.load()

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        """Delete the collection."""
        if self._collection_name:
            try:
                utility.drop_collection(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._num_vectors = 0

    def save_index(self, path: str) -> None:
        """Milvus manages persistence internally."""
        pass

    def load_index(self, path: str) -> float:
        """Load existing collection."""
        start_time = time.perf_counter()
        self._collection = Collection(name=path)
        self._collection.load()
        self._num_vectors = self._collection.num_entities
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
        params = search_params or {}

        # Build search params based on index type
        index_type = self._index_config.type.upper() if self._index_config else "HNSW"

        if index_type in ("IVF_FLAT", "IVF", "IVF_SQ8", "IVF_PQ"):
            milvus_params = {"nprobe": params.get("nprobe", 16)}
        elif index_type == "HNSW":
            milvus_params = {"ef": params.get("ef", 128)}
        else:
            milvus_params = params

        # Build filter expression
        expr = self._build_filter(filters) if filters else None

        all_indices = []
        all_distances = []
        latencies = []

        for query in queries:
            start_time = time.perf_counter()

            results = self._collection.search(
                data=[query.tolist()],
                anns_field="vector",
                param=milvus_params,
                limit=k,
                expr=expr,
                output_fields=["id"],
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            if results and len(results) > 0:
                hits = results[0]
                indices = [hit.id for hit in hits]
                distances = [hit.distance for hit in hits]
            else:
                indices, distances = [], []

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
        """Build Milvus filter expression."""
        if not filters:
            return None

        expressions = []
        for f in filters:
            if f.operator == "eq":
                if isinstance(f.value, str):
                    expressions.append(f'{f.field} == "{f.value}"')
                else:
                    expressions.append(f"{f.field} == {f.value}")
            elif f.operator == "gt":
                expressions.append(f"{f.field} > {f.value}")
            elif f.operator == "gte":
                expressions.append(f"{f.field} >= {f.value}")
            elif f.operator == "lt":
                expressions.append(f"{f.field} < {f.value}")
            elif f.operator == "lte":
                expressions.append(f"{f.field} <= {f.value}")
            elif f.operator == "in":
                expressions.append(f"{f.field} in {f.value}")

        return " and ".join(expressions) if expressions else None

    def insert(self, vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[int]] = None) -> float:
        if ids is None:
            ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        start_time = time.perf_counter()
        self._collection.insert([ids, vectors.tolist()])
        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None) -> float:
        start_time = time.perf_counter()
        self._collection.delete(f"id in {ids}")
        self._collection.insert([ids, vectors.tolist()])
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        start_time = time.perf_counter()
        self._collection.delete(f"id in {ids}")
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        if self._collection is None:
            return {}
        return {
            "num_vectors": self._collection.num_entities,
            "dimensions": self._dimensions,
            "index_type": self._index_config.type if self._index_config else None,
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

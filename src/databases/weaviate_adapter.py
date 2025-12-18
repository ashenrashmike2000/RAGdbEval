"""
Weaviate vector database adapter.

Weaviate is an AI-native vector database with GraphQL API, hybrid search,
and module ecosystem for embedding generation.

Documentation: https://weaviate.io/developers/weaviate
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
    import weaviate
    from weaviate.classes import config as wvc
    from weaviate.classes.query import Filter, MetadataQuery

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


@register_database("weaviate")
class WeaviateAdapter(VectorDBInterface):
    """Weaviate vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not installed. Install with: pip install weaviate-client")
        super().__init__(config)
        self._client = None
        self._collection = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "weaviate"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="weaviate",
            display_name="Weaviate",
            version="4.0.0",
            type=DatabaseType.DATABASE,
            language="Go",
            license="BSD-3-Clause",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=False,
            is_distributed=True,
        )

    def connect(self) -> None:
        """Connect to Weaviate server."""
        conn_config = self.config.get("connection", {})
        host = conn_config.get("host", "localhost")
        http_port = conn_config.get("http_port", 8080)
        grpc_port = conn_config.get("grpc_port", 50051)

        self._client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=grpc_port,
        )
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self._client:
            self._client.close()
        self._client = None
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create Weaviate collection and index."""
        self.validate_vectors(vectors)
        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        start_time = time.perf_counter()

        # Map distance metric
        distance_map = {
            DistanceMetric.L2: wvc.VectorDistances.L2_SQUARED,
            DistanceMetric.COSINE: wvc.VectorDistances.COSINE,
            DistanceMetric.IP: wvc.VectorDistances.DOT,
        }

        # HNSW config
        params = index_config.params
        vector_index = wvc.Configure.VectorIndex.hnsw(
            ef_construction=params.get("efConstruction", 128),
            max_connections=params.get("maxConnections", 32),
            ef=params.get("ef", 100),
            distance_metric=distance_map.get(distance_metric, wvc.VectorDistances.COSINE),
        )

        # Define properties for metadata
        properties = [
            wvc.Property(name="vec_id", data_type=wvc.DataType.INT),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
            wvc.Property(name="price", data_type=wvc.DataType.NUMBER),
            wvc.Property(name="timestamp", data_type=wvc.DataType.INT),
            wvc.Property(name="active", data_type=wvc.DataType.BOOL),
        ]

        # Create collection
        self._collection_name = f"Benchmark_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.collections.create(
            name=self._collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=vector_index,
            properties=properties,
        )

        # Insert data
        if ids is None:
            ids = list(range(n_vectors))

        batch_size = self.config.get("batch", {}).get("insert_batch_size", 1000)

        with self._collection.batch.dynamic() as batch:
            for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
                props = {"vec_id": vec_id}
                if metadata and i < len(metadata):
                    props.update(metadata[i])

                batch.add_object(
                    properties=props,
                    vector=vector.tolist(),
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id))),
                )

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        """Delete the collection."""
        if self._client and self._collection_name:
            try:
                self._client.collections.delete(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._num_vectors = 0

    def save_index(self, path: str) -> None:
        """Weaviate manages persistence internally."""
        pass

    def load_index(self, path: str) -> float:
        """Load existing collection."""
        start_time = time.perf_counter()
        self._collection = self._client.collections.get(path)
        self._collection_name = path
        self._num_vectors = self._collection.aggregate.over_all(total_count=True).total_count
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

        weaviate_filter = self._build_filter(filters) if filters else None

        for query in queries:
            start_time = time.perf_counter()

            results = self._collection.query.near_vector(
                near_vector=query.tolist(),
                limit=k,
                filters=weaviate_filter,
                return_metadata=MetadataQuery(distance=True),
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            indices = []
            distances = []

            for obj in results.objects:
                vec_id = obj.properties.get("vec_id", -1)
                indices.append(vec_id)
                distances.append(obj.metadata.distance if obj.metadata else 0.0)

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

    def _build_filter(self, filters: List[FilterCondition]):
        """Build Weaviate filter."""
        if not filters:
            return None

        conditions = []
        for f in filters:
            field_filter = Filter.by_property(f.field)
            if f.operator == "eq":
                conditions.append(field_filter.equal(f.value))
            elif f.operator == "gt":
                conditions.append(field_filter.greater_than(f.value))
            elif f.operator == "gte":
                conditions.append(field_filter.greater_or_equal(f.value))
            elif f.operator == "lt":
                conditions.append(field_filter.less_than(f.value))
            elif f.operator == "lte":
                conditions.append(field_filter.less_or_equal(f.value))

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            result = conditions[0]
            for c in conditions[1:]:
                result = result & c
            return result
        return None

    def insert(self, vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[int]] = None) -> float:
        if ids is None:
            ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))

        start_time = time.perf_counter()
        with self._collection.batch.dynamic() as batch:
            for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
                props = {"vec_id": vec_id}
                if metadata and i < len(metadata):
                    props.update(metadata[i])
                batch.add_object(properties=props, vector=vector.tolist())

        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None) -> float:
        start_time = time.perf_counter()
        for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id)))
            props = {"vec_id": vec_id}
            if metadata and i < len(metadata):
                props.update(metadata[i])
            self._collection.data.update(uuid=obj_uuid, properties=props, vector=vector.tolist())
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        start_time = time.perf_counter()
        for vec_id in ids:
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id)))
            self._collection.data.delete_by_id(obj_uuid)
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        if self._collection is None:
            return {}
        count = self._collection.aggregate.over_all(total_count=True).total_count
        return {"num_vectors": count, "dimensions": self._dimensions, "index_type": "HNSW"}

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

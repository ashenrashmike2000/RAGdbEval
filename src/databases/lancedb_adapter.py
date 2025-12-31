"""
LanceDB vector database adapter.
"""

import time
import uuid
import shutil
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd

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
        self._db_path = config.get("connection", {}).get("uri", "./lancedb_data")

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="lancedb",
            display_name="LanceDB",
            version="0.4.x",
            type=DatabaseType.DATABASE,
            language="Rust/Python",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE],
            supported_index_types=[IndexType.IVF_PQ], # LanceDB uses IVF-PQ, not HNSW
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=True,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Connect to LanceDB (Embedded)."""
        self._db = lancedb.connect(self._db_path)
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create a table and index vectors."""
        self.validate_vectors(vectors)

        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        prefix = self.config.get("collection", {}).get("name_prefix", "benchmark")
        self._table_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()

        print(f"🚀 LanceDB: Ingesting {n_vectors} vectors (PyArrow Table Mode)...")

        vector_ids = ids if ids is not None else list(range(n_vectors))

        # 1. Create ID Array (Converted to STRING to support UUIDs later)
        pa_ids = pa.array([str(i) for i in vector_ids])

        # 2. Create Vector Array (FixedSizeList)
        flat_vectors = vectors.reshape(-1)
        pa_vectors = pa.FixedSizeListArray.from_arrays(pa.array(flat_vectors), dimensions)

        # 3. Create Table Schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimensions))
        ])

        data = pa.Table.from_arrays([pa_ids, pa_vectors], schema=schema)

        # Drop if exists
        if self._table_name in self._db.table_names():
            self._db.drop_table(self._table_name)

        self._table = self._db.create_table(self._table_name, data=data)

        # === FIX: PARAMETER MAPPING (Generic -> LanceDB) ===
        # LanceDB uses 'num_partitions' (IVF) and 'num_sub_vectors' (PQ).
        # We must map your benchmark's 'ef_construct' and 'm' to these concepts.

        params = index_config.params or {}

        # 1. Map ef_construct -> num_partitions (Search Space Granularity)
        # Default rule: sqrt(n_vectors)
        default_partitions = int(np.sqrt(n_vectors))
        num_partitions = default_partitions

        if "ef_construct" in params:
            # Scale partitions: Higher ef_construct = More partitions (finer granular search)
            # Example: ef=100 -> 256 partitions. ef=200 -> 512 partitions.
            ef = params["ef_construct"]
            num_partitions = max(256, int(ef * 2.5))

        # 2. Map m -> num_sub_vectors (Quantization Precision)
        # PQ sub-vectors must divide the dimension size.
        # We try to target m*4 as a rough heuristic for "good compression vs accuracy".
        target_m = params.get("m", 16)

        def get_valid_sub_vectors(dim, target_m):
             # Try to find a divisor close to target_m * 4 (e.g. 16*4 = 64 sub-vectors)
             # Higher num_sub_vectors = Higher accuracy, Larger index
             goal = target_m * 4

             # Check exact match first
             if dim % goal == 0: return goal

             # Find closest divisor
             best_divisor = 1
             min_diff = float('inf')

             for i in range(1, dim + 1):
                 if dim % i == 0:
                     diff = abs(i - goal)
                     if diff < min_diff:
                         min_diff = diff
                         best_divisor = i
             return best_divisor

        num_sub_vectors = get_valid_sub_vectors(self._dimensions, target_m)

        metric_map = {
            DistanceMetric.L2: "L2",
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.IP: "cosine" # LanceDB uses cosine for IP usually (normalized)
        }
        metric = metric_map.get(distance_metric, "L2")

        print(f"🔨 LanceDB: Building IVF-PQ (partitions={num_partitions}, sub_vectors={num_sub_vectors})...")

        self._table.create_index(
            metric=metric,
            vector_column_name="vector",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors
        )

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        if self._table_name and self._db:
            if self._table_name in self._db.table_names():
                self._db.drop_table(self._table_name)
        self._table = None
        self._table_name = ""

    def save_index(self, path: str) -> None:
        pass

    def load_index(self, path: str) -> float:
        return 0.0

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:

        if not self._table:
            raise RuntimeError("Table not created")

        self.validate_vectors(queries)

        params = search_params or {}

        # === FIX: SEARCH PARAM MAPPING ===
        # LanceDB uses 'nprobes'. Map 'ef' or 'nprobe' to it.
        nprobes = 20 # Default
        if "nprobe" in params:
            nprobes = params["nprobe"]
        elif "ef" in params:
            # Map ef (HNSW) to nprobes (IVF) roughly.
            # ef=100 -> nprobes=20 is a safe bet.
            nprobes = max(1, int(params["ef"] / 5))

        latencies = []
        all_indices = []
        all_distances = []

        for query in queries:
            start_q = time.perf_counter()

            results = self._table.search(query) \
                .metric(self._distance_metric.value if isinstance(self._distance_metric, DistanceMetric) else "L2") \
                .nprobes(nprobes) \
                .limit(k) \
                .to_pandas()

            latencies.append((time.perf_counter() - start_q) * 1000)

            # Convert string IDs back to Integers for metrics calculation
            try:
                indices = [int(i) for i in results["id"].values]
            except ValueError:
                indices = [0] * len(results)

            dists = results["_distance"].values.tolist()

            all_indices.append(indices)
            all_distances.append(dists)

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies
        )

    # Standard Bulk Interface
    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        # Ensure IDs are strings
        data = [{"id": str(id), "vector": vec} for id, vec in zip(ids, vectors)]
        start = time.perf_counter()
        self._table.add(data)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        self.delete(ids)
        return self.insert(vectors, metadata, ids)

    def delete(self, ids: List[int]) -> float:
        start = time.perf_counter()
        # IDs are strings now
        ids_str = ", ".join([f"'{i}'" for i in ids])
        self._table.delete(f"id IN ({ids_str})")
        self._num_vectors -= len(ids)
        return time.perf_counter() - start

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._table: return {}

        # Calculate size of the specific table directory
        import os
        total_size = 0
        table_path = os.path.join(self._db_path, f"{self._table_name}.lance")

        if os.path.exists(table_path):
            for dirpath, _, filenames in os.walk(table_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)

        return {
            "num_vectors": len(self._table),
            "dimensions": self._dimensions,
            "index_size_bytes": total_size
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params = params

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params

    # Single-Item Wrappers for Benchmarking
    def insert_one(self, id: str, vector: np.ndarray):
        data = [{"id": str(id), "vector": vector}]
        self._table.add(data)

    def delete_one(self, id: str):
        self._table.delete(f"id = '{str(id)}'")

    def update_one(self, id: str, vector: np.ndarray):
        self.delete_one(id)
        self.insert_one(id, vector)
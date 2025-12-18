"""
Base class for dataset loaders.

This module defines the abstract interface for loading benchmark datasets,
following the standard format used by ANN-Benchmarks.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    All dataset implementations should inherit from this class and implement
    the required methods for loading vectors, queries, and ground truth.
    """

    def __init__(self, config: Dict[str, Any], data_dir: str = "./data"):
        """
        Initialize the dataset loader.

        Args:
            config: Dataset-specific configuration
            data_dir: Base directory for dataset storage
        """
        self.config = config
        self.data_dir = Path(data_dir) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._vectors: Optional[NDArray[np.float32]] = None
        self._queries: Optional[NDArray[np.float32]] = None
        self._ground_truth: Optional[NDArray[np.int64]] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return dataset information."""
        pass

    @property
    def vectors(self) -> NDArray[np.float32]:
        """Get base vectors (lazy loading)."""
        if self._vectors is None:
            self._vectors = self.load_vectors()
        return self._vectors

    @property
    def queries(self) -> NDArray[np.float32]:
        """Get query vectors (lazy loading)."""
        if self._queries is None:
            self._queries = self.load_queries()
        return self._queries

    @property
    def ground_truth(self) -> NDArray[np.int64]:
        """Get ground truth neighbors (lazy loading)."""
        if self._ground_truth is None:
            self._ground_truth = self.load_ground_truth()
        return self._ground_truth

    @property
    def metadata(self) -> Optional[List[Dict[str, Any]]]:
        """Get metadata for vectors (lazy loading)."""
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def load_vectors(self) -> NDArray[np.float32]:
        """
        Load base vectors.

        Returns:
            Array of shape (n_vectors, dimensions)
        """
        pass

    @abstractmethod
    def load_queries(self) -> NDArray[np.float32]:
        """
        Load query vectors.

        Returns:
            Array of shape (n_queries, dimensions)
        """
        pass

    @abstractmethod
    def load_ground_truth(self) -> NDArray[np.int64]:
        """
        Load ground truth nearest neighbors.

        Returns:
            Array of shape (n_queries, k) with neighbor indices
        """
        pass

    def load_metadata(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load or generate metadata for vectors.

        Returns:
            List of metadata dictionaries, one per vector
        """
        return self._generate_synthetic_metadata()

    @abstractmethod
    def download(self) -> None:
        """
        Download the dataset files.

        Should download all necessary files to self.data_dir.
        """
        pass

    # =========================================================================
    # Common Methods
    # =========================================================================

    def is_downloaded(self) -> bool:
        """Check if dataset files are already downloaded."""
        return self.data_dir.exists() and any(self.data_dir.iterdir())

    def ensure_downloaded(self) -> None:
        """Ensure dataset is downloaded, downloading if necessary."""
        if not self.is_downloaded():
            print(f"Downloading {self.name} dataset...")
            self.download()
            print(f"Download complete: {self.data_dir}")

    def get_subset(
        self,
        size: int,
        seed: int = 42,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        """
        Get a subset of the dataset.

        Args:
            size: Number of vectors to include
            seed: Random seed for reproducibility

        Returns:
            Tuple of (vectors, queries, ground_truth) for subset
        """
        np.random.seed(seed)

        vectors = self.vectors
        queries = self.queries
        ground_truth = self.ground_truth

        if size >= len(vectors):
            return vectors, queries, ground_truth

        # Select random subset of vectors
        indices = np.random.choice(len(vectors), size, replace=False)
        indices = np.sort(indices)

        subset_vectors = vectors[indices]

        # Recompute ground truth for subset
        # Map original indices to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # Filter ground truth to only include vectors in subset
        new_ground_truth = []
        for gt_row in ground_truth:
            new_row = []
            for idx in gt_row:
                if idx in index_map:
                    new_row.append(index_map[idx])
            new_ground_truth.append(new_row)

        # Pad or truncate to fixed k
        k = ground_truth.shape[1]
        padded_gt = np.full((len(queries), k), -1, dtype=np.int64)
        for i, row in enumerate(new_ground_truth):
            length = min(len(row), k)
            padded_gt[i, :length] = row[:length]

        return subset_vectors, queries, padded_gt

    def compute_ground_truth(
        self,
        vectors: NDArray[np.float32],
        queries: NDArray[np.float32],
        k: int = 100,
        metric: DistanceMetric = DistanceMetric.L2,
    ) -> NDArray[np.int64]:
        """
        Compute exact ground truth nearest neighbors.

        Args:
            vectors: Base vectors
            queries: Query vectors
            k: Number of neighbors
            metric: Distance metric

        Returns:
            Array of shape (n_queries, k) with neighbor indices
        """
        from scipy.spatial.distance import cdist

        n_queries = len(queries)
        ground_truth = np.zeros((n_queries, k), dtype=np.int64)

        # Compute in batches to manage memory
        batch_size = 100
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            query_batch = queries[i:end_idx]

            if metric == DistanceMetric.L2:
                distances = cdist(query_batch, vectors, metric='euclidean')
            elif metric == DistanceMetric.COSINE:
                distances = cdist(query_batch, vectors, metric='cosine')
            elif metric == DistanceMetric.IP:
                # Inner product: higher is better, so negate
                distances = -np.dot(query_batch, vectors.T)
            else:
                distances = cdist(query_batch, vectors, metric='euclidean')

            # Get top-k indices
            for j, dist_row in enumerate(distances):
                top_k = np.argpartition(dist_row, k)[:k]
                top_k = top_k[np.argsort(dist_row[top_k])]
                ground_truth[i + j] = top_k

        return ground_truth

    def _generate_synthetic_metadata(self) -> List[Dict[str, Any]]:
        """Generate synthetic metadata for filtering benchmarks."""
        np.random.seed(42)
        n_vectors = len(self.vectors)

        metadata_config = self.config.get("metadata", {})
        if not metadata_config.get("generate", True):
            return None

        fields = metadata_config.get("fields", {})
        metadata = []

        for i in range(n_vectors):
            item = {}

            # Category field
            if "category" in fields:
                cat_config = fields["category"]
                values = cat_config.get("values", ["A", "B", "C", "D", "E"])
                item["category"] = np.random.choice(values)

            # Price field
            if "price" in fields:
                price_config = fields["price"]
                item["price"] = np.random.uniform(
                    price_config.get("min", 0),
                    price_config.get("max", 1000),
                )

            # Timestamp field
            if "timestamp" in fields:
                ts_config = fields["timestamp"]
                item["timestamp"] = np.random.randint(
                    ts_config.get("min", 1609459200),
                    ts_config.get("max", 1704067200),
                )

            # Active field
            if "active" in fields:
                active_config = fields["active"]
                item["active"] = np.random.random() < active_config.get("probability_true", 0.8)

            metadata.append(item)

        return metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# =============================================================================
# File Format Readers
# =============================================================================


def read_fvecs(filename: str) -> NDArray[np.float32]:
    """
    Read vectors from .fvecs file format.

    Format: Each vector is preceded by its dimension (int32).
    """
    with open(filename, 'rb') as f:
        # Read all data
        data = np.fromfile(f, dtype=np.float32)

    # First value is dimension
    d = int(data[0].view(np.int32))

    # Reshape: each row has (1 + d) values (dim prefix + vector)
    data = data.reshape(-1, d + 1)

    # Return only vector part (skip dimension prefix)
    return data[:, 1:].copy()


def read_ivecs(filename: str) -> NDArray[np.int32]:
    """
    Read integer vectors from .ivecs file format.

    Format: Same as fvecs but with int32 values.
    """
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)

    d = data[0]
    data = data.reshape(-1, d + 1)

    return data[:, 1:].copy()


def read_bvecs(filename: str) -> NDArray[np.uint8]:
    """
    Read byte vectors from .bvecs file format.
    """
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)

    d = int(np.frombuffer(data[:4], dtype=np.int32)[0])
    data = data.reshape(-1, d + 4)

    return data[:, 4:].copy()


def read_fbin(filename: str) -> NDArray[np.float32]:
    """
    Read vectors from .fbin (Yandex binary) format.

    Format: 8-byte header (n_vectors, dimensions as int32), then vectors.
    """
    with open(filename, 'rb') as f:
        n_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]
        vectors = np.fromfile(f, dtype=np.float32)

    return vectors.reshape(n_vectors, dimensions)


def read_ibin(filename: str) -> NDArray[np.int32]:
    """
    Read integer vectors from .ibin format.
    """
    with open(filename, 'rb') as f:
        n_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]
        vectors = np.fromfile(f, dtype=np.int32)

    return vectors.reshape(n_vectors, dimensions)

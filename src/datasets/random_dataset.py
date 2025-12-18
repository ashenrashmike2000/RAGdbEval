"""Random synthetic dataset for controlled experiments."""

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader
from src.datasets.factory import register_dataset


@register_dataset("random")
class RandomDatasetLoader(DatasetLoader):
    """Synthetic random vectors for controlled benchmarks."""

    @property
    def name(self) -> str:
        return "random"

    @property
    def info(self) -> DatasetInfo:
        specs = self.config.get("specs", {}).get("default", {})
        return DatasetInfo(
            name="random",
            display_name="Random",
            description="Synthetic random vectors",
            num_vectors=specs.get("num_vectors", 1000000),
            num_queries=specs.get("num_queries", 10000),
            dimensions=specs.get("dimensions", 128),
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
        )

    def download(self) -> None:
        """No download needed - vectors are generated."""
        pass

    def is_downloaded(self) -> bool:
        return True  # Always available

    def load_vectors(self) -> NDArray[np.float32]:
        gen_config = self.config.get("generation", {})
        seed = gen_config.get("seed", 42)
        np.random.seed(seed)

        specs = self.config.get("specs", {}).get("default", {})
        n = specs.get("num_vectors", 1000000)
        d = specs.get("dimensions", 128)

        dist_type = gen_config.get("distribution", {}).get("type", "gaussian")

        if dist_type == "gaussian":
            params = gen_config.get("distribution", {}).get("params", {}).get("gaussian", {})
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            vectors = np.random.normal(mean, std, (n, d)).astype(np.float32)
        elif dist_type == "uniform":
            params = gen_config.get("distribution", {}).get("params", {}).get("uniform", {})
            low = params.get("min", -1.0)
            high = params.get("max", 1.0)
            vectors = np.random.uniform(low, high, (n, d)).astype(np.float32)
        else:
            vectors = np.random.randn(n, d).astype(np.float32)

        if gen_config.get("normalize", False):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-8)

        return vectors

    def load_queries(self) -> NDArray[np.float32]:
        gen_config = self.config.get("generation", {})
        np.random.seed(gen_config.get("seed", 42) + 1)

        specs = self.config.get("specs", {}).get("default", {})
        n = specs.get("num_queries", 10000)
        d = specs.get("dimensions", 128)

        return np.random.randn(n, d).astype(np.float32)

    def load_ground_truth(self) -> NDArray[np.int64]:
        vectors = self.vectors
        queries = self.queries
        return self.compute_ground_truth(vectors, queries, k=100, metric=DistanceMetric.L2)

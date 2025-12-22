"""Random synthetic dataset for controlled experiments - Persisted Version."""

import struct
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader, read_fbin, read_ibin
from src.datasets.factory import register_dataset


@register_dataset("random")
class RandomDatasetLoader(DatasetLoader):
    """
    Synthetic random vectors that are generated ONCE and saved to disk.
    This mimics a downloaded dataset to save generation time on subsequent runs.
    """

    @property
    def name(self) -> str:
        return "random"

    @property
    def info(self) -> DatasetInfo:
        specs = self.config.get("specs", {}).get("default", {})
        return DatasetInfo(
            name="random",
            display_name="Random (Synthetic)",
            description="Synthetic random vectors (Saved to disk)",
            num_vectors=specs.get("num_vectors", 1_000_000),
            num_queries=specs.get("num_queries", 10_000),
            dimensions=specs.get("dimensions", 128),
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
        )

    def download(self) -> None:
        """
        Generates the random data and SAVES it to disk.
        This acts as the 'download' step.
        """
        # Define file paths
        base_path = self.data_dir / "base.fbin"
        query_path = self.data_dir / "query.fbin"
        gt_path = self.data_dir / "ground_truth.ibin"

        # Check if already exists
        if base_path.exists() and query_path.exists() and gt_path.exists():
            print("Random dataset already generated and saved.")
            return

        print("Generating random dataset... (This happens only once)")

        # 1. Generate Vectors
        base_vectors = self._generate_vectors(is_query=False)
        query_vectors = self._generate_vectors(is_query=True)

        # 2. Compute Ground Truth (Heavy computation)
        print("Computing Ground Truth... (This may take a moment)")
        # We use Brute Force to ensure exact ground truth
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='l2')
        nbrs.fit(base_vectors)
        ground_truth_indices = nbrs.kneighbors(query_vectors, return_distance=False)

        # 3. Save to Disk (Persist as .fbin / .ibin)
        print("Saving files to disk...")
        self._write_fbin(str(base_path), base_vectors)
        self._write_fbin(str(query_path), query_vectors)
        self._write_ibin(str(gt_path), ground_truth_indices.astype(np.int32))

        print(f"Done! Random dataset saved to {self.data_dir}")

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "base.fbin"))

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "query.fbin"))

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        # Load int32 from file, cast to int64 for the system
        return read_ibin(str(self.data_dir / "ground_truth.ibin")).astype(np.int64)

    # --- Helper Methods for Generation & Saving ---

    def _generate_vectors(self, is_query: bool = False) -> NDArray[np.float32]:
        """Handles the logic for creating random numpy arrays."""
        specs = self.config.get("specs", {}).get("default", {})
        gen_config = self.config.get("generation", {})

        # Determine size and seed
        if is_query:
            n = specs.get("num_queries", 10_000)
            seed = gen_config.get("seed", 42) + 1
        else:
            n = specs.get("num_vectors", 1_000_000)
            seed = gen_config.get("seed", 42)

        d = specs.get("dimensions", 128)
        np.random.seed(seed)

        dist_type = gen_config.get("distribution", {}).get("type", "gaussian")

        print(f"Generating {n} vectors ({dist_type})...")

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

    def _write_fbin(self, filename: str, data: NDArray[np.float32]) -> None:
        """Writes float32 data to .fbin format (BigANN standard)."""
        with open(filename, "wb") as f:
            n, d = data.shape
            # Header: num_vectors, dim
            f.write(struct.pack("ii", n, d))
            # Body: data
            f.write(data.tobytes())

    def _write_ibin(self, filename: str, data: NDArray[np.int32]) -> None:
        """Writes int32 data to .ibin format (BigANN standard)."""
        with open(filename, "wb") as f:
            n, d = data.shape
            # Header: num_vectors, dim
            f.write(struct.pack("ii", n, d))
            # Body: data
            f.write(data.tobytes())
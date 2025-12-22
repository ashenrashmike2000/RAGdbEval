"""DEEP1M Custom Loader - Slices 1M from Deep10M and computes local GT."""

import os
import struct
import numpy as np
import requests
from pathlib import Path
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader, read_fbin, read_ibin
from src.datasets.factory import register_dataset

@register_dataset("deep1m")
class DEEP1MLoader(DatasetLoader):
    """
    Custom DEEP1M loader.

    Since 'deep1m' does not exist officially, this loader:
    1. Downloads only the first 384MB of the deep10m dataset.
    2. Generates a valid deep1m file.
    3. Computes ground truth locally for this specific slice.
    """

    @property
    def name(self) -> str:
        return "deep1m"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="deep1m",
            display_name="DEEP-1M (Custom)",
            description="1M deep learning features sliced from DEEP10M",
            num_vectors=1_000_000,
            num_queries=10_000,
            dimensions=96,
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
            source_url="https://research.yandex.com/datasets/biganns",
        )

    def download(self) -> None:
        """Download partial data and generate missing ground truth."""
        base_url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"

        # Define paths
        base_file = self.data_dir / "base.1M.fbin"
        query_file = self.data_dir / "query.public.10K.fbin"
        gt_file = self.data_dir / "ground_truth.1M.ibin"

        # 1. Download Base Vectors (Smart Partial Download)
        if not base_file.exists():
            print("Downloading first 1M vectors from Deep10M (approx. 384MB)...")
            self._download_partial_fbin(
                url=f"{base_url}/base.10M.fbin",
                dest_path=base_file,
                num_vectors=1_000_000,
                dim=96
            )

        # 2. Download Query Vectors (Full Download)
        if not query_file.exists():
            print("Downloading query vectors...")
            self._download_file(f"{base_url}/query.public.10K.fbin", query_file)

        # 3. Compute Ground Truth (Since official GT is for 10M, we must compute for 1M)
        if not gt_file.exists():
            print("Computing Ground Truth for 1M slice (this may take a minute)...")
            self._compute_ground_truth(base_file, query_file, gt_file)

    def _download_partial_fbin(self, url: str, dest_path: Path, num_vectors: int, dim: int) -> None:
        """Downloads only the necessary bytes for N vectors using HTTP Range."""
        # Header is 8 bytes (2 integers). Data is num_vectors * dim * 4 bytes (float32)
        total_bytes = 8 + (num_vectors * dim * 4)

        headers = {"Range": f"bytes=0-{total_bytes - 1}"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            # We need to rewrite the header because the source says "10M" but we only have 1M
            # Read original header (8 bytes)
            original_header = response.raw.read(8)
            n_rows, n_dim = struct.unpack("ii", original_header)

            # Write NEW header with correct count (1,000,000)
            f.write(struct.pack("ii", num_vectors, n_dim))

            # Stream the rest of the requested data
            # We read total_bytes - 8 because we handled the header manually
            chunk_size = 8192
            bytes_written = 8

            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

    def _download_file(self, url: str, dest: Path) -> None:
        """Simple file downloader."""
        response = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _compute_ground_truth(self, base_path: Path, query_path: Path, gt_path: Path) -> None:
        """Calculates exact nearest neighbors for the new 1M slice."""
        print("Loading data for GT generation...")
        base = read_fbin(str(base_path))
        queries = read_fbin(str(query_path))

        print(f"Running Brute Force search on {len(base)} vectors...")
        # Use Brute Force to find top 100 neighbors
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='l2')
        nbrs.fit(base)
        dist, ind = nbrs.kneighbors(queries)

        # Save as IBIN format
        print(f"Saving generated Ground Truth to {gt_path.name}...")
        self._write_ibin(str(gt_path), ind.astype(np.int32))

    def _write_ibin(self, filename: str, data: NDArray[np.int32]) -> None:
        """Writes integers to IBIN format."""
        with open(filename, "wb") as f:
            n_samples, k = data.shape
            f.write(struct.pack("ii", n_samples, k))
            f.write(data.tobytes())

    # --- Standard Loaders ---
    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "base.1M.fbin"))

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "query.public.10K.fbin"))

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        # Note: The computed GT is int32, but standard interface expects int64
        return read_ibin(str(self.data_dir / "ground_truth.1M.ibin")).astype(np.int64)
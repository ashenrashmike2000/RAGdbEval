"""
SIFT1M dataset loader.

SIFT1M contains 1 million 128-dimensional SIFT descriptors from image patches.
This is one of the most widely used ANN benchmark datasets.

Source: http://corpus-texmex.irisa.fr/
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader, read_fvecs, read_ivecs
from src.datasets.downloader import download_ftp, extract_archive
from src.datasets.factory import register_dataset


@register_dataset("sift1m")
class SIFT1MLoader(DatasetLoader):
    """SIFT1M dataset loader."""

    @property
    def name(self) -> str:
        return "sift1m"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="sift1m",
            display_name="SIFT-1M",
            description="1M SIFT descriptors from image patches",
            num_vectors=1000000,
            num_queries=10000,
            dimensions=128,
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
            source_url="http://corpus-texmex.irisa.fr/",
            size_mb=500,
        )

    def download(self) -> None:
        """Download SIFT1M dataset."""
        archive_url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        archive_path = self.data_dir / "sift.tar.gz"

        # Download archive
        download_ftp(archive_url, str(archive_path))

        # Extract
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        """Load base vectors."""
        self.ensure_downloaded()
        filepath = self.data_dir / "sift" / "sift_base.fvecs"
        return read_fvecs(str(filepath))

    def load_queries(self) -> NDArray[np.float32]:
        """Load query vectors."""
        self.ensure_downloaded()
        filepath = self.data_dir / "sift" / "sift_query.fvecs"
        return read_fvecs(str(filepath))

    def load_ground_truth(self) -> NDArray[np.int64]:
        """Load ground truth."""
        self.ensure_downloaded()
        filepath = self.data_dir / "sift" / "sift_groundtruth.ivecs"
        return read_ivecs(str(filepath)).astype(np.int64)

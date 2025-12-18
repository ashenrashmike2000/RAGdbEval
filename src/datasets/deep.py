"""DEEP1M dataset loader - Deep learning features from GoogLeNet."""

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader, read_fbin, read_ibin
from src.datasets.downloader import download_file
from src.datasets.factory import register_dataset


@register_dataset("deep1m")
class DEEP1MLoader(DatasetLoader):
    """DEEP1M dataset loader."""

    @property
    def name(self) -> str:
        return "deep1m"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="deep1m",
            display_name="DEEP-1M",
            description="1M deep learning features from GoogLeNet (96-dim)",
            num_vectors=1000000,
            num_queries=10000,
            dimensions=96,
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
            source_url="https://research.yandex.com/datasets/biganns",
        )

    def download(self) -> None:
        """Download DEEP1M dataset."""
        base_url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"

        files = {
            "base.1M.fbin": "base vectors",
            "query.public.10K.fbin": "query vectors",
            "ground_truth.public.10K.ibin": "ground truth",
        }

        for filename, desc in files.items():
            url = f"{base_url}/{filename}"
            dest = self.data_dir / filename
            if not dest.exists():
                print(f"Downloading {desc}...")
                download_file(url, str(dest))

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "base.1M.fbin"))

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fbin(str(self.data_dir / "query.public.10K.fbin"))

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        return read_ibin(str(self.data_dir / "ground_truth.public.10K.ibin")).astype(np.int64)

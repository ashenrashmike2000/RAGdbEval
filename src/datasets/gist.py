"""GIST1M dataset loader - Global image descriptors (960-dim)."""

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader, read_fvecs, read_ivecs
from src.datasets.downloader import download_ftp, extract_archive
from src.datasets.factory import register_dataset


@register_dataset("gist1m")
class GIST1MLoader(DatasetLoader):
    """GIST1M dataset loader."""

    @property
    def name(self) -> str:
        return "gist1m"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="gist1m",
            display_name="GIST-1M",
            description="1M GIST global image descriptors (960-dim)",
            num_vectors=1000000,
            num_queries=1000,
            dimensions=960,
            data_type="float32",
            distance_metric=DistanceMetric.L2,
            ground_truth_k=100,
            source_url="http://corpus-texmex.irisa.fr/",
        )

    def download(self) -> None:
        archive_url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
        archive_path = self.data_dir / "gist.tar.gz"
        download_ftp(archive_url, str(archive_path))
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fvecs(str(self.data_dir / "gist" / "gist_base.fvecs"))

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        return read_fvecs(str(self.data_dir / "gist" / "gist_query.fvecs"))

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        return read_ivecs(str(self.data_dir / "gist" / "gist_groundtruth.ivecs")).astype(np.int64)

"""GloVe dataset loader - Word embeddings."""

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader
from src.datasets.downloader import download_file, extract_archive
from src.datasets.factory import register_dataset


@register_dataset("glove")
class GloVeLoader(DatasetLoader):
    """GloVe word embeddings loader."""

    @property
    def name(self) -> str:
        return "glove"

    @property
    def info(self) -> DatasetInfo:
        dims = self.config.get("specs", {}).get("default", {}).get("dimensions", 100)
        return DatasetInfo(
            name="glove",
            display_name="GloVe",
            description=f"GloVe word embeddings ({dims}-dim)",
            num_vectors=400000,
            num_queries=10000,
            dimensions=dims,
            data_type="float32",
            distance_metric=DistanceMetric.COSINE,
            ground_truth_k=100,
            source_url="https://nlp.stanford.edu/projects/glove/",
        )

    def download(self) -> None:
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        archive_path = self.data_dir / "glove.6B.zip"
        download_file(url, str(archive_path))
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        dims = self.config.get("specs", {}).get("default", {}).get("dimensions", 100)
        filepath = self.data_dir / f"glove.6B.{dims}d.txt"

        vectors = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                vector = [float(x) for x in parts[1:]]
                vectors.append(vector)

        return np.array(vectors, dtype=np.float32)

    def load_queries(self) -> NDArray[np.float32]:
        vectors = self.vectors
        np.random.seed(42)
        indices = np.random.choice(len(vectors), 10000, replace=False)
        return vectors[indices].copy()

    def load_ground_truth(self) -> NDArray[np.int64]:
        vectors = self.vectors
        queries = self.queries
        return self.compute_ground_truth(vectors, queries, k=100, metric=DistanceMetric.COSINE)

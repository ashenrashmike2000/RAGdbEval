"""MS MARCO dataset loader - Text passage embeddings."""

import numpy as np
from numpy.typing import NDArray

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader
from src.datasets.downloader import download_file, extract_archive
from src.datasets.factory import register_dataset


@register_dataset("msmarco")
class MSMARCOLoader(DatasetLoader):
    """MS MARCO passage embeddings loader."""

    @property
    def name(self) -> str:
        return "msmarco"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="msmarco",
            display_name="MS MARCO",
            description="MS MARCO passage embeddings (768-dim)",
            num_vectors=8841823,
            num_queries=6980,
            dimensions=768,
            data_type="float32",
            distance_metric=DistanceMetric.COSINE,
            ground_truth_k=100,
            source_url="https://microsoft.github.io/msmarco/",
        )

    def download(self) -> None:
        # Download from BEIR repository (pre-computed embeddings)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        archive_path = self.data_dir / "msmarco.zip"

        print("Downloading MS MARCO dataset (this may take a while)...")
        download_file(url, str(archive_path))
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        # Check for pre-computed embeddings or generate them
        embeddings_path = self.data_dir / "corpus_embeddings.npy"

        if embeddings_path.exists():
            return np.load(str(embeddings_path))

        # Generate embeddings using sentence-transformers (if available)
        try:
            from sentence_transformers import SentenceTransformer
            import json

            print("Generating embeddings (this will take a while)...")
            model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')

            # Load corpus
            corpus_path = self.data_dir / "msmarco" / "corpus.jsonl"
            texts = []
            with open(corpus_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item.get('text', ''))

            # Generate in batches
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
            np.save(str(embeddings_path), embeddings)
            return embeddings.astype(np.float32)

        except ImportError:
            raise RuntimeError(
                "sentence-transformers required for MS MARCO embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        queries_path = self.data_dir / "query_embeddings.npy"

        if queries_path.exists():
            return np.load(str(queries_path))

        try:
            from sentence_transformers import SentenceTransformer
            import json

            model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')

            queries_file = self.data_dir / "msmarco" / "queries.jsonl"
            texts = []
            with open(queries_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item.get('text', ''))

            embeddings = model.encode(texts, show_progress_bar=True)
            np.save(str(queries_path), embeddings)
            return embeddings.astype(np.float32)

        except ImportError:
            raise RuntimeError("sentence-transformers required")

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        gt_path = self.data_dir / "ground_truth.npy"

        if gt_path.exists():
            return np.load(str(gt_path))

        # Compute ground truth
        vectors = self.vectors
        queries = self.queries
        gt = self.compute_ground_truth(vectors, queries, k=100, metric=DistanceMetric.COSINE)
        np.save(str(gt_path), gt)
        return gt

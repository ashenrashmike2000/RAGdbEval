"""MS MARCO dataset loader - Text passage embeddings with subset support."""

import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from src.core.types import DatasetInfo, DistanceMetric
from src.datasets.base import DatasetLoader
from src.datasets.downloader import download_file, extract_archive
from src.datasets.factory import register_dataset


@register_dataset("msmarco")
class MSMARCOLoader(DatasetLoader):
    """MS MARCO passage embeddings loader with configurable subset support."""

    @property
    def name(self) -> str: return "msmarco"

    @property
    def info(self) -> DatasetInfo:
        subset_size = self.config.get("subset_size", 100000)
        num_queries = self.config.get("specs", {}).get("num_queries", 1000)
        return DatasetInfo(
            name="msmarco",
            display_name="MS MARCO",
            description=f"MS MARCO passage embeddings (768-dim, {subset_size} vectors)",
            num_vectors=subset_size,
            num_queries=num_queries,
            dimensions=768,
            data_type="float32",
            distance_metric=DistanceMetric.COSINE,
            ground_truth_k=100,
            source_url="https://microsoft.github.io/msmarco/",
        )

    def download(self) -> None:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        archive_path = self.data_dir / "msmarco.zip"
        if (self.data_dir / "msmarco").exists(): return
        print("Downloading MS MARCO dataset...")
        download_file(url, str(archive_path))
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        subset_size = self.config.get("subset_size", 100000)
        embeddings_path = self.data_dir / f"corpus_embeddings_{subset_size}.npy"

        if embeddings_path.exists():
            return np.load(str(embeddings_path))

        print(f"Generating embeddings for {subset_size} passages...")
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("embedding", {}).get("model", 'sentence-transformers/msmarco-distilbert-base-v4')
            model = SentenceTransformer(model_name)

            corpus_path = self.data_dir / "msmarco" / "corpus.jsonl"
            texts = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= subset_size: break
                    item = json.loads(line)
                    texts.append(item.get('title', '') + ' ' + item.get('text', ''))

            embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            np.save(str(embeddings_path), embeddings)
            return embeddings.astype(np.float32)
        except ImportError:
            raise RuntimeError("sentence-transformers required. pip install sentence-transformers")

    def load_queries(self) -> NDArray[np.float32]:
        self.ensure_downloaded()
        num_queries = self.config.get("specs", {}).get("num_queries", 1000)
        queries_path = self.data_dir / f"query_embeddings_{num_queries}.npy"

        if queries_path.exists():
            return np.load(str(queries_path))

        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("embedding", {}).get("model", 'sentence-transformers/msmarco-distilbert-base-v4')
            model = SentenceTransformer(model_name)

            queries_file = self.data_dir / "msmarco" / "queries.jsonl"

            if not queries_file.exists():
                print("⚠️ Query file not found. Using DISJOINT corpus subset as queries...")
                # FIX: Use slicing from end, NOT random sampling to avoid leakage
                vectors = self.vectors
                # Ensure we don't crash if vectors are too small
                if len(vectors) <= num_queries:
                    raise ValueError("Vector set too small to split for queries.")
                return vectors[-num_queries:].copy()

            texts = []
            with open(queries_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_queries: break
                    item = json.loads(line)
                    texts.append(item.get('text', ''))

            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            np.save(str(queries_path), embeddings)
            return embeddings.astype(np.float32)
        except ImportError:
            raise RuntimeError("sentence-transformers required")

    def load_ground_truth(self) -> NDArray[np.int64]:
        self.ensure_downloaded()
        subset_size = self.config.get("subset_size", 100000)
        gt_path = self.data_dir / f"ground_truth_subset_{subset_size}.npy"

        if gt_path.exists():
            return np.load(str(gt_path))

        print("Computing ground truth (one-time operation)...")
        vectors = self.vectors
        queries = self.queries

        # Use FAISS for exact search if available
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            _, gt = index.search(queries, k=100)
            np.save(str(gt_path), gt)
            return gt.astype(np.int64)
        except ImportError:
            gt = self.compute_ground_truth(vectors, queries, k=100, metric=DistanceMetric.COSINE)
            np.save(str(gt_path), gt)
            return gt
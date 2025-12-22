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
    """MS MARCO passage embeddings loader with configurable subset size."""

    @property
    def name(self) -> str:
        return "msmarco"

    @property
    def info(self) -> DatasetInfo:
        # Get subset size from config
        subset_size = self.config.get("subset_size", 100000)

        # Get query count from config specs
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
        """Download MS MARCO dataset."""
        # Download from BEIR repository (pre-computed embeddings)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        archive_path = self.data_dir / "msmarco.zip"

        # Check if extracted folder already exists to avoid re-downloading
        if (self.data_dir / "msmarco").exists():
            return

        print("Downloading MS MARCO dataset (this may take a while)...")
        download_file(url, str(archive_path))
        extract_archive(str(archive_path), str(self.data_dir), remove_archive=True)

    def load_vectors(self) -> NDArray[np.float32]:
        """Load base vectors with subset support."""
        self.ensure_downloaded()

        # Get subset size from config
        subset_size = self.config.get("subset_size", 100000)

        # FIXED: Include subset size in filename to prevent caching errors
        embeddings_path = self.data_dir / f"corpus_embeddings_{subset_size}.npy"

        if embeddings_path.exists():
            print(f"Loading pre-computed embeddings (subset: {subset_size})...")
            embeddings = np.load(str(embeddings_path))
            # Verify length just in case
            if len(embeddings) == subset_size:
                return embeddings
            print("Cached embeddings size mismatch. Regenerating...")

        # Generate embeddings using sentence-transformers
        print(f"Generating embeddings for {subset_size} passages (this will take a while)...")

        try:
            from sentence_transformers import SentenceTransformer

            # Use model from config or default
            model_name = self.config.get("embedding", {}).get("model", 'sentence-transformers/msmarco-distilbert-base-v4')
            model = SentenceTransformer(model_name)

            # Load corpus
            corpus_path = self.data_dir / "msmarco" / "corpus.jsonl"

            if not corpus_path.exists():
                raise FileNotFoundError(
                    f"Corpus file not found: {corpus_path}\n"
                    "Please download the MS MARCO dataset first."
                )

            texts = []
            print(f"Loading {subset_size} passages from corpus...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= subset_size:
                        break
                    item = json.loads(line)
                    # Combine title and text for better embeddings
                    text = item.get('title', '') + ' ' + item.get('text', '')
                    texts.append(text.strip())

            print(f"Loaded {len(texts)} passages. Generating embeddings...")

            # Generate embeddings in batches
            batch_size = self.config.get("embedding", {}).get("batch_size", 32)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )

            # Save for future use
            print(f"Saving embeddings to {embeddings_path.name}...")
            np.save(str(embeddings_path), embeddings)

            return embeddings.astype(np.float32)

        except ImportError:
            raise RuntimeError(
                "sentence-transformers required for MS MARCO embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}")

    def load_queries(self) -> NDArray[np.float32]:
        """Load query vectors with subset support."""
        self.ensure_downloaded()

        # Get number of queries from config
        num_queries = self.config.get("specs", {}).get("num_queries", 1000)

        # FIXED: Include query count in filename
        queries_path = self.data_dir / f"query_embeddings_{num_queries}.npy"

        if queries_path.exists():
            queries = np.load(str(queries_path))
            if len(queries) == num_queries:
                return queries

        try:
            from sentence_transformers import SentenceTransformer

            print(f"Generating query embeddings ({num_queries} queries)...")
            model_name = self.config.get("embedding", {}).get("model", 'sentence-transformers/msmarco-distilbert-base-v4')
            model = SentenceTransformer(model_name)

            queries_file = self.data_dir / "msmarco" / "queries.jsonl"

            if not queries_file.exists():
                print("Query file not found. Using corpus subset as queries...")
                vectors = self.vectors
                np.random.seed(42)
                indices = np.random.choice(len(vectors), num_queries, replace=False)
                return vectors[indices].copy()

            texts = []
            with open(queries_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_queries:
                        break
                    item = json.loads(line)
                    texts.append(item.get('text', ''))

            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            np.save(str(queries_path), embeddings)
            return embeddings.astype(np.float32)

        except ImportError:
            raise RuntimeError("sentence-transformers required")
        except Exception as e:
            print(f"Warning: Could not generate query embeddings ({e}). Using corpus subset.")
            vectors = self.vectors
            np.random.seed(42)
            indices = np.random.choice(len(vectors), num_queries, replace=False)
            return vectors[indices].copy()

    def load_ground_truth(self) -> NDArray[np.int64]:
        """Load or compute ground truth."""
        self.ensure_downloaded()

        subset_size = self.config.get("subset_size", 100000)

        # FIXED: Include subset size in GT filename
        # Ground truth depends on specific subset of vectors!
        gt_path = self.data_dir / f"ground_truth_subset_{subset_size}.npy"

        if gt_path.exists():
            return np.load(str(gt_path))

        # Compute ground truth (this takes time but is one-time)
        print("Computing ground truth (one-time operation)...")
        vectors = self.vectors
        queries = self.queries

        # Use FAISS for fast ground truth computation
        try:
            import faiss

            # Build flat index for exact search
            # Note: Vectors are already normalized, so Inner Product == Cosine Similarity
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)

            # Search for ground truth
            k = self.config.get("specs", {}).get("ground_truth_k", 100)
            _, gt = index.search(queries, k=k)

            np.save(str(gt_path), gt)
            return gt.astype(np.int64)

        except ImportError:
            print("FAISS not available. Using slower ground truth computation...")
            k = self.config.get("specs", {}).get("ground_truth_k", 100)
            gt = self.compute_ground_truth(vectors, queries, k=k, metric=DistanceMetric.COSINE)
            np.save(str(gt_path), gt)
            return gt
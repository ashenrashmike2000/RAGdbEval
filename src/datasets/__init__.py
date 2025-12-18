"""
Dataset loaders for benchmark datasets.

Supported datasets:
    - SIFT1M: 1M SIFT descriptors (128-dim)
    - DEEP1M: 1M deep learning features (96-dim)
    - MS MARCO: 8.8M text embeddings (768-dim)
    - GloVe: Word embeddings (50-300 dim)
    - GIST1M: 1M GIST global features (960-dim)
    - Random: Synthetic random vectors
"""

from src.datasets.base import DatasetLoader
from src.datasets.factory import get_dataset, list_available_datasets

__all__ = [
    "DatasetLoader",
    "get_dataset",
    "list_available_datasets",
]

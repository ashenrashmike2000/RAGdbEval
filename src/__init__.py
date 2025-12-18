"""
VectorDB-Bench: Comprehensive Vector Database Benchmarking Framework

A research-grade benchmarking framework for evaluating vector database performance
across multiple dimensions: quality, performance, resource usage, and operational metrics.

Supported Databases:
    - FAISS (Meta)
    - Qdrant
    - Milvus
    - LanceDB
    - Weaviate
    - Chroma
    - pgvector

Supported Datasets:
    - SIFT1M
    - DEEP1M
    - MS MARCO
    - GloVe
    - GIST1M
    - Random (synthetic)

Reference Benchmarks:
    - ANN-Benchmarks (https://ann-benchmarks.com)
    - VectorDBBench (https://github.com/zilliztech/VectorDBBench)
    - Big-ANN-Benchmarks (https://big-ann-benchmarks.com)
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

from src.core.config import Config, load_config
from src.core.types import (
    BenchmarkResult,
    DatasetInfo,
    DatabaseInfo,
    MetricsResult,
    SearchResult,
)

__all__ = [
    "Config",
    "load_config",
    "BenchmarkResult",
    "DatasetInfo",
    "DatabaseInfo",
    "MetricsResult",
    "SearchResult",
]

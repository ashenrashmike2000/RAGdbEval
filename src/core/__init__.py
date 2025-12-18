"""Core module containing base classes, types, and configuration utilities."""

from src.core.base import VectorDBInterface
from src.core.config import Config, load_config
from src.core.types import (
    BenchmarkResult,
    DatabaseInfo,
    DatasetInfo,
    DistanceMetric,
    IndexConfig,
    MetricsResult,
    OperationalMetrics,
    PerformanceMetrics,
    QualityMetrics,
    ResourceMetrics,
    ScalabilityMetrics,
    SearchResult,
)

__all__ = [
    "VectorDBInterface",
    "Config",
    "load_config",
    "BenchmarkResult",
    "DatabaseInfo",
    "DatasetInfo",
    "DistanceMetric",
    "IndexConfig",
    "MetricsResult",
    "OperationalMetrics",
    "PerformanceMetrics",
    "QualityMetrics",
    "ResourceMetrics",
    "ScalabilityMetrics",
    "SearchResult",
]

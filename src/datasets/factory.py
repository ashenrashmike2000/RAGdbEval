"""
Factory for creating dataset loaders.
"""

from typing import Any, Dict, List, Optional, Type

from src.datasets.base import DatasetLoader
from src.core.config import load_dataset_config

_DATASET_REGISTRY: Dict[str, Type[DatasetLoader]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset loader."""
    def decorator(cls: Type[DatasetLoader]) -> Type[DatasetLoader]:
        _DATASET_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_dataset(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    data_dir: str = "./data",
) -> DatasetLoader:
    """Get a dataset loader instance."""
    name_lower = name.lower()

    if name_lower not in _DATASET_REGISTRY:
        _import_dataset(name_lower)

    if name_lower not in _DATASET_REGISTRY:
        available = ", ".join(list_available_datasets())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    if config is None:
        try:
            config = load_dataset_config(name_lower)
        except FileNotFoundError:
            config = {}

    loader_class = _DATASET_REGISTRY[name_lower]
    return loader_class(config, data_dir)


def list_available_datasets() -> List[str]:
    """List all registered datasets."""
    _import_all_datasets()
    return sorted(_DATASET_REGISTRY.keys())


def _import_dataset(name: str) -> None:
    """Import a dataset module by name."""
    modules = {
        "sift1m": "src.datasets.sift",
        "deep1m": "src.datasets.deep",
        "msmarco": "src.datasets.msmarco",
        "glove": "src.datasets.glove",
        "gist1m": "src.datasets.gist",
        "random": "src.datasets.random_dataset",
    }
    module_name = modules.get(name.lower())
    if module_name:
        try:
            __import__(module_name)
        except ImportError:
            pass


def _import_all_datasets() -> None:
    """Import all dataset modules."""
    for name in ["sift1m", "deep1m", "msmarco", "glove", "gist1m", "random"]:
        _import_dataset(name)

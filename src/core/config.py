"""
Configuration management for the Vector Database Benchmarking Framework.

This module handles loading, validating, and accessing configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Configuration Models
# =============================================================================


class ExperimentConfig(BaseModel):
    """Configuration for experiment execution."""

    runs: int = Field(default=5, ge=1, description="Number of runs per configuration")
    warmup_queries: int = Field(default=1000, ge=0, description="Warmup queries")
    measurement_queries: int = Field(default=10000, ge=1, description="Queries for measurement")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    confidence_level: float = Field(default=0.95, ge=0, le=1, description="Confidence level")


class QueryConfig(BaseModel):
    """Configuration for query execution."""

    k_values: List[int] = Field(default=[1, 10, 50, 100], description="Top-K values")
    num_queries: int = Field(default=10000, ge=1, description="Number of queries")
    thread_counts: List[int] = Field(default=[1, 4, 8, 16, 32], description="Thread counts")
    batch_sizes: List[int] = Field(default=[1, 10, 100], description="Batch sizes")


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""

    quality: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    resource: Dict[str, Any] = Field(default_factory=dict)
    operational: Dict[str, Any] = Field(default_factory=dict)
    scalability: Dict[str, Any] = Field(default_factory=dict)


class HardwareConfig(BaseModel):
    """Configuration for hardware settings."""

    auto_detect: bool = Field(default=True, description="Auto-detect hardware")
    cpu_limit: Optional[int] = Field(default=None, description="CPU core limit")
    memory_limit_gb: int = Field(default=16, ge=1, description="Memory limit in GB")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration")
    gpu_device_id: int = Field(default=0, ge=0, description="GPU device ID")


class OutputConfig(BaseModel):
    """Configuration for output."""

    results_dir: str = Field(default="./results", description="Results directory")
    formats: Dict[str, bool] = Field(
        default={"json": True, "csv": True, "latex": True},
        description="Output formats",
    )
    plots: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    """Configuration for database selection."""

    active: str = Field(default="faiss", description="Active database")
    compare_all: bool = Field(default=False, description="Compare all databases")
    available: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    """Configuration for dataset selection."""

    active: str = Field(default="sift1m", description="Active dataset")
    compare_all: bool = Field(default=False, description="Compare all datasets")
    data_dir: str = Field(default="./data", description="Data directory")
    available: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration model."""

    benchmark: Dict[str, str] = Field(default_factory=dict)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    distance_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    workloads: Dict[str, Any] = Field(default_factory=dict)
    filtering: Dict[str, Any] = Field(default_factory=dict)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tradeoff: Dict[str, Any] = Field(default_factory=dict)

    # Paths
    config_dir: Path = Field(default=Path("./config"), exclude=True)

    @field_validator("config_dir", mode="before")
    @classmethod
    def validate_config_dir(cls, v):
        return Path(v) if isinstance(v, str) else v

    def get_database_config(self, db_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific database.

        Args:
            db_name: Database name (defaults to active database)

        Returns:
            Database configuration dictionary
        """
        name = db_name or self.database.active
        db_info = self.database.available.get(name, {})

        if not db_info:
            raise ValueError(f"Unknown database: {name}")

        # Load database-specific config file
        config_file = db_info.get("config_file")
        if config_file:
            db_config_path = self.config_dir / config_file
            if db_config_path.exists():
                with open(db_config_path) as f:
                    return yaml.safe_load(f)

        return db_info

    def get_dataset_config(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name: Dataset name (defaults to active dataset)

        Returns:
            Dataset configuration dictionary
        """
        name = dataset_name or self.dataset.active
        ds_info = self.dataset.available.get(name, {})

        if not ds_info:
            raise ValueError(f"Unknown dataset: {name}")

        # Load dataset-specific config file
        config_file = ds_info.get("config_file")
        if config_file:
            ds_config_path = self.config_dir / config_file
            if ds_config_path.exists():
                with open(ds_config_path) as f:
                    return yaml.safe_load(f)

        return ds_info

    def get_enabled_databases(self) -> List[str]:
        """Get list of enabled databases."""
        return [
            name
            for name, info in self.database.available.items()
            if info.get("enabled", False)
        ]

    def get_enabled_datasets(self) -> List[str]:
        """Get list of enabled datasets."""
        return [
            name
            for name, info in self.dataset.available.items()
            if info.get("enabled", False)
        ]


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to main configuration file (default: config/default.yaml)
        config_dir: Directory containing configuration files

    Returns:
        Config object with loaded configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    # Determine config directory
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
    else:
        config_dir = Path(config_dir)

    # Determine config file path
    if config_path is None:
        config_path = config_dir / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load main configuration
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Add config directory to the config
    config_dict["config_dir"] = config_dir

    # Create and validate config
    return Config(**config_dict)


def load_database_config(
    db_name: str,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load database-specific configuration.

    Args:
        db_name: Name of the database
        config_dir: Directory containing configuration files

    Returns:
        Database configuration dictionary
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / "databases" / f"{db_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Database configuration not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_config(
    dataset_name: str,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load dataset-specific configuration.

    Args:
        dataset_name: Name of the dataset
        config_dir: Directory containing configuration files

    Returns:
        Dataset configuration dictionary
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / "datasets" / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Config object with default settings
    """
    return Config()


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# Hardware Detection
# =============================================================================


def detect_hardware() -> Dict[str, Any]:
    """
    Detect hardware configuration.

    Returns:
        Dictionary with hardware information
    """
    import platform

    try:
        import cpuinfo
        import psutil

        cpu_info = cpuinfo.get_cpu_info()
        memory = psutil.virtual_memory()

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": {
                "brand": cpu_info.get("brand_raw", "Unknown"),
                "arch": cpu_info.get("arch", "Unknown"),
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "frequency_mhz": cpu_info.get("hz_actual_friendly", "Unknown"),
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
            },
            "gpu": _detect_gpu(),
        }
    except ImportError:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": {"cores_logical": os.cpu_count()},
            "memory": {},
            "gpu": {},
        }


def _detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU configuration.

    Returns:
        Dictionary with GPU information
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(", ")
                    gpus.append({"name": parts[0], "memory": parts[1] if len(parts) > 1 else "Unknown"})
            return {"nvidia": gpus}
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return {"available": False}

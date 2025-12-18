"""
Resource metrics for memory, disk, and CPU usage evaluation.
"""

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil

from src.core.types import ResourceMetrics


def measure_memory_usage() -> Dict[str, int]:
    """
    Measure current memory usage.

    Returns:
        Dictionary with memory metrics in bytes
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "rss": mem_info.rss,  # Resident Set Size
        "vms": mem_info.vms,  # Virtual Memory Size
        "shared": getattr(mem_info, 'shared', 0),
        "peak": getattr(mem_info, 'peak_wset', mem_info.rss),  # Peak memory (Windows) or RSS
    }


def measure_disk_usage(path: str) -> Dict[str, int]:
    """
    Measure disk usage for a path.

    Args:
        path: Directory or file path

    Returns:
        Dictionary with disk metrics in bytes
    """
    path = Path(path)

    if not path.exists():
        return {"size": 0, "files": 0}

    if path.is_file():
        return {"size": path.stat().st_size, "files": 1}

    total_size = 0
    file_count = 0

    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    return {"size": total_size, "files": file_count}


def measure_cpu_usage(duration_sec: float = 1.0) -> Dict[str, float]:
    """
    Measure CPU usage over a duration.

    Args:
        duration_sec: Duration to measure

    Returns:
        Dictionary with CPU metrics
    """
    process = psutil.Process(os.getpid())

    cpu_times_start = process.cpu_times()
    time.sleep(duration_sec)
    cpu_times_end = process.cpu_times()

    user_time = cpu_times_end.user - cpu_times_start.user
    system_time = cpu_times_end.system - cpu_times_start.system

    # CPU percent over the interval
    cpu_percent = process.cpu_percent(interval=None)

    return {
        "user_time_sec": user_time,
        "system_time_sec": system_time,
        "total_time_sec": user_time + system_time,
        "percent": cpu_percent,
        "num_threads": process.num_threads(),
    }


class ResourceMonitor:
    """
    Context manager for monitoring resource usage during operations.

    Example:
        with ResourceMonitor() as monitor:
            # Do some work
            pass
        print(monitor.peak_memory_bytes)
    """

    def __init__(self, sample_interval_sec: float = 0.1):
        """
        Initialize resource monitor.

        Args:
            sample_interval_sec: Sampling interval for peak detection
        """
        self.sample_interval = sample_interval_sec
        self.process = psutil.Process(os.getpid())

        self._start_memory = 0
        self._peak_memory = 0
        self._start_cpu_times = None
        self._start_time = 0
        self._end_time = 0
        self._monitoring = False

    def __enter__(self):
        self._start_memory = self.process.memory_info().rss
        self._peak_memory = self._start_memory
        self._start_cpu_times = self.process.cpu_times()
        self._start_time = time.perf_counter()
        self._monitoring = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.perf_counter()
        self._monitoring = False
        # Final memory sample
        current = self.process.memory_info().rss
        self._peak_memory = max(self._peak_memory, current)
        return False

    @property
    def elapsed_sec(self) -> float:
        """Elapsed time in seconds."""
        if self._monitoring:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time

    @property
    def memory_delta_bytes(self) -> int:
        """Memory change from start."""
        current = self.process.memory_info().rss
        return current - self._start_memory

    @property
    def peak_memory_bytes(self) -> int:
        """Peak memory usage."""
        return self._peak_memory

    @property
    def cpu_time_sec(self) -> float:
        """Total CPU time used."""
        current = self.process.cpu_times()
        user = current.user - self._start_cpu_times.user
        system = current.system - self._start_cpu_times.system
        return user + system

    def sample(self) -> None:
        """Take a memory sample (call periodically for accurate peak)."""
        if self._monitoring:
            current = self.process.memory_info().rss
            self._peak_memory = max(self._peak_memory, current)


def measure_index_build(
    build_fn: Callable,
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure resource usage during index building.

    Args:
        build_fn: Function that builds the index
        index_path: Path where index is stored (for disk measurement)

    Returns:
        Dictionary with build metrics
    """
    with ResourceMonitor() as monitor:
        build_fn()

    result = {
        "build_time_sec": monitor.elapsed_sec,
        "memory_delta_bytes": monitor.memory_delta_bytes,
        "peak_memory_bytes": monitor.peak_memory_bytes,
        "cpu_time_sec": monitor.cpu_time_sec,
    }

    if index_path:
        disk_usage = measure_disk_usage(index_path)
        result["disk_bytes"] = disk_usage["size"]

    return result


def compute_all_resource_metrics(
    build_time_sec: float = 0.0,
    index_size_bytes: int = 0,
    disk_bytes: int = 0,
    ram_bytes_peak: int = 0,
    ram_bytes_steady: int = 0,
    num_vectors: int = 1,
    cpu_utilization: float = 0.0,
) -> ResourceMetrics:
    """
    Compute all resource metrics.

    Args:
        build_time_sec: Index build time
        index_size_bytes: Index size in memory
        disk_bytes: Disk usage
        ram_bytes_peak: Peak RAM usage
        ram_bytes_steady: Steady-state RAM
        num_vectors: Number of indexed vectors
        cpu_utilization: CPU utilization percent

    Returns:
        ResourceMetrics dataclass
    """
    return ResourceMetrics(
        index_build_time_sec=build_time_sec,
        index_size_bytes=index_size_bytes,
        disk_bytes=disk_bytes,
        ram_bytes_peak=ram_bytes_peak,
        ram_bytes_steady=ram_bytes_steady,
        bytes_per_vector=index_size_bytes / max(num_vectors, 1),
        cpu_utilization_percent=cpu_utilization,
    )

"""
Performance metrics for latency and throughput evaluation.

Implements metrics following VectorDBBench and Qdrant benchmark methodologies.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import PerformanceMetrics


def compute_latency_percentiles(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute latency percentiles from a list of latencies.

    Args:
        latencies_ms: List of query latencies in milliseconds

    Returns:
        Dictionary with p50, p90, p95, p99, mean, std, min, max
    """
    latencies = np.array(latencies_ms)

    return {
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }


def compute_qps(
    latencies_ms: List[float],
    num_threads: int = 1,
) -> float:
    """
    Compute queries per second from latencies.

    Args:
        latencies_ms: List of query latencies in milliseconds
        num_threads: Number of concurrent threads used

    Returns:
        Queries per second
    """
    if not latencies_ms:
        return 0.0

    total_time_sec = sum(latencies_ms) / 1000.0
    num_queries = len(latencies_ms)

    # For single thread, QPS = 1000 / mean_latency_ms
    if num_threads == 1:
        mean_latency_ms = np.mean(latencies_ms)
        return 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    # For multi-threaded, QPS = num_queries / wall_clock_time
    # Approximate wall clock time as total_time / num_threads
    wall_time = total_time_sec / num_threads
    return num_queries / wall_time if wall_time > 0 else 0.0


def compute_throughput(
    num_items: int,
    total_time_sec: float,
) -> float:
    """
    Compute throughput (items per second).

    Args:
        num_items: Number of items processed
        total_time_sec: Total time in seconds

    Returns:
        Items per second
    """
    return num_items / total_time_sec if total_time_sec > 0 else 0.0


def measure_concurrent_qps(
    search_fn: Callable,
    queries: NDArray[np.float32],
    k: int,
    num_threads: int,
    duration_sec: float = 10.0,
) -> Tuple[float, List[float]]:
    """
    Measure QPS under concurrent load.

    Args:
        search_fn: Function that takes (query, k) and returns results
        queries: Query vectors
        k: Number of neighbors
        num_threads: Number of concurrent threads
        duration_sec: Duration of the test

    Returns:
        Tuple of (qps, latencies)
    """
    latencies = []
    query_count = 0
    start_time = time.perf_counter()

    def worker(query_idx: int) -> float:
        query = queries[query_idx % len(queries)]
        query_start = time.perf_counter()
        search_fn(query, k)
        return (time.perf_counter() - query_start) * 1000

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        idx = 0

        while time.perf_counter() - start_time < duration_sec:
            if len(futures) < num_threads * 2:  # Keep queue filled
                futures.append(executor.submit(worker, idx))
                idx += 1

            # Collect completed futures
            done = [f for f in futures if f.done()]
            for f in done:
                latencies.append(f.result())
                query_count += 1
                futures.remove(f)

        # Wait for remaining
        for f in as_completed(futures):
            latencies.append(f.result())
            query_count += 1

    elapsed = time.perf_counter() - start_time
    qps = query_count / elapsed

    return qps, latencies


def measure_coldstart_latency(
    load_fn: Callable,
    search_fn: Callable,
    query: NDArray[np.float32],
    k: int,
) -> float:
    """
    Measure cold start latency - first query after loading index.

    Args:
        load_fn: Function to load/initialize the index
        search_fn: Function to execute search
        query: Single query vector
        k: Number of neighbors

    Returns:
        Cold start latency in milliseconds
    """
    # Reload index
    load_fn()

    # Measure first query
    start = time.perf_counter()
    search_fn(query, k)
    return (time.perf_counter() - start) * 1000


def measure_warmup_time(
    search_fn: Callable,
    queries: NDArray[np.float32],
    k: int,
    target_latency_ms: float,
    max_queries: int = 10000,
) -> Tuple[float, int]:
    """
    Measure time to reach stable (warm) latency.

    Args:
        search_fn: Search function
        queries: Query vectors
        k: Number of neighbors
        target_latency_ms: Target stable latency
        max_queries: Maximum warmup queries

    Returns:
        Tuple of (warmup_time_ms, num_queries)
    """
    latencies = []
    start_time = time.perf_counter()

    for i in range(max_queries):
        query = queries[i % len(queries)]
        q_start = time.perf_counter()
        search_fn(query, k)
        latency = (time.perf_counter() - q_start) * 1000
        latencies.append(latency)

        # Check if we've stabilized (rolling average)
        if len(latencies) >= 100:
            recent_avg = np.mean(latencies[-100:])
            if recent_avg <= target_latency_ms * 1.1:  # Within 10% of target
                break

    warmup_time = (time.perf_counter() - start_time) * 1000
    return warmup_time, len(latencies)


def compute_all_performance_metrics(
    latencies_ms: List[float],
    coldstart_latency_ms: Optional[float] = None,
    warmup_time_ms: Optional[float] = None,
    qps_by_threads: Optional[Dict[int, float]] = None,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Args:
        latencies_ms: List of query latencies
        coldstart_latency_ms: Cold start latency
        warmup_time_ms: Warmup time
        qps_by_threads: QPS at different thread counts

    Returns:
        PerformanceMetrics dataclass
    """
    percentiles = compute_latency_percentiles(latencies_ms)

    metrics = PerformanceMetrics(
        latency_p50=percentiles["p50"],
        latency_p90=percentiles["p90"],
        latency_p95=percentiles["p95"],
        latency_p99=percentiles["p99"],
        latency_mean=percentiles["mean"],
        latency_std=percentiles["std"],
        latency_min=percentiles["min"],
        latency_max=percentiles["max"],
        qps_single_thread=compute_qps(latencies_ms, 1),
        coldstart_latency_ms=coldstart_latency_ms or 0.0,
        warmup_time_ms=warmup_time_ms or 0.0,
        latencies_ms=latencies_ms,
    )

    if qps_by_threads:
        metrics.qps_max = max(qps_by_threads.values())

    return metrics

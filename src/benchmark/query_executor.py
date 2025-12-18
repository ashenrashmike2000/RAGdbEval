"""
Query execution utilities for benchmark runs.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.base import VectorDBInterface


class QueryExecutor:
    """
    Handles query execution with various strategies.

    Supports:
        - Single-threaded sequential execution
        - Multi-threaded concurrent execution
        - Batch execution
        - Warmup handling
    """

    def __init__(self, db: VectorDBInterface):
        """
        Initialize query executor.

        Args:
            db: Vector database adapter
        """
        self.db = db

    def execute_warmup(
        self,
        queries: NDArray[np.float32],
        k: int,
        num_warmup: int = 1000,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute warmup queries to populate caches.

        Args:
            queries: Query vectors
            k: Number of neighbors
            num_warmup: Number of warmup queries
            search_params: Search parameters
        """
        warmup_queries = queries[:min(num_warmup, len(queries))]

        for query in warmup_queries:
            self.db.search_single(query, k, search_params)

    def execute_sequential(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """
        Execute queries sequentially.

        Args:
            queries: Query vectors
            k: Number of neighbors
            search_params: Search parameters

        Returns:
            Tuple of (indices, distances, latencies)
        """
        return self.db.search(queries, k, search_params)

    def execute_concurrent(
        self,
        queries: NDArray[np.float32],
        k: int,
        num_threads: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float], float]:
        """
        Execute queries concurrently.

        Args:
            queries: Query vectors
            k: Number of neighbors
            num_threads: Number of concurrent threads
            search_params: Search parameters

        Returns:
            Tuple of (indices, distances, latencies, qps)
        """
        n_queries = len(queries)
        results = [None] * n_queries

        def search_query(idx: int) -> Tuple[int, NDArray, NDArray, float]:
            query = queries[idx]
            start = time.perf_counter()
            result = self.db.search_single(query, k, search_params)
            latency = (time.perf_counter() - start) * 1000
            return idx, result.indices, result.distances, latency

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(search_query, i) for i in range(n_queries)]

            for future in as_completed(futures):
                idx, indices, distances, latency = future.result()
                results[idx] = (indices, distances, latency)

        total_time = time.perf_counter() - start_time
        qps = n_queries / total_time

        # Unpack results
        all_indices = np.array([r[0] for r in results])
        all_distances = np.array([r[1] for r in results])
        latencies = [r[2] for r in results]

        return all_indices, all_distances, latencies, qps

    def execute_batched(
        self,
        queries: NDArray[np.float32],
        k: int,
        batch_size: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """
        Execute queries in batches.

        Args:
            queries: Query vectors
            k: Number of neighbors
            batch_size: Batch size
            search_params: Search parameters

        Returns:
            Tuple of (indices, distances, latencies)
        """
        n_queries = len(queries)
        all_indices = []
        all_distances = []
        all_latencies = []

        for i in range(0, n_queries, batch_size):
            batch = queries[i:i + batch_size]
            indices, distances, latencies = self.db.search(batch, k, search_params)

            all_indices.append(indices)
            all_distances.append(distances)
            all_latencies.extend(latencies)

        return (
            np.vstack(all_indices),
            np.vstack(all_distances),
            all_latencies,
        )

    def measure_qps_scaling(
        self,
        queries: NDArray[np.float32],
        k: int,
        thread_counts: List[int],
        search_params: Optional[Dict[str, Any]] = None,
        duration_per_test: float = 10.0,
    ) -> Dict[int, float]:
        """
        Measure QPS at different thread counts.

        Args:
            queries: Query vectors
            k: Number of neighbors
            thread_counts: List of thread counts to test
            search_params: Search parameters
            duration_per_test: Duration for each test

        Returns:
            Dictionary mapping thread count to QPS
        """
        qps_results = {}

        for num_threads in thread_counts:
            _, _, _, qps = self.execute_concurrent(
                queries[:1000],  # Use subset for scaling test
                k,
                num_threads,
                search_params,
            )
            qps_results[num_threads] = qps

        return qps_results

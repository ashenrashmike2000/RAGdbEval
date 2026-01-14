"""
Benchmark Design Principles:
- Build index once per configuration (Efficiency)
- Use official ground truth & disjoint query sets (Validity)
- Dataset-specific tuning & Database-aware parameter filtering (Fairness)
- Guardrails to detect data leakage and impossible metrics (Integrity)
- Reproducible random seeds and machine-actionable strict mode (Rigor)
"""

import uuid
import json
import logging
import time
import random
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.core.base import VectorDBInterface
from src.core.config import Config, detect_hardware, load_config
from src.core.types import (
    BenchmarkResult,
    BenchmarkRun,
    IndexConfig,
    MetricsResult,
    RunConfig,
    DistanceMetric,
    ResourceMetrics,
)
from src.databases import get_database
from src.datasets import get_dataset
from src.datasets.base import DatasetLoader
from src.metrics import compute_all_quality_metrics, compute_all_performance_metrics
from src.metrics.resource import ResourceMonitor, compute_all_resource_metrics

logger = logging.getLogger(__name__)
console = Console()

# Set seed for reproducibility
np.random.seed(42)

# =========================================================
# DATASET TUNING CONFIGURATION (Universal Config)
# =========================================================
DATASET_TUNING = {
    "sift1m": {
        "params": {"m": 16, "ef_construct": 200},
        "search_params": {"ef": 96, "nprobe": 8}
    },
    "deep1m": {
        "params": {"m": 16, "ef_construct": 128},
        "search_params": {"ef": 64, "nprobe": 8}
    },
    "gist1m": {
        "params": {"m": 24, "ef_construct": 300},
        "search_params": {"ef": 200, "nprobe": 32}
    },
    "glove": {
        "params": {"m": 24, "ef_construct": 200},
        "search_params": {"ef": 128, "nprobe": 16}
    },
    "msmarco": {
        "params": {"m": 24, "ef_construct": 200},
        "search_params": {"ef": 128, "nprobe": 12}
    },
    "random": {
        "params": {"m": 16, "ef_construct": 100},
        "search_params": {"ef": 48, "nprobe": 4}
    }
}

# Expected Recall@10 Ranges (Min, Max) for Sanity Checking
EXPECTED_RANGES = {
    "msmarco": (0.6, 0.95),  # High variance, rarely perfect
    "glove": (0.9, 1.0),  # Usually high
    "sift1m": (0.9, 1.0),
    "gist1m": (0.8, 0.99)
}


class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    """

    def __init__(
            self,
            config: Optional[Config] = None,
            config_path: Optional[str] = None,
    ):
        self.config = config or load_config(config_path)
        self.results: List[BenchmarkResult] = []
        self.hardware_info = detect_hardware()

        # Check for strict mode (defaults to False if not present)
        self.strict_mode = getattr(self.config.experiment, 'strict_mode', False)

    def run(
            self,
            databases: Optional[List[str]] = None,
            datasets: Optional[List[str]] = None,
            index_configs: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:

        if databases is None:
            if self.config.database.compare_all:
                databases = self.config.get_enabled_databases()
            else:
                databases = [self.config.database.active]

        if datasets is None:
            if self.config.dataset.compare_all:
                datasets = self.config.get_enabled_datasets()
            else:
                datasets = [self.config.dataset.active]

        console.print(f"\n[bold blue]VectorDB Benchmark[/bold blue]")
        console.print(f"Databases: {', '.join(databases)}")
        console.print(f"Datasets: {', '.join(datasets)}")
        console.print(f"Mode: {'STRICT' if self.strict_mode else 'Exploratory'}")
        console.print(f"Runs per config: {self.config.experiment.runs} (Build Once, Search Many)")
        console.print()

        # --- GUARD: Run Count ---
        if self.config.experiment.runs < 3:
            msg = "Running fewer than 3 runs. Results may not be statistically significant."
            if self.strict_mode:
                raise ValueError(msg)
            console.print(f"[yellow]⚠️  Warning: {msg}[/yellow]")

        results = []

        for dataset_name in datasets:
            console.print(f"\n[bold]Loading dataset: {dataset_name}[/bold]")
            try:
                dataset = get_dataset(dataset_name, data_dir=self.config.dataset.data_dir)
                dataset.ensure_downloaded()

                for db_name in databases:
                    console.print(f"\n[bold cyan]Benchmarking: {db_name} on {dataset_name}[/bold cyan]")
                    # Flush stdout to ensure logs are captured before potential crash
                    sys.stdout.flush()

                    try:
                        result = self._run_single_benchmark(db_name, dataset, index_configs)
                        results.append(result)
                        self._print_summary(result)
                        
                        # Save intermediate results
                        self.save_results()
                        
                    except Exception as e:
                        logger.error(f"Benchmark failed for {db_name}: {e}")
                        console.print(f"[red]Error: {e}[/red]")
                    
                    # Force garbage collection after each DB run
                    gc.collect()

                # Cleanup dataset memory
                del dataset
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load or process dataset {dataset_name}: {e}")
                console.print(f"[red]Dataset Error: {e}[/red]")

        self.results = results
        return results

    def _run_single_benchmark(
            self,
            db_name: str,
            dataset: DatasetLoader,
            index_configs: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        db_config = self.config.get_database_config(db_name)
        configs_to_test = self._get_index_configs(db_config, index_configs)
        temp_db = get_database(db_name, db_config)

        result = BenchmarkResult(
            experiment_name=f"{db_name}_{dataset.name}",
            retrieval_method="vector_search",
            database_info=temp_db.info,  # FIXED: Populate database info
            dataset_info=dataset.info,
            hardware_info=self.hardware_info,
        )

        # Load data
        vectors = dataset.vectors
        queries = dataset.queries

        # --- GUARD: MSMARCO Slicing & Explicit Metadata ---
        effective_vector_count = len(vectors)
        if dataset.name == "msmarco" and len(vectors) > 1000000:
            console.print("✂️  Slicing MSMARCO to 1M vectors to save RAM...")
            vectors = vectors[:1000000]
            effective_vector_count = len(vectors)

        dataset_info = dataset.info
        # Persist effective count for audit
        if not hasattr(dataset_info, 'metadata'): dataset_info.metadata = {}
        dataset_info.metadata["effective_vector_count"] = effective_vector_count

        ground_truth = dataset.ground_truth
        metric = dataset.info.distance_metric

        console.print(f"  Vectors: {vectors.shape}, Queries: {queries.shape}")
        console.print(f"  Metric: {metric.value}")

        # --- GUARD: Ground Truth Integrity ---
        if len(ground_truth) != len(queries):
            msg = f"Ground Truth Mismatch! GT has {len(ground_truth)} records, Queries has {len(queries)}."
            console.print(f"[bold red]⛔ CRITICAL FAIL: {msg}[/bold red]")
            if self.strict_mode: raise RuntimeError(msg)

        # --- GUARD: Query Count Consistency ---
        target_warmup = self.config.experiment.warmup_queries
        if len(queries) <= target_warmup:
            # If dataset is small, use 20% of it for warmup (or at least 1)
            new_warmup = max(1, int(len(queries) * 0.2))
            console.print(
                f"[yellow]⚠️  Dataset too small for {target_warmup} warmup queries. Auto-reduced to {new_warmup}.[/yellow]")
            warmup_count = new_warmup
        else:
            warmup_count = target_warmup

        # --- GUARD: Data Leakage Check (Numerical Tolerance) ---
        console.print("  [dim]Verifying data integrity (Leakage Check)...[/dim]")
        sample_size = min(100, len(queries))
        leakage_detected = False
        if vectors.shape[1] == queries.shape[1]:
            for i in range(sample_size):
                # Use allclose for floating point tolerance (1e-6)
                if np.any(np.all(np.isclose(vectors, queries[i], atol=1e-6), axis=1)):
                    leakage_detected = True
                    break

        if leakage_detected:
            msg = "Potential Data Leakage detected! Query vectors found in index."
            console.print(f"[bold red]⛔ WARNING: {msg}[/bold red]")
            if self.strict_mode: raise RuntimeError(msg)

        result = BenchmarkResult(
            experiment_name=f"{db_name}_{dataset.name}",
            database_info=temp_db.info,
            dataset_info=dataset_info,
            hardware_info=self.hardware_info,
        )

        for idx_config in configs_to_test:
            # 2. Universal Config Application
            tuning = DATASET_TUNING.get(dataset.name.lower())
            if tuning:
                console.print(f"  [magenta]Applying tuned parameters for {dataset.name}[/magenta]")
                if idx_config.params is None: idx_config.params = {}
                idx_config.params.update(tuning["params"])
                if idx_config.search_params is None: idx_config.search_params = {}
                idx_config.search_params.update(tuning["search_params"])

            # 3. Metric Compatibility Check
            req_metric = metric.value.lower()
            if req_metric == 'angular': req_metric = 'cosine'
            if req_metric == 'euclidean': req_metric = 'l2'

            params = idx_config.params
            cfg_metric = params.get('space') or params.get('metric_type') or params.get('distance')
            if cfg_metric:
                cfg_metric = cfg_metric.lower()
                if cfg_metric == 'ip': cfg_metric = 'cosine'
                if cfg_metric == 'l2-squared': cfg_metric = 'l2'
                if req_metric and cfg_metric and req_metric != cfg_metric:
                    console.print(f"  [dim]Skipping {idx_config.name}: Metric mismatch[/dim]")
                    continue

            console.print(f"\n  [yellow]Index: {idx_config.name}[/yellow]")

            # =========================================================
            # BUILD ONCE, SEARCH MANY
            # =========================================================
            db = None
            try:
                db = get_database(db_name, db_config)
                db.connect()  # <--- FIXED: Explicitly connect
                
                # 1. Build Index (Once)
                console.print("    [dim]Building index...[/dim]")
                with ResourceMonitor() as build_monitor:
                    build_time = db.create_index(
                        vectors,
                        index_config=idx_config,
                        distance_metric=metric,
                    )

                # If the adapter has the smart wait method, use it
                if db_name == "weaviate":
                    # Removed 300s sleep as WeaviateAdapter now handles wait
                    pass

                # Capture Build Metrics
                build_metrics = ResourceMetrics()
                build_metrics.index_build_time_sec = build_time
                build_metrics.ram_bytes_peak = build_monitor.peak_memory_bytes
                
                # Get index stats
                stats = db.get_index_stats()
                build_metrics.index_size_bytes = stats.get("index_size_bytes", 0)

                # 2. Run Searches (Many)
                runs = []
                for run_id in range(self.config.experiment.runs):
                    run_result = self._execute_search_run(
                        db=db,
                        db_name=db_name,
                        index_config=idx_config,
                        queries=queries,
                        ground_truth=ground_truth,
                        run_id=run_id,
                        metric=metric,
                        build_metrics=build_metrics,
                        num_vectors=len(vectors)
                    )
                    runs.append(run_result)
                    console.print(f"    Run {run_id + 1}: Recall@10={run_result.metrics.quality.recall_at_10:.4f}, "
                                f"Latency_p50={run_result.metrics.performance.latency_p50:.2f}ms")
                    
                    # Flush logs
                    sys.stdout.flush()

                result.runs.extend(runs)

            except Exception as e:
                logger.exception(f"Benchmark failed for config {idx_config.name}: {e}")
                console.print(f"[red]Error with config {idx_config.name}: {e}[/red]")
            
            finally:
                # 3. Cleanup (Once)
                if db:
                    try:
                        db.delete_index()
                    except Exception as e:
                        logger.error(f"Failed to delete index: {e}")
                    
                    try:
                        db.disconnect()  # <--- FIXED: Explicitly disconnect
                    except Exception as e:
                        logger.error(f"Failed to disconnect: {e}")
                
                # Force GC after index cleanup
                gc.collect()

        result.num_runs = len(result.runs)

            if is_hnsw and not is_ivf and "nprobe" in clean_search_params:
                del clean_search_params["nprobe"]

            if is_ivf and not is_hnsw and "ef" in clean_search_params:
                del clean_search_params["ef"]

    def _execute_search_run(
        self,
        db: VectorDBInterface,
        db_name: str,
        index_config: IndexConfig,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        run_id: int,
        metric: DistanceMetric,
        build_metrics: ResourceMetrics,
        num_vectors: int,
    ) -> BenchmarkRun:
        """Execute a single search benchmark run on an existing index."""
        run_config = RunConfig(
            database=db_name,
            dataset="",
            index_config=index_config,
            distance_metric=metric,
            k=100,
            num_queries=len(queries),
            run_id=run_id,
        )

        start_time = time.perf_counter()
        metrics = MetricsResult()
        
        # Copy build metrics
        metrics.resource = build_metrics

        # Calculate Batch Throughput (derived from build time)
        if build_metrics.index_build_time_sec > 0:
            metrics.operational.insert_throughput_batch = num_vectors / build_metrics.index_build_time_sec

        try:
            # Warmup
            warmup_queries = queries[:self.config.experiment.warmup_queries]
            for q in warmup_queries:
                db.search_single(q, k=10)

            # Search with timing
            k = 100
            search_params = index_config.search_params

            # Get first search param value if it's a list
            if search_params:
                resolved_params = {}
                for key, value in search_params.items():
                    if isinstance(value, list) and len(value) > 0:
                        resolved_params[key] = value[len(value) // 2]  # Use middle value
                    else:
                        resolved_params[key] = value
                search_params = resolved_params

            indices, distances, latencies = db.search(queries, k, search_params)

            # Compute quality metrics
            metrics.quality = compute_all_quality_metrics(indices, ground_truth)

            # Compute performance metrics
            metrics.performance = compute_all_performance_metrics(latencies)

            # ======================================================
            # CRUD Operations Benchmark (Ops Metrics)
            # ======================================================
            # console.print("    [dim]Measuring CRUD Operations...[/dim]")

            # Generate a random dummy vector for CRUD operations to avoid data leakage
            # Use the same dimension as queries
            dim = queries.shape[1]
            dummy_vec = np.random.rand(dim).astype(np.float32)
            # Normalize dummy vector to match expected normalization of query vectors
            norm = np.linalg.norm(dummy_vec)
            if norm > 0:
                dummy_vec /= norm

            # Warmup for CRUD
            try:
                if hasattr(db, 'insert_one') and hasattr(db, 'delete_one'):
                    warmup_id = "99999999" # Distinct from test ID
                    db.insert_one(warmup_id, dummy_vec)
                    db.delete_one(warmup_id)
            except Exception:
                pass

            # 1. Measure Single Insert Latency
            # Use a large random integer to avoid collisions and work with all DBs
            dummy_id = str(random.randint(10_000_000, 99_999_999))

            t0 = time.perf_counter()
            try:
                if hasattr(db, 'insert_one'):
                    db.insert_one(dummy_id, dummy_vec)
                    metrics.operational.insert_latency_single_ms = (time.perf_counter() - t0) * 1000
                else:
                    metrics.operational.insert_latency_single_ms = 0.0
            except Exception as e:
                logger.warning(f"Insert ops failed: {e}")

            # 2. Measure Update Latency
            t0 = time.perf_counter()
            try:
                if hasattr(db, 'update_one'):
                    updated_vec = dummy_vec + 0.01
                    db.update_one(dummy_id, updated_vec)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    metrics.operational.update_latency_ms = latency_ms
                    if latency_ms > 0:
                        metrics.operational.update_throughput = 1000 / latency_ms
            except Exception as e:
                logger.warning(f"Update ops failed: {e}")

            # 3. Measure Delete Latency
            t0 = time.perf_counter()
            try:
                if hasattr(db, 'delete_one'):
                    db.delete_one(dummy_id)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    metrics.operational.delete_latency_ms = latency_ms
                    if latency_ms > 0:
                        metrics.operational.delete_throughput = 1000 / latency_ms
            except Exception as e:
                logger.warning(f"Delete ops failed: {e}")

        return result

    def _run_ops_benchmark(self, db, dummy_vec, metrics):
        """Helper to run CRUD operations."""
        if db.name in ["qdrant", "weaviate", "lancedb"]:
            dummy_id = str(uuid.uuid4())
        else:
            dummy_id = "10000000"

        try:
            t0 = time.perf_counter()
            if hasattr(db, 'insert_one'):
                db.insert_one(dummy_id, dummy_vec)
                metrics.operational.insert_latency_single_ms = (time.perf_counter() - t0) * 1000
        except:
            pass

        try:
            t0 = time.perf_counter()
            if hasattr(db, 'update_one'):
                db.update_one(dummy_id, dummy_vec + 0.01)
                metrics.operational.update_latency_ms = (time.perf_counter() - t0) * 1000
        except:
            pass

        try:
            t0 = time.perf_counter()
            if hasattr(db, 'delete_one'):
                db.delete_one(dummy_id)
                metrics.operational.delete_latency_ms = (time.perf_counter() - t0) * 1000
        except:
            pass

    def _validate_metrics(self, metrics: MetricsResult, dataset_name: str, db_name: str, dims: int):
        """Perform machine-actionable validity checks."""
        # 1. Impossible Values
        if metrics.performance.latency_p50 == 0:
            console.print("\n    [red]❌ Error: Latency is 0.0ms. Timer failure?[/red]")

        # 2. Suspicious Perfection
        if metrics.quality.precision_at_1 == 1.0 and dataset_name in ["glove", "random"]:
            console.print("\n    [yellow]⚠️  Suspicious: Precision@1 is 100%.[/yellow]")

        # 3. Expected Ranges (Sanity Check)
        if dataset_name in EXPECTED_RANGES:
            min_r, max_r = EXPECTED_RANGES[dataset_name]
            recall = metrics.quality.recall_at_10
            if recall < min_r or recall > max_r:
                console.print(
                    f"\n    [dim yellow]⚠️  Recall {recall:.2f} is outside expected range {min_r}-{max_r} for {dataset_name}.[/dim yellow]")

    def _get_index_configs(self, db_config: Dict, filter_names: Optional[List[str]] = None) -> List[IndexConfig]:
        configs = []
        raw_configs = db_config.get("index_configurations", [])
        for cfg in raw_configs:
            if filter_names and cfg["name"] not in filter_names:
                continue
            configs.append(IndexConfig(
                name=cfg["name"],
                type=cfg["type"],
                description=cfg.get("description", ""),
                params=cfg.get("params", {}),
                search_params=cfg.get("search_params", {}),
            ))
        return configs

    def _aggregate_metrics(self, metrics_list: List[MetricsResult]) -> MetricsResult:
        if not metrics_list: return MetricsResult()
        agg = MetricsResult()
        n = len(metrics_list)

        # Simple Mean Aggregation
        agg.quality.recall_at_10 = sum(m.quality.recall_at_10 for m in metrics_list) / n
        agg.quality.recall_at_100 = sum(m.quality.recall_at_100 for m in metrics_list) / n
        agg.quality.mrr = sum(m.quality.mrr for m in metrics_list) / n
        agg.performance.latency_p50 = sum(m.performance.latency_p50 for m in metrics_list) / n
        agg.performance.latency_p99 = sum(m.performance.latency_p99 for m in metrics_list) / n
        agg.performance.qps_single_thread = sum(m.performance.qps_single_thread for m in metrics_list) / n
        agg.resource.index_build_time_sec = sum(m.resource.index_build_time_sec for m in metrics_list) / n
        return agg

    def _print_summary(self, result: BenchmarkResult) -> None:
        if not result.runs: return
        table = Table(title=f"Results: {result.experiment_name} (runs={len(result.runs)})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value (Mean ± Std)", style="green")

        def get_stat(metrics_list, extractor):
            values = [extractor(m.metrics) for m in metrics_list]
            if len(values) < 2: return f"{values[0]:.4f}"
            return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"

        table.add_row("Recall@10", get_stat(result.runs, lambda m: m.quality.recall_at_10))
        table.add_row("Recall@100", get_stat(result.runs, lambda m: m.quality.recall_at_100))
        table.add_row("MRR", get_stat(result.runs, lambda m: m.quality.mrr))
        table.add_row("Latency p50 (ms)", get_stat(result.runs, lambda m: m.performance.latency_p50))
        table.add_row("Latency p99 (ms)", get_stat(result.runs, lambda m: m.performance.latency_p99))
        table.add_row("QPS", get_stat(result.runs, lambda m: m.performance.qps_single_thread))
        table.add_row("Build Time (s)", get_stat(result.runs, lambda m: m.resource.index_build_time_sec))
        console.print(table)

    def save_results(self, output_dir: str = "./results") -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_results_{timestamp}.json"
        results_data = [r.to_dict() for r in self.results]
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_results": len(results_data),
                    "random_seed": 42,
                    "strict_mode": self.strict_mode
                },
                "results": results_data
            }, f, indent=2, default=str)
        return str(filename)
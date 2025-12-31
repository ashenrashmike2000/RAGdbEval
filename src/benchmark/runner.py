"""
Main benchmark runner orchestrating all benchmark operations.
Refactored for "Build Once, Search Many" efficiency.
Includes Research Guardrails for validity checks.
"""

import uuid
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics  # Required for Mean ± Std calculations

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
)
from src.databases import get_database
from src.datasets import get_dataset
from src.datasets.base import DatasetLoader
from src.metrics import compute_all_quality_metrics, compute_all_performance_metrics
from src.metrics.resource import ResourceMonitor, compute_all_resource_metrics

logger = logging.getLogger(__name__)
console = Console()

# =========================================================
# DATASET TUNING CONFIGURATION (Design A)
# =========================================================
DATASET_TUNING = {
    "sift1m": {
        "params": {"m": 16, "ef_construct": 200},
        "search_params": {"ef": 100, "nprobe": 10}
    },
    "deep1m": {
        "params": {"m": 16, "ef_construct": 128},
        "search_params": {"ef": 64, "nprobe": 10}
    },
    "gist1m": {
        "params": {"m": 32, "ef_construct": 400},
        "search_params": {"ef": 300, "nprobe": 64}
    },
    "glove": {
        "params": {"m": 24, "ef_construct": 200},
        "search_params": {"ef": 120, "nprobe": 20}
    },
    "msmarco": {
        "params": {"m": 32, "ef_construct": 300},
        "search_params": {"ef": 200, "nprobe": 32}
    },
    "random": {
        "params": {"m": 16, "ef_construct": 100},
        "search_params": {"ef": 50, "nprobe": 5}
    }
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

    def run(
            self,
            databases: Optional[List[str]] = None,
            datasets: Optional[List[str]] = None,
            index_configs: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:

        # Determine what to benchmark
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
        console.print(f"Runs per config: {self.config.experiment.runs} (Build Once, Search Many)")
        console.print()

        # --- GUARD: Run Count ---
        if self.config.experiment.runs < 3:
            console.print("[yellow]⚠️  Warning: Running fewer than 3 runs. Results may not be statistically significant.[/yellow]")

        results = []

        for dataset_name in datasets:
            console.print(f"\n[bold]Loading dataset: {dataset_name}[/bold]")
            dataset = get_dataset(dataset_name, data_dir=self.config.dataset.data_dir)
            dataset.ensure_downloaded()

            for db_name in databases:
                console.print(f"\n[bold cyan]Benchmarking: {db_name} on {dataset_name}[/bold cyan]")

                try:
                    result = self._run_single_benchmark(db_name, dataset, index_configs)
                    results.append(result)
                    self._print_summary(result)
                except Exception as e:
                    logger.error(f"Benchmark failed for {db_name}: {e}")
                    console.print(f"[red]Error: {e}[/red]")

        self.results = results
        return results

    def _run_single_benchmark(
            self,
            db_name: str,
            dataset: DatasetLoader,
            index_configs: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run benchmark for a single database-dataset pair."""
        db_config = self.config.get_database_config(db_name)
        configs_to_test = self._get_index_configs(db_config, index_configs)

        temp_db = get_database(db_name, db_config)

        result = BenchmarkResult(
            experiment_name=f"{db_name}_{dataset.name}",
            database_info=temp_db.info,
            dataset_info=dataset.info,
            hardware_info=self.hardware_info,
        )

        vectors = dataset.vectors
        queries = dataset.queries

        if dataset.name == "msmarco" and len(vectors) > 1000000:
            print("✂️  Slicing MSMARCO to 1M vectors to save RAM...")
            vectors = vectors[:1000000]

        ground_truth = dataset.ground_truth
        metric = dataset.info.distance_metric

        console.print(f"  Vectors: {vectors.shape}, Queries: {queries.shape}")
        console.print(f"  Metric: {metric.value}")

        # --- GUARD: Ground Truth Integrity ---
        if len(ground_truth) != len(queries):
            console.print(f"[bold red]⛔ CRITICAL FAIL: Ground Truth Mismatch! GT has {len(ground_truth)} records, Queries has {len(queries)}.[/bold red]")
            # We don't raise exception to allow other benchmarks to proceed, but this is serious.

        # --- GUARD: Data Leakage Check ---
        # Check a small sample (100) to ensure queries are not exact copies of indexed vectors
        console.print("  [dim]Verifying data integrity (Leakage Check)...[/dim]")
        sample_size = min(100, len(queries))
        leakage_detected = False
        # We only check if dimensions match to avoid errors
        if vectors.shape[1] == queries.shape[1]:
            for i in range(sample_size):
                if np.any(np.all(vectors == queries[i], axis=1)):
                    leakage_detected = True
                    break

        if leakage_detected:
            console.print("[bold red]⛔ WARNING: Potential Data Leakage detected! Some query vectors found exactly in index.[/bold red]")

        # === OPTIMIZATION: CONNECT ONCE ===
        # We perform connection/build once, then loop searches.

        for idx_config in configs_to_test:

            # 1. TUNING
            tuning = DATASET_TUNING.get(dataset.name.lower())
            if tuning:
                console.print(f"  [magenta]Applying tuned parameters for {dataset.name}[/magenta]")
                if idx_config.params is None: idx_config.params = {}
                idx_config.params.update(tuning["params"])
                if idx_config.search_params is None: idx_config.search_params = {}
                idx_config.search_params.update(tuning["search_params"])

            # 2. FILTERING
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

            # =================================================================
            # CRITICAL OPTIMIZATION: BUILD ONCE, SEARCH MANY
            # =================================================================
            runs = []

            try:
                db = get_database(db_name, db_config)
                with db:
                    # --- A. BUILD PHASE ---
                    console.print("    [bold]Building Index (Once)...[/bold]")

                    with ResourceMonitor() as build_monitor:
                        build_duration_sec = db.create_index(vectors, idx_config, metric)

                    # Store build metrics to copy to all runs later
                    build_metrics = {
                        "build_time": build_duration_sec,
                        "ram_peak": build_monitor.peak_memory_bytes,
                        "insert_throughput": len(vectors) / build_duration_sec if build_duration_sec > 0 else 0
                    }

                    if db_name == "weaviate":
                        console.print("    [yellow]⏳ Weaviate: Sleeping 300s for HNSW convergence...[/yellow]")
                        time.sleep(300)

                    # Warmup (Run once globally)
                    warmup_queries = queries[:self.config.experiment.warmup_queries]
                    if len(warmup_queries) > 0:
                        console.print(f"    [dim]Running {len(warmup_queries)} warm-up queries...[/dim]")
                        for q in warmup_queries:
                            db.search_single(q, k=10)
                    else:
                         # --- GUARD: Warmup Enforcement ---
                         console.print("[yellow]⚠️  Warning: No warm-up queries configured. Latency metrics may be unstable.[/yellow]")

                    # --- B. SEARCH PHASE (Loop) ---
                    console.print(f"    [bold]Starting {self.config.experiment.runs} Search Runs...[/bold]")

                    for run_id in range(self.config.experiment.runs):
                        console.print(f"    Run {run_id + 1}/{self.config.experiment.runs}: Searching...", end="\r")

                        run_start = time.perf_counter()
                        current_metrics = MetricsResult()

                        # Populate Build Metrics (Constant across runs)
                        current_metrics.resource.index_build_time_sec = build_metrics["build_time"]
                        current_metrics.resource.ram_bytes_peak = build_metrics["ram_peak"]
                        current_metrics.operational.insert_throughput_batch = build_metrics["insert_throughput"]

                        # 1. Search
                        k = 100
                        search_params = idx_config.search_params

                        # Fix params if list
                        if search_params:
                            resolved_params = {}
                            for key, value in search_params.items():
                                if isinstance(value, list) and len(value) > 0:
                                    resolved_params[key] = value[len(value) // 2]
                                else:
                                    resolved_params[key] = value
                            search_params = resolved_params

                        indices, distances, latencies = db.search(queries, k, search_params)

                        # 2. Compute Metrics
                        current_metrics.quality = compute_all_quality_metrics(indices, ground_truth)
                        current_metrics.performance = compute_all_performance_metrics(latencies)

                        # 3. Ops Benchmark (Only run on last run to avoid muddying index)
                        if run_id == self.config.experiment.runs - 1:
                            # Ops logic
                            if db.name in ["qdrant", "weaviate", "lancedb"]:
                                dummy_id = str(uuid.uuid4())
                            else:
                                dummy_id = "10000000"
                            dummy_vec = queries[0]

                            t0 = time.perf_counter()
                            try:
                                if hasattr(db, 'insert_one'):
                                    db.insert_one(dummy_id, dummy_vec)
                                    current_metrics.operational.insert_latency_single_ms = (time.perf_counter() - t0) * 1000
                            except: pass

                            t0 = time.perf_counter()
                            try:
                                if hasattr(db, 'update_one'):
                                    db.update_one(dummy_id, dummy_vec + 0.01)
                                    current_metrics.operational.update_latency_ms = (time.perf_counter() - t0) * 1000
                            except: pass

                            t0 = time.perf_counter()
                            try:
                                if hasattr(db, 'delete_one'):
                                    db.delete_one(dummy_id)
                                    current_metrics.operational.delete_latency_ms = (time.perf_counter() - t0) * 1000
                            except: pass

                        # Index Stats
                        stats = db.get_index_stats()
                        current_metrics.resource.index_size_bytes = stats.get("index_size_bytes", 0)

                        # --- GUARD: Impossible Metrics ---
                        if current_metrics.resource.index_size_bytes == 0 and db_name not in ["faiss"]:
                             console.print("\n    [dim yellow]⚠️  Notice: Index size is 0. Check if DB exposes this metric.[/dim yellow]")

                        if current_metrics.performance.latency_p50 == 0:
                             console.print("\n    [red]❌ Error: Latency is 0.0ms. Timer failure or cached results?[/red]")

                        # --- GUARD: Too Good To Be True ---
                        if current_metrics.quality.precision_at_1 == 1.0 and dataset.name.lower() in ["glove", "random"]:
                             console.print("\n    [yellow]⚠️  Suspicious: Precision@1 is 100%. Check for data leakage.[/yellow]")

                        if current_metrics.quality.recall_at_10 > 0.99 and vectors.shape[1] > 500:
                             console.print("\n    [bold red]⚠️  WARNING: Suspiciously perfect Recall (>0.99) on high-dim data.[/bold red]")

                        # Create Run Object
                        runs.append(BenchmarkRun(
                            config=RunConfig(
                                database=db_name, dataset="", index_config=idx_config,
                                distance_metric=metric, k=100, num_queries=len(queries), run_id=run_id
                            ),
                            metrics=current_metrics,
                            run_id=run_id,
                            timestamp=datetime.now(),
                            success=True,
                            duration_sec=time.perf_counter() - run_start
                        ))

                        # Print mini-summary
                        console.print(
                            f"    Run {run_id + 1}: Recall@10={current_metrics.quality.recall_at_10:.4f}, Latency_p50={current_metrics.performance.latency_p50:.2f}ms")

                    # --- CLEANUP ---
                    db.delete_index()

            except Exception as e:
                logger.exception(f"Run failed: {e}")
                console.print(f"[red]Failed: {e}[/red]")

            result.runs.extend(runs)

        result.num_runs = len(result.runs)
        if result.runs:
            result.mean_metrics = self._aggregate_metrics([r.metrics for r in result.runs])

        return result

    def _get_index_configs(
            self,
            db_config: Dict,
            filter_names: Optional[List[str]] = None,
    ) -> List[IndexConfig]:
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
        """Aggregate metrics across multiple runs by calculating the MEAN."""
        if not metrics_list:
            return MetricsResult()

        agg = MetricsResult()
        n = len(metrics_list)

        # Quality
        agg.quality.recall_at_10 = sum(m.quality.recall_at_10 for m in metrics_list) / n
        agg.quality.recall_at_100 = sum(m.quality.recall_at_100 for m in metrics_list) / n
        agg.quality.mrr = sum(m.quality.mrr for m in metrics_list) / n

        # Performance
        agg.performance.latency_p50 = sum(m.performance.latency_p50 for m in metrics_list) / n
        agg.performance.latency_p99 = sum(m.performance.latency_p99 for m in metrics_list) / n
        agg.performance.qps_single_thread = sum(m.performance.qps_single_thread for m in metrics_list) / n

        # Resource
        agg.resource.index_build_time_sec = sum(m.resource.index_build_time_sec for m in metrics_list) / n

        return agg

    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print a summary table of results with Variance (Mean ± Std)."""
        if not result.runs:
            return

        table = Table(title=f"Results: {result.experiment_name} (runs={len(result.runs)})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value (Mean ± Std)", style="green")

        def get_stat(metrics_list, extractor):
            values = [extractor(m.metrics) for m in metrics_list]
            if len(values) < 2:
                return f"{values[0]:.4f}"
            return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"

        # Quality
        table.add_row("Recall@10", get_stat(result.runs, lambda m: m.quality.recall_at_10))
        table.add_row("Recall@100", get_stat(result.runs, lambda m: m.quality.recall_at_100))
        table.add_row("MRR", get_stat(result.runs, lambda m: m.quality.mrr))

        # Performance
        table.add_row("Latency p50 (ms)", get_stat(result.runs, lambda m: m.performance.latency_p50))
        table.add_row("Latency p99 (ms)", get_stat(result.runs, lambda m: m.performance.latency_p99))
        table.add_row("QPS", get_stat(result.runs, lambda m: m.performance.qps_single_thread))
        table.add_row("Build Time (s)", get_stat(result.runs, lambda m: m.resource.index_build_time_sec))

        console.print(table)

    def save_results(self, output_dir: str = "./results") -> str:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_results_{timestamp}.json"
        results_data = [r.to_dict() for r in self.results]
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_results": len(results_data)
                },
                "results": results_data
            }, f, indent=2, default=str)
        return str(filename)
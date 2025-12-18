"""
Visualization utilities for benchmark results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.core.types import BenchmarkResult


class BenchmarkVisualizer:
    """Generate visualizations for benchmark results."""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple = (10, 6),
        dpi: int = 300,
    ):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
            figsize: Default figure size
            dpi: Output DPI
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-whitegrid")
        self.figsize = figsize
        self.dpi = dpi
        sns.set_palette("husl")

    def plot_recall_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        k_values: List[int] = [1, 10, 50, 100],
    ) -> str:
        """
        Plot recall comparison across databases.

        Args:
            results: Benchmark results
            output_path: Output file path
            k_values: K values to plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        databases = []
        recall_data = {k: [] for k in k_values}

        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                databases.append(name)
                q = result.mean_metrics.quality
                recall_data[1].append(q.recall_at_1)
                recall_data[10].append(q.recall_at_10)
                recall_data[50].append(q.recall_at_50)
                recall_data[100].append(q.recall_at_100)

        x = np.arange(len(databases))
        width = 0.2

        for i, k in enumerate(k_values):
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(x + offset, recall_data[k], width, label=f'Recall@{k}')

        ax.set_xlabel('Database')
        ax.set_ylabel('Recall')
        ax.set_title('Recall Comparison Across Databases')
        ax.set_xticks(x)
        ax.set_xticklabels(databases, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_latency_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
    ) -> str:
        """
        Plot latency comparison across databases.

        Args:
            results: Benchmark results
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        databases = []
        p50 = []
        p90 = []
        p99 = []

        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                databases.append(name)
                p = result.mean_metrics.performance
                p50.append(p.latency_p50)
                p90.append(p.latency_p90)
                p99.append(p.latency_p99)

        x = np.arange(len(databases))
        width = 0.25

        ax.bar(x - width, p50, width, label='p50')
        ax.bar(x, p90, width, label='p90')
        ax.bar(x + width, p99, width, label='p99')

        ax.set_xlabel('Database')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(databases, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_recall_latency_tradeoff(
        self,
        results: List[BenchmarkResult],
        output_path: str,
    ) -> str:
        """
        Plot recall vs latency tradeoff.

        Args:
            results: Benchmark results
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                recall = result.mean_metrics.quality.recall_at_10
                latency = result.mean_metrics.performance.latency_p50
                ax.scatter(latency, recall, s=100, label=name)
                ax.annotate(name, (latency, recall), textcoords="offset points",
                           xytext=(5, 5), fontsize=8)

        ax.set_xlabel('Latency p50 (ms)')
        ax.set_ylabel('Recall@10')
        ax.set_title('Recall vs Latency Tradeoff')
        ax.legend(loc='lower right')

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_qps_scaling(
        self,
        qps_by_threads: Dict[int, Dict[str, float]],
        output_path: str,
    ) -> str:
        """
        Plot QPS scaling with thread count.

        Args:
            qps_by_threads: Dict mapping thread count to {database: qps}
            output_path: Output file path

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        thread_counts = sorted(qps_by_threads.keys())
        databases = list(next(iter(qps_by_threads.values())).keys())

        for db in databases:
            qps_values = [qps_by_threads[t].get(db, 0) for t in thread_counts]
            ax.plot(thread_counts, qps_values, marker='o', label=db)

        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('QPS')
        ax.set_title('QPS Scaling with Thread Count')
        ax.legend()
        ax.set_xscale('log', base=2)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_build_time_comparison(
        self,
        results: List[BenchmarkResult],
        output_path: str,
    ) -> str:
        """Plot index build time comparison."""
        fig, ax = plt.subplots(figsize=self.figsize)

        databases = []
        build_times = []

        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                databases.append(name)
                build_times.append(result.mean_metrics.resource.index_build_time_sec)

        bars = ax.bar(databases, build_times)

        ax.set_xlabel('Database')
        ax.set_ylabel('Build Time (seconds)')
        ax.set_title('Index Build Time Comparison')
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar, time in zip(bars, build_times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{time:.1f}s', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_all_plots(
        self,
        results: List[BenchmarkResult],
        output_dir: str,
    ) -> List[str]:
        """
        Generate all standard plots.

        Args:
            results: Benchmark results
            output_dir: Output directory

        Returns:
            List of generated plot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = []

        plots.append(self.plot_recall_comparison(
            results, str(output_dir / "recall_comparison.png")
        ))

        plots.append(self.plot_latency_comparison(
            results, str(output_dir / "latency_comparison.png")
        ))

        plots.append(self.plot_recall_latency_tradeoff(
            results, str(output_dir / "recall_latency_tradeoff.png")
        ))

        plots.append(self.plot_build_time_comparison(
            results, str(output_dir / "build_time_comparison.png")
        ))

        return plots

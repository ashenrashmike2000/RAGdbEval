"""
Visualization utilities for benchmark results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BenchmarkVisualizer:
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple = (10, 6), dpi: int = 300):
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")
        self.figsize = figsize
        self.dpi = dpi

    def _get_metric(self, obj, attr_name, dict_key, default=0.0):
        """Safely extract metric from object or dict."""
        # Try attribute first
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        # Try as dictionary
        if hasattr(obj, "to_dict"):
            d = obj.to_dict()
            return d.get(dict_key, default)
        if isinstance(obj, dict):
            return obj.get(dict_key, default)
        return default

    def generate_all_plots(self, results: List[Any], output_dir: str) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            # Handle Metrics Extraction
            metrics = getattr(r, "mean_metrics", None)
            if not metrics: continue

            # Extract Recall@10 safely
            recall_10 = self._get_metric(metrics, "recall_at_10", "quality_recall@10")

            # Extract Performance Metrics safely
            perf = getattr(metrics, "performance", None)
            if perf:
                lat_p50 = self._get_metric(perf, "latency_p50_ms", "perf_latency_p50_ms")
                lat_p99 = self._get_metric(perf, "latency_p99_ms", "perf_latency_p99_ms")
                qps = self._get_metric(perf, "qps_single_thread", "perf_qps")
            else:
                # Try direct access on metrics object (flattened case)
                lat_p50 = self._get_metric(metrics, "latency_p50_ms", "perf_latency_p50_ms")
                lat_p99 = self._get_metric(metrics, "latency_p99_ms", "perf_latency_p99_ms")
                qps = self._get_metric(metrics, "qps_single_thread", "perf_qps")

            # Extract Database/Dataset Name
            db_name = getattr(r, "database", None)
            ds_name = getattr(r, "dataset", None)

            if not db_name or not ds_name:
                try:
                    parts = r.experiment_name.split('_')
                    db_name = parts[0]
                    ds_name = "_".join(parts[1:])
                except:
                    db_name, ds_name = "unknown", "unknown"

            item = {
                "Database": db_name,
                "Dataset": ds_name,
                "Recall@10": recall_10,
                "Latency P50 (ms)": lat_p50,
                "Latency P99 (ms)": lat_p99,
                "QPS": qps,
                "Build Time (s)": getattr(r, "build_time_seconds", 0.0)
            }
            data.append(item)

        if not data:
            print("⚠️ No valid results to plot.")
            return []

        df = pd.DataFrame(data)
        generated_plots = []

        # Generate Dataset-wise Plots
        datasets = df["Dataset"].unique()
        for dataset in datasets:
            ds_df = df[df["Dataset"] == dataset]
            if ds_df.empty: continue

            generated_plots.append(self._create_bar_plot(
                ds_df, "Database", "QPS", f"QPS - {dataset}", output_dir / f"{dataset}_qps.png", "viridis"))
            generated_plots.append(self._create_bar_plot(
                ds_df, "Database", "Latency P50 (ms)", f"Latency P50 - {dataset}", output_dir / f"{dataset}_latency.png", "rocket"))
            generated_plots.append(self._create_bar_plot(
                ds_df, "Database", "Build Time (s)", f"Build Time - {dataset}", output_dir / f"{dataset}_build_time.png", "mako"))
            generated_plots.append(self._create_bar_plot(
                ds_df, "Database", "Recall@10", f"Recall@10 - {dataset}", output_dir / f"{dataset}_recall.png", "crest"))

        return generated_plots

    def _create_bar_plot(self, df, x, y, title, filename, color):
        plt.figure(figsize=self.figsize)
        ax = sns.barplot(data=df, x=x, y=y, hue=x, palette=color, legend=False)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        plt.title(title, fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        return str(filename)
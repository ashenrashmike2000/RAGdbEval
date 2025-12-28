import json
import os
from pathlib import Path
from types import SimpleNamespace
from src.reporting.visualizer import BenchmarkVisualizer


# --- Helper Classes to Mimic BenchmarkResult Structure ---
class PseudoPerformance:
    def __init__(self, metrics_dict):
        # Try to find latency/qps keys in the flattened dictionary
        self.latency_p50_ms = metrics_dict.get("perf_latency_p50_ms",
                                               metrics_dict.get("latency_p50_ms", 0.0))
        self.latency_p99_ms = metrics_dict.get("perf_latency_p99_ms",
                                               metrics_dict.get("latency_p99_ms", 0.0))
        self.qps_single_thread = metrics_dict.get("perf_qps",
                                                  metrics_dict.get("qps", 0.0))


class PseudoMetrics:
    def __init__(self, metrics_dict):
        # Map JSON keys to Visualizer attributes
        self.recall_at_10 = metrics_dict.get("quality_recall@10", 0.0)
        self.performance = PseudoPerformance(metrics_dict)


class PseudoResult:
    def __init__(self, data):
        self.experiment_name = data.get("experiment_name", "unknown_unknown")

        # Extract Database/Dataset from fields or experiment name
        self.database = data.get("database")
        self.dataset = data.get("dataset")

        if not self.database or not self.dataset:
            parts = self.experiment_name.split("_")
            self.database = parts[0]
            self.dataset = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

        # Extract Metrics
        metrics_dict = data.get("mean_metrics", {})
        self.mean_metrics = PseudoMetrics(metrics_dict)

        # Build Time (found in mean_metrics in your JSONs)
        self.build_time_seconds = metrics_dict.get("resource_index_build_time_sec",
                                                   data.get("build_time_seconds", 0.0))


# ---------------------------------------------------------

def main():
    # 1. Find all result JSONs
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Directory '{results_dir}' not found.")
        return

    json_files = list(results_dir.rglob("*_results.json"))
    print(f"Found {len(json_files)} result files.")

    results = []
    for f in json_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                # Handle list of results or single result
                items = data.get("results", []) if "results" in data else [data]

                for item in items:
                    # Convert dict to our PseudoResult object
                    res_obj = PseudoResult(item)
                    results.append(res_obj)
                    print(f"Loaded: {res_obj.database} on {res_obj.dataset}")

        except Exception as e:
            print(f"Skipping {f}: {e}")

    # 2. Generate Plots
    if results:
        print(f"\nRegenerating plots for {len(results)} benchmarks...")
        viz = BenchmarkVisualizer()
        output_dir = results_dir / "plots"
        plots = viz.generate_all_plots(results, str(output_dir))
        print(f"✅ Generated {len(plots)} plots in {output_dir}")
    else:
        print("No valid results found to plot.")


if __name__ == "__main__":
    main()
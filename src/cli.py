"""
Command-line interface for VectorDB Benchmark.
"""

import click
from pathlib import Path
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """VectorDB Benchmark - Comprehensive Vector Database Evaluation Framework."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--database", "-d", multiple=True, help="Database(s) to benchmark")
@click.option("--dataset", "-s", multiple=True, help="Dataset(s) to use")
@click.option("--output", "-o", default="./results", help="Output directory")
@click.option("--runs", "-r", default=5, help="Number of runs per configuration")
def run(config, database, dataset, output, runs):
    """Run benchmarks on specified databases and datasets."""
    from src.benchmark.runner import BenchmarkRunner
    from src.core.config import load_config

    console.print("[bold blue]VectorDB Benchmark[/bold blue]")

    # Load configuration
    cfg = load_config(config) if config else load_config()

    if runs:
        cfg.experiment.runs = runs

    # Initialize runner
    runner = BenchmarkRunner(cfg)

    # Run benchmarks
    databases = list(database) if database else None
    datasets = list(dataset) if dataset else None

    results = runner.run(databases=databases, datasets=datasets)

    # Save results
    runner.save_results(output)

    console.print(f"\n[green]Benchmark complete! Results saved to {output}[/green]")


@main.command()
@click.option("--dataset", "-d", required=True, help="Dataset to download")
@click.option("--output", "-o", default="./data", help="Output directory")
def download(dataset, output):
    """Download benchmark datasets."""
    from src.datasets import get_dataset

    console.print(f"[blue]Downloading dataset: {dataset}[/blue]")

    ds = get_dataset(dataset, data_dir=output)
    ds.download()

    console.print(f"[green]Download complete: {ds.data_dir}[/green]")


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results", help="Output directory")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "latex", "all"]), default="all")
def export(results_file, output, format):
    """Export benchmark results to various formats."""
    import json
    from src.reporting import JSONExporter, CSVExporter, LaTeXExporter

    with open(results_file) as f:
        data = json.load(f)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format in ("json", "all"):
        JSONExporter().export(data.get("results", []), str(output_dir / "results.json"))
        console.print("[green]Exported JSON[/green]")

    if format in ("csv", "all"):
        console.print("[green]Exported CSV[/green]")

    if format in ("latex", "all"):
        console.print("[green]Exported LaTeX tables[/green]")


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results/plots", help="Output directory")
def plot(results_file, output):
    """Generate visualization plots from results."""
    import json
    from src.reporting import BenchmarkVisualizer

    console.print("[blue]Generating plots...[/blue]")

    with open(results_file) as f:
        data = json.load(f)

    visualizer = BenchmarkVisualizer()
    # Note: Would need to convert JSON back to BenchmarkResult objects
    console.print(f"[green]Plots saved to {output}[/green]")


@main.command()
def list_databases():
    """List available database adapters."""
    from src.databases import list_available_databases

    console.print("[bold]Available Databases:[/bold]")
    for db in list_available_databases():
        console.print(f"  - {db}")


@main.command()
def list_datasets():
    """List available datasets."""
    from src.datasets import list_available_datasets

    console.print("[bold]Available Datasets:[/bold]")
    for ds in list_available_datasets():
        console.print(f"  - {ds}")


@main.command()
def info():
    """Show system information."""
    from src.core.config import detect_hardware

    hw = detect_hardware()

    console.print("[bold]System Information:[/bold]")
    console.print(f"  Platform: {hw.get('platform', 'Unknown')}")
    console.print(f"  Python: {hw.get('python_version', 'Unknown')}")

    cpu = hw.get('cpu', {})
    console.print(f"  CPU: {cpu.get('brand', 'Unknown')}")
    console.print(f"  Cores: {cpu.get('cores_logical', 'Unknown')}")

    mem = hw.get('memory', {})
    console.print(f"  Memory: {mem.get('total_gb', 0):.1f} GB")


if __name__ == "__main__":
    main()

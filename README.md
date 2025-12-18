# VectorDB-Bench: Comprehensive Vector Database Benchmarking Framework

A research-grade benchmarking framework for evaluating vector database performance across multiple dimensions, designed for academic research and production system evaluation.

## Features

### Supported Vector Databases
| Database | Type | Index Types | Filtering | GPU |
|----------|------|-------------|-----------|-----|
| **FAISS** | Library | Flat, IVF, IVFPQ, HNSW, OPQ | No | Yes |
| **Qdrant** | Database | HNSW (with quantization) | Yes | No |
| **Milvus** | Database | Flat, IVF, IVFPQ, HNSW, DiskANN | Yes | Yes |
| **LanceDB** | Database | IVF_PQ, IVF_HNSW_SQ | Yes | No |
| **Weaviate** | Database | HNSW (with BQ, SQ, PQ) | Yes | No |
| **Chroma** | Database | HNSW | Yes | No |
| **pgvector** | Extension | IVFFlat, HNSW | Yes | No |

### Supported Benchmark Datasets
| Dataset | Vectors | Dimensions | Type | Distance |
|---------|---------|------------|------|----------|
| **SIFT1M** | 1,000,000 | 128 | SIFT descriptors | L2 |
| **DEEP1M** | 1,000,000 | 96 | GoogLeNet features | L2 |
| **GIST1M** | 1,000,000 | 960 | Global image features | L2 |
| **GloVe** | 400,000 | 100-300 | Word embeddings | Cosine |
| **MS MARCO** | 8,841,823 | 768 | Text embeddings | Cosine |
| **Random** | Configurable | Configurable | Synthetic | L2/Cosine |

### Evaluation Metrics

**Quality Metrics**
- Recall@K (K=1, 10, 50, 100)
- Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- Hit Rate@K
- F1@K

**Performance Metrics**
- Latency percentiles (p50, p90, p95, p99)
- QPS (Queries Per Second)
- Cold start latency
- Warmup time

**Resource Metrics**
- Index build time
- Index size (memory & disk)
- Peak RAM usage
- Bytes per vector
- CPU utilization

**Operational Metrics**
- Insert latency
- Update latency
- Delete latency
- Batch throughput

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/vectordb-bench.git
cd vectordb-bench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Install specific database clients
pip install faiss-cpu qdrant-client pymilvus lancedb weaviate-client chromadb pgvector psycopg2-binary
```

## Quick Start

### 1. Download a Dataset
```bash
python -m src.cli download --dataset sift1m --output ./data
```

### 2. Run a Simple Benchmark
```bash
python scripts/run_benchmark.py --database faiss --dataset sift1m --runs 5
```

### 3. Compare Multiple Databases
```bash
python scripts/run_benchmark.py \
    --database faiss qdrant milvus lancedb \
    --dataset sift1m \
    --export json csv latex plots
```

### 4. Run All Enabled Configurations
```bash
python scripts/run_benchmark.py --all
```

## Configuration

### Main Configuration (`config/default.yaml`)

```yaml
# Select active database and dataset
database:
  active: "faiss"
  compare_all: false

dataset:
  active: "sift1m"
  compare_all: false

# Experiment settings
experiment:
  runs: 5
  warmup_queries: 1000
  measurement_queries: 10000
  seed: 42

# Query settings
query:
  k_values: [1, 10, 50, 100]
  num_queries: 10000
  thread_counts: [1, 4, 8, 16, 32]
```

### Database-Specific Configuration (`config/databases/faiss.yaml`)

```yaml
index_configurations:
  - name: "HNSW32"
    type: "HNSW"
    params:
      M: 32
      efConstruction: 200
    search_params:
      efSearch: [32, 64, 128, 256, 512]
```

## Output Formats

### JSON Results
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "num_results": 4
  },
  "results": [
    {
      "experiment_name": "faiss_sift1m",
      "mean_metrics": {
        "quality_recall@10": 0.9847,
        "perf_latency_p50_ms": 0.42,
        "resource_index_build_time_sec": 12.5
      }
    }
  ]
}
```

### LaTeX Tables
```latex
\begin{table}[htbp]
\centering
\caption{Quality Metrics Comparison}
\begin{tabular}{lcccc}
\toprule
Database & Recall@10 & Recall@100 & MRR & NDCG@10 \\
\midrule
FAISS & \textbf{0.985} & 0.998 & 0.923 & 0.967 \\
Qdrant & 0.982 & \textbf{0.999} & \textbf{0.925} & \textbf{0.968} \\
\bottomrule
\end{tabular}
\end{table}
```

### Visualization Plots
- Recall comparison bar charts
- Latency comparison (p50, p90, p99)
- Recall vs Latency tradeoff scatter plots
- QPS scaling with thread count
- Build time comparison

## Project Structure

```
RAGdbEval/
├── config/
│   ├── default.yaml              # Main configuration
│   ├── databases/                # Database configs
│   │   ├── faiss.yaml
│   │   ├── qdrant.yaml
│   │   ├── milvus.yaml
│   │   ├── lancedb.yaml
│   │   ├── weaviate.yaml
│   │   ├── chroma.yaml
│   │   └── pgvector.yaml
│   └── datasets/                 # Dataset configs
│       ├── sift1m.yaml
│       ├── deep1m.yaml
│       ├── msmarco.yaml
│       ├── glove.yaml
│       ├── gist1m.yaml
│       └── random.yaml
├── src/
│   ├── core/                     # Base classes and types
│   │   ├── base.py               # VectorDBInterface
│   │   ├── types.py              # Data types
│   │   └── config.py             # Configuration
│   ├── databases/                # Database adapters
│   ├── datasets/                 # Dataset loaders
│   ├── metrics/                  # Metric computation
│   │   ├── quality.py            # Recall, Precision, MRR, NDCG
│   │   ├── performance.py        # Latency, QPS
│   │   └── resource.py           # Memory, CPU
│   ├── benchmark/                # Benchmark runner
│   └── reporting/                # Export and visualization
├── scripts/
│   └── run_benchmark.py          # Main entry point
├── tests/                        # Test suite
├── results/                      # Output directory
└── data/                         # Downloaded datasets
```

## Methodology

This framework follows established benchmarking methodologies:

1. **ANN-Benchmarks** - Standard for algorithm-level comparison
2. **VectorDBBench** - Production-realistic database benchmarking
3. **Big-ANN-Benchmarks** - NeurIPS competition methodology

### Key Principles

- **Reproducibility**: Fixed random seeds, documented configurations
- **Fair Comparison**: Same hardware, memory limits, query sets
- **Statistical Rigor**: Multiple runs, confidence intervals
- **Recall-Latency Tradeoff**: Parameter tuning for target recall

## Adding New Components

### Adding a New Database

1. Create adapter in `src/databases/newdb_adapter.py`
2. Inherit from `VectorDBInterface`
3. Implement required methods
4. Register with `@register_database("newdb")`
5. Create config in `config/databases/newdb.yaml`

### Adding a New Dataset

1. Create loader in `src/datasets/newdata.py`
2. Inherit from `DatasetLoader`
3. Implement `load_vectors`, `load_queries`, `load_ground_truth`
4. Create config in `config/datasets/newdata.yaml`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vectordb_bench,
  title={VectorDB-Bench: Comprehensive Vector Database Benchmarking Framework},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/vectordb-bench}
}
```

## References

- [ANN-Benchmarks](https://ann-benchmarks.com/)
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench)
- [Big-ANN-Benchmarks](https://big-ann-benchmarks.com/)
- [MS MARCO](https://microsoft.github.io/msmarco/)
- [SIFT1M/GIST1M](http://corpus-texmex.irisa.fr/)

## License

MIT License - see [LICENSE](LICENSE) for details.

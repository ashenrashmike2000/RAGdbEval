"""Tests for metrics computation."""

import numpy as np
import pytest

from src.metrics.quality import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_map_at_k,
    compute_hit_rate_at_k,
    compute_f1_at_k,
)


class TestQualityMetrics:
    """Test quality metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        # Perfect retrieval
        self.perfect_retrieved = np.array([[0, 1, 2, 3, 4]])
        self.perfect_gt = np.array([[0, 1, 2, 3, 4]])

        # Partial retrieval
        self.partial_retrieved = np.array([[0, 5, 2, 6, 4]])
        self.partial_gt = np.array([[0, 1, 2, 3, 4]])

        # No overlap
        self.no_overlap_retrieved = np.array([[5, 6, 7, 8, 9]])
        self.no_overlap_gt = np.array([[0, 1, 2, 3, 4]])

    def test_recall_perfect(self):
        """Test recall with perfect retrieval."""
        recall = compute_recall_at_k(self.perfect_retrieved, self.perfect_gt, k=5)
        assert recall == 1.0

    def test_recall_partial(self):
        """Test recall with partial overlap."""
        recall = compute_recall_at_k(self.partial_retrieved, self.partial_gt, k=5)
        assert recall == 0.6  # 3 out of 5

    def test_recall_no_overlap(self):
        """Test recall with no overlap."""
        recall = compute_recall_at_k(self.no_overlap_retrieved, self.no_overlap_gt, k=5)
        assert recall == 0.0

    def test_precision_perfect(self):
        """Test precision with perfect retrieval."""
        precision = compute_precision_at_k(self.perfect_retrieved, self.perfect_gt, k=5)
        assert precision == 1.0

    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        mrr = compute_mrr(self.perfect_retrieved, self.perfect_gt)
        assert mrr == 1.0

    def test_mrr_later_position(self):
        """Test MRR when relevant result is not first."""
        retrieved = np.array([[5, 0, 2, 3, 4]])
        gt = np.array([[0, 1, 2, 3, 4]])
        mrr = compute_mrr(retrieved, gt)
        assert mrr == 0.5  # First relevant at position 2

    def test_hit_rate(self):
        """Test hit rate computation."""
        hit_rate = compute_hit_rate_at_k(self.partial_retrieved, self.partial_gt, k=5)
        assert hit_rate == 1.0  # At least one relevant found

    def test_ndcg_perfect(self):
        """Test NDCG with perfect retrieval."""
        ndcg = compute_ndcg_at_k(self.perfect_retrieved, self.perfect_gt, k=5)
        assert ndcg == 1.0


class TestPerformanceMetrics:
    """Test performance metrics."""

    def test_latency_percentiles(self):
        """Test latency percentile computation."""
        from src.metrics.performance import compute_latency_percentiles

        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        percentiles = compute_latency_percentiles(latencies)

        assert percentiles["p50"] == 5.5
        assert percentiles["mean"] == 5.5
        assert percentiles["min"] == 1.0
        assert percentiles["max"] == 10.0

    def test_qps_computation(self):
        """Test QPS computation."""
        from src.metrics.performance import compute_qps

        # 10 queries, each taking 1ms = 1000 QPS
        latencies = [1.0] * 10
        qps = compute_qps(latencies, num_threads=1)
        assert qps == 1000.0


class TestResourceMetrics:
    """Test resource metrics."""

    def test_memory_measurement(self):
        """Test memory measurement."""
        from src.metrics.resource import measure_memory_usage

        mem = measure_memory_usage()
        assert "rss" in mem
        assert mem["rss"] > 0

    def test_resource_monitor(self):
        """Test ResourceMonitor context manager."""
        from src.metrics.resource import ResourceMonitor
        import numpy as np

        with ResourceMonitor() as monitor:
            # Allocate some memory
            data = np.zeros((1000, 1000), dtype=np.float32)
            del data

        assert monitor.elapsed_sec > 0
        assert monitor.peak_memory_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

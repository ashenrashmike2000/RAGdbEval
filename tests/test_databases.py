"""Tests for database adapters."""

import numpy as np
import pytest

from src.core.types import DistanceMetric, IndexConfig


class TestFAISSAdapter:
    """Test FAISS adapter."""

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 128).astype(np.float32)

    @pytest.fixture
    def sample_queries(self):
        """Create sample queries."""
        np.random.seed(43)
        return np.random.randn(10, 128).astype(np.float32)

    def test_faiss_flat_index(self, sample_vectors, sample_queries):
        """Test FAISS flat index creation and search."""
        try:
            from src.databases.faiss_adapter import FAISSAdapter
        except ImportError:
            pytest.skip("FAISS not installed")

        config = {}
        adapter = FAISSAdapter(config)

        index_config = IndexConfig(
            name="Flat",
            type="Flat",
            description="Exact search",
            params={},
        )

        adapter.connect()

        # Create index
        build_time = adapter.create_index(
            sample_vectors,
            index_config,
            DistanceMetric.L2,
        )

        assert build_time > 0
        assert adapter.num_vectors == 1000

        # Search
        indices, distances, latencies = adapter.search(sample_queries, k=10)

        assert indices.shape == (10, 10)
        assert distances.shape == (10, 10)
        assert len(latencies) == 10

        # Verify exact search returns query itself if in index
        # (For flat index, this should be exact)

        adapter.disconnect()

    def test_faiss_ivf_index(self, sample_vectors, sample_queries):
        """Test FAISS IVF index."""
        try:
            from src.databases.faiss_adapter import FAISSAdapter
        except ImportError:
            pytest.skip("FAISS not installed")

        config = {"training": {"train_size_ratio": 1.0}}
        adapter = FAISSAdapter(config)

        index_config = IndexConfig(
            name="IVF100",
            type="IVF",
            description="IVF index",
            params={"nlist": 100},
            search_params={"nprobe": 10},
        )

        adapter.connect()
        adapter.create_index(sample_vectors, index_config, DistanceMetric.L2)

        indices, distances, latencies = adapter.search(
            sample_queries, k=10, search_params={"nprobe": 10}
        )

        assert indices.shape == (10, 10)
        assert all(lat > 0 for lat in latencies)

        adapter.disconnect()


class TestDatabaseFactory:
    """Test database factory."""

    def test_list_databases(self):
        """Test listing available databases."""
        from src.databases import list_available_databases

        databases = list_available_databases()
        assert isinstance(databases, list)
        # FAISS should always be available if installed
        # assert "faiss" in databases

    def test_get_unknown_database(self):
        """Test error on unknown database."""
        from src.databases import get_database

        with pytest.raises(ValueError):
            get_database("unknown_database_xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

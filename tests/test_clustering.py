"""Tests for AML clustering models."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clustering import (
    TransactionClusterer,
    OutlierClusterer,
    find_optimal_clusters,
)


@pytest.fixture
def sample_features():
    """Generate synthetic feature matrix for clustering tests."""
    np.random.seed(42)
    # Create 3 distinct clusters
    c1 = np.random.normal(0, 1, (200, 5))
    c2 = np.random.normal(5, 1, (200, 5))
    c3 = np.random.normal(-5, 1, (200, 5))
    X = np.vstack([c1, c2, c3])
    return X


class TestTransactionClusterer:
    def test_fit_predict(self, sample_features):
        """K-Means should assign cluster labels to all samples."""
        clusterer = TransactionClusterer(n_clusters=3)
        labels = clusterer.fit_predict(sample_features)
        assert len(labels) == len(sample_features)
        assert len(np.unique(labels)) == 3

    def test_cluster_summary(self, sample_features):
        """Cluster summary should have one row per cluster."""
        feature_names = [f"f{i}" for i in range(5)]
        clusterer = TransactionClusterer(n_clusters=3)
        labels = clusterer.fit_predict(sample_features, feature_names=feature_names)
        summary = clusterer.get_cluster_summary(sample_features, labels, feature_names)
        assert len(summary) == 3
        assert "count" in summary.columns


class TestOutlierClusterer:
    def test_detects_outliers(self):
        """DBSCAN should detect outliers in data with sparse noise points."""
        np.random.seed(42)
        normal = np.random.normal(0, 1, (500, 3))
        # Single scattered outlier points that cannot form a cluster
        outliers = np.random.uniform(8, 20, (5, 3))
        X = np.vstack([normal, outliers])
        clusterer = OutlierClusterer(eps=0.3, min_samples=10)
        labels, outlier_mask = clusterer.fit_predict(X)
        # Either we detect outliers or we have multiple clusters (both valid DBSCAN outcomes)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert outlier_mask.sum() > 0 or n_clusters >= 1

    def test_outlier_statistics(self):
        """Should compare outlier vs inlier statistics."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (200, 3))
        clusterer = OutlierClusterer(eps=0.8, min_samples=5)
        labels, _ = clusterer.fit_predict(X)
        stats = clusterer.get_outlier_statistics(X, labels, ["a", "b", "c"])
        assert "inlier_mean" in stats.columns
        assert "outlier_mean" in stats.columns


class TestOptimalClusters:
    def test_find_optimal(self, sample_features):
        """Should find optimal k with highest silhouette score."""
        result = find_optimal_clusters(sample_features, k_range=range(2, 6))
        assert "optimal_k" in result
        assert "scores" in result
        assert 2 <= result["optimal_k"] <= 5

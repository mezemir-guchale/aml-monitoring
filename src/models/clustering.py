"""Clustering models for AML transaction behavior segmentation.

Uses K-Means for behavior segmentation and DBSCAN for outlier detection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TransactionClusterer:
    """Segment transaction behaviors using K-Means clustering."""

    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self.scaler = StandardScaler()
        self.cluster_profiles = None

    def fit_predict(self, X, feature_names=None):
        """Fit K-Means and assign cluster labels.

        Args:
            X: Feature matrix.
            feature_names: Feature names for profiling.

        Returns:
            Array of cluster labels.
        """
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)

        # Compute silhouette score on a sample for efficiency
        sample_size = min(10000, len(X_scaled))
        idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        sil_score = silhouette_score(X_scaled[idx], labels[idx])
        logger.info(f"K-Means ({self.n_clusters} clusters): silhouette={sil_score:.4f}")

        # Profile each cluster
        if feature_names is not None:
            self._profile_clusters(X, labels, feature_names)

        return labels

    def _profile_clusters(self, X, labels, feature_names):
        """Create profiles for each cluster.

        Args:
            X: Feature matrix.
            labels: Cluster labels.
            feature_names: Feature names.
        """
        df = pd.DataFrame(X, columns=feature_names)
        df["cluster"] = labels

        self.cluster_profiles = df.groupby("cluster").agg(["mean", "count"]).round(4)
        cluster_sizes = df["cluster"].value_counts().sort_index()
        logger.info(f"Cluster sizes:\n{cluster_sizes.to_string()}")

    def get_cluster_summary(self, X, labels, feature_names):
        """Get summary statistics per cluster.

        Args:
            X: Feature matrix.
            labels: Cluster labels.
            feature_names: Feature column names.

        Returns:
            DataFrame with mean feature values per cluster.
        """
        df = pd.DataFrame(X, columns=feature_names)
        df["cluster"] = labels
        summary = df.groupby("cluster").mean().round(4)
        summary["count"] = df.groupby("cluster").size()
        return summary


class OutlierClusterer:
    """Use DBSCAN to find outlier clusters in transaction data."""

    def __init__(self, eps=0.5, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

    def fit_predict(self, X):
        """Fit DBSCAN and return labels.

        Points labeled -1 are outliers (noise).

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (labels array, outlier_mask boolean array).
        """
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = (labels == -1).sum()
        outlier_pct = n_outliers / len(labels) if len(labels) > 0 else 0

        logger.info(
            f"DBSCAN: {n_clusters} clusters, {n_outliers} outliers ({outlier_pct:.2%})"
        )

        return labels, labels == -1

    def get_outlier_statistics(self, X, labels, feature_names=None):
        """Compare outlier vs non-outlier statistics.

        Args:
            X: Feature matrix.
            labels: DBSCAN labels.
            feature_names: Feature names.

        Returns:
            DataFrame comparing outliers vs inliers.
        """
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=names)
        df["is_outlier"] = (labels == -1).astype(int)

        comparison = df.groupby("is_outlier").mean().T
        comparison.columns = ["inlier_mean", "outlier_mean"]
        comparison["ratio"] = np.where(
            comparison["inlier_mean"] != 0,
            comparison["outlier_mean"] / comparison["inlier_mean"],
            0,
        )
        return comparison.round(4)


def find_optimal_clusters(X, k_range=None, random_state=42):
    """Find optimal number of clusters using silhouette score.

    Args:
        X: Feature matrix.
        k_range: Range of k values to test.
        random_state: Random seed.

    Returns:
        Dict with optimal_k and silhouette scores.
    """
    if k_range is None:
        k_range = range(3, 11)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample for speed
    sample_size = min(10000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        scores[k] = score
        logger.info(f"  k={k}: silhouette={score:.4f}")

    optimal_k = max(scores, key=scores.get)
    logger.info(f"Optimal k={optimal_k} (silhouette={scores[optimal_k]:.4f})")

    return {
        "optimal_k": optimal_k,
        "scores": scores,
    }

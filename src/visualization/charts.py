"""Visualization charts for AML transaction monitoring."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def plot_anomaly_scatter(df, score_col="ensemble_score", output_path=None):
    """Scatter plot of transactions colored by anomaly score.

    Args:
        df: DataFrame with amount, score, and suspicious columns.
        score_col: Column name of the anomaly score.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sample = df.sample(min(10000, len(df)), random_state=42)

    scatter = ax.scatter(
        sample.index,
        sample["amount"],
        c=sample[score_col],
        cmap="RdYlGn_r",
        s=3,
        alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Anomaly Score")

    # Highlight known suspicious
    sus = sample[sample["is_suspicious"] == 1]
    ax.scatter(sus.index, sus["amount"], c="red", s=15, alpha=0.8, marker="x", label="Known Suspicious")

    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Amount ($)")
    ax.set_title("Transaction Anomaly Scores")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Anomaly scatter plot saved to {output_path}")
    plt.close(fig)


def plot_cluster_visualization(X, labels, feature_names=None, output_path=None):
    """Visualize clusters using first two principal components.

    Args:
        X: Feature matrix.
        labels: Cluster labels.
        feature_names: Feature names.
        output_path: Path to save figure.
    """
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(10, 8))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X if not hasattr(X, 'values') else X.values)

    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = f"Cluster {label}" if label != -1 else "Outliers"
        marker = "o" if label != -1 else "x"
        size = 5 if label != -1 else 20
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[i]], s=size, alpha=0.5, label=name, marker=marker,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Transaction Clusters (PCA Projection)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Cluster visualization saved to {output_path}")
    plt.close(fig)


def plot_alert_timeline(df, score_col="ensemble_score", threshold=0.5, output_path=None):
    """Plot alert volume over time.

    Args:
        df: DataFrame with timestamp and scores.
        score_col: Score column name.
        threshold: Alert threshold.
        output_path: Path to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    temp = df.copy()
    temp["timestamp"] = pd.to_datetime(temp["timestamp"])
    temp["date"] = temp["timestamp"].dt.date
    temp["is_alert"] = (temp[score_col] >= threshold).astype(int)

    # Daily alert volume
    daily_alerts = temp.groupby("date")["is_alert"].sum()
    ax1.bar(daily_alerts.index, daily_alerts.values, color="coral", alpha=0.7, width=1)
    ax1.set_ylabel("Number of Alerts")
    ax1.set_title("Daily Alert Volume")
    ax1.grid(True, alpha=0.3)

    # Daily transaction volume
    daily_txns = temp.groupby("date")["amount"].count()
    ax2.bar(daily_txns.index, daily_txns.values, color="steelblue", alpha=0.7, width=1)
    ax2.set_ylabel("Total Transactions")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Alert timeline saved to {output_path}")
    plt.close(fig)


def plot_suspicious_pattern_heatmap(df, output_path=None):
    """Heatmap of suspicious patterns by hour and day of week.

    Args:
        df: DataFrame with suspicious transactions.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    suspicious = df[df["is_suspicious"] == 1].copy()
    if len(suspicious) == 0:
        logger.warning("No suspicious transactions for heatmap.")
        plt.close(fig)
        return

    suspicious["timestamp"] = pd.to_datetime(suspicious["timestamp"])
    suspicious["hour"] = suspicious["timestamp"].dt.hour
    suspicious["day_of_week"] = suspicious["timestamp"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = suspicious.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Suspicious Transaction Heatmap")
    plt.colorbar(im, ax=ax, label="Count")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Suspicious pattern heatmap saved to {output_path}")
    plt.close(fig)


def plot_detection_by_pattern(pattern_stats, output_path=None):
    """Bar chart of detection rates by suspicious pattern type.

    Args:
        pattern_stats: DataFrame from evaluate_by_pattern().
        output_path: Path to save figure.
    """
    if pattern_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    patterns = pattern_stats.index.tolist()
    rates = pattern_stats["detection_rate"].values
    colors = ["#e74c3c" if r < 0.5 else "#f39c12" if r < 0.8 else "#27ae60" for r in rates]

    bars = ax.bar(patterns, rates, color=colors, edgecolor="gray")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Suspicious Pattern Type")
    ax.set_ylim(0, 1.1)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=10)

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Detection by pattern chart saved to {output_path}")
    plt.close(fig)


def generate_all_charts(df, cluster_X=None, cluster_labels=None, pattern_stats=None, output_dir=None):
    """Generate all AML monitoring charts.

    Args:
        df: DataFrame with scores and features.
        cluster_X: Feature matrix for clustering visualization.
        cluster_labels: Cluster labels.
        pattern_stats: Pattern detection statistics.
        output_dir: Output directory.
    """
    if output_dir is None:
        output_dir = "reports/figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "ensemble_score" in df.columns:
        plot_anomaly_scatter(df, output_path=f"{output_dir}/anomaly_scatter.png")
        plot_alert_timeline(df, output_path=f"{output_dir}/alert_timeline.png")

    plot_suspicious_pattern_heatmap(df, output_path=f"{output_dir}/suspicious_heatmap.png")

    if cluster_X is not None and cluster_labels is not None:
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(cluster_X)
        plot_cluster_visualization(X_scaled, cluster_labels, output_path=f"{output_dir}/clusters.png")

    if pattern_stats is not None:
        plot_detection_by_pattern(pattern_stats, output_path=f"{output_dir}/detection_by_pattern.png")

    logger.info(f"All charts saved to {output_dir}")

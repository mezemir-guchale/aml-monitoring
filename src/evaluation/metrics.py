"""Evaluation metrics for AML transaction monitoring."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_detection(y_true, y_pred, y_scores=None):
    """Evaluate anomaly detection performance against known suspicious labels.

    Args:
        y_true: True suspicious labels (1=suspicious, 0=normal).
        y_pred: Predicted alert labels.
        y_scores: Continuous anomaly scores (optional).

    Returns:
        Dict with precision, recall, F1, and additional metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_alerts": int(y_pred.sum()),
        "total_suspicious": int(y_true.sum()),
        "alert_rate": float(y_pred.mean()),
    }

    logger.info(
        f"Detection: Precision={precision:.3f}, Recall={recall:.3f}, "
        f"F1={f1:.3f}, Alerts={results['total_alerts']}"
    )

    return results


def alert_volume_analysis(y_scores, thresholds=None):
    """Analyze alert volumes at different score thresholds.

    Helps calibrate the alerting threshold for operational feasibility.

    Args:
        y_scores: Continuous anomaly scores.
        thresholds: List of thresholds to evaluate.

    Returns:
        DataFrame with alert counts at each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    scores = np.asarray(y_scores)
    n_total = len(scores)

    rows = []
    for t in thresholds:
        n_alerts = (scores >= t).sum()
        rows.append({
            "threshold": t,
            "n_alerts": int(n_alerts),
            "alert_rate": float(n_alerts / n_total),
            "daily_alerts_est": float(n_alerts / 365),  # rough estimate
        })

    result = pd.DataFrame(rows)
    logger.info(f"Alert volume analysis:\n{result.to_string(index=False)}")
    return result


def evaluate_by_pattern(df, score_col="ensemble_score", threshold=0.5):
    """Evaluate detection performance by suspicious pattern type.

    Args:
        df: DataFrame with pattern_type, is_suspicious, and score columns.
        score_col: Column name of the anomaly score.
        threshold: Alert threshold.

    Returns:
        DataFrame with per-pattern detection rates.
    """
    suspicious = df[df["is_suspicious"] == 1].copy()

    if len(suspicious) == 0:
        logger.warning("No suspicious transactions found for pattern analysis.")
        return pd.DataFrame()

    suspicious["detected"] = (suspicious[score_col] >= threshold).astype(int)

    pattern_stats = suspicious.groupby("pattern_type").agg(
        total=("detected", "count"),
        detected=("detected", "sum"),
        avg_score=(score_col, "mean"),
    )
    pattern_stats["detection_rate"] = (pattern_stats["detected"] / pattern_stats["total"]).round(3)
    pattern_stats = pattern_stats.sort_values("detection_rate", ascending=False)

    logger.info(f"Detection by pattern:\n{pattern_stats.to_string()}")
    return pattern_stats


def compute_sar_metrics(n_alerts, n_true_suspicious, n_investigated, n_filed):
    """Compute Suspicious Activity Report (SAR) filing metrics.

    Args:
        n_alerts: Total alerts generated.
        n_true_suspicious: Known suspicious transactions.
        n_investigated: Alerts investigated by compliance.
        n_filed: SARs actually filed.

    Returns:
        Dict with SAR-related metrics.
    """
    metrics = {
        "total_alerts": n_alerts,
        "investigation_rate": n_investigated / n_alerts if n_alerts > 0 else 0,
        "sar_filing_rate": n_filed / n_investigated if n_investigated > 0 else 0,
        "alert_to_sar_ratio": n_alerts / n_filed if n_filed > 0 else float("inf"),
        "coverage": n_filed / n_true_suspicious if n_true_suspicious > 0 else 0,
    }

    logger.info(f"SAR Metrics: Alert-to-SAR ratio={metrics['alert_to_sar_ratio']:.1f}")
    return metrics

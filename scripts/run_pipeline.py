#!/usr/bin/env python3
"""End-to-end AML transaction monitoring pipeline."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.generate_dataset import generate_transactions
from src.features.engineering import engineer_all_features, get_feature_columns
from src.models.detector import (
    IsolationForestDetector,
    LOFDetector,
    RuleBasedDetector,
    EnsembleDetector,
)
from src.models.clustering import TransactionClusterer, OutlierClusterer
from src.evaluation.metrics import (
    evaluate_detection,
    alert_volume_analysis,
    evaluate_by_pattern,
)
from src.visualization.charts import generate_all_charts


def main():
    config = Config(os.path.join(project_root, "configs", "config.yaml"))
    logger = setup_logger("pipeline", log_file=config.get_path("logging", "log_file"))

    logger.info("=" * 60)
    logger.info("AML TRANSACTION MONITORING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Generate data
    logger.info("\n--- Step 1: Generate Transaction Data ---")
    n_txns = config.get("data", "n_transactions", default=200000)
    seed = config.get("data", "random_state", default=42)
    df, accounts = generate_transactions(n_transactions=n_txns, random_state=seed)

    raw_path = config.get_path("data", "raw_path")
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    accounts.to_csv(raw_path.replace("transactions.csv", "accounts.csv"), index=False)

    # Step 2: Feature engineering
    logger.info("\n--- Step 2: Feature Engineering ---")
    df_feat = engineer_all_features(df)

    processed_path = config.get_path("data", "processed_path")
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_columns(df_feat)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    X = df_feat[feature_cols].values

    # Step 3: Anomaly Detection
    logger.info("\n--- Step 3: Anomaly Detection ---")

    # Isolation Forest
    logger.info("Training Isolation Forest...")
    if_params = config.get("models", "isolation_forest", default={})
    if_detector = IsolationForestDetector(**if_params)
    if_detector.fit(X)
    if_scores = if_detector.score_samples(X)
    logger.info(f"  IF mean score: {if_scores.mean():.4f}")

    # LOF (on a sample for speed)
    logger.info("Running Local Outlier Factor...")
    sample_size = min(50000, len(X))
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    lof_params = config.get("models", "lof", default={})
    lof_detector = LOFDetector(**lof_params)
    _, lof_sample_scores = lof_detector.fit_predict(X[sample_idx])

    # Expand LOF scores to full dataset (use IF for non-sampled)
    lof_scores = np.full(len(X), 0.0)
    lof_scores[sample_idx] = lof_sample_scores

    # Rule-based detection
    logger.info("Running rule-based detection...")
    rule_detector = RuleBasedDetector()
    rule_scores, rule_details = rule_detector.detect(df_feat)

    # Ensemble
    logger.info("Computing ensemble scores...")
    ensemble_weights = config.get("models", "ensemble", "weights", default={})
    threshold = config.get("models", "threshold", default=0.5)
    ensemble = EnsembleDetector(weights=ensemble_weights, threshold=threshold)
    ensemble_result = ensemble.combine_scores({
        "isolation_forest": if_scores,
        "lof": lof_scores,
        "rules": rule_scores,
    })

    df_feat["ensemble_score"] = ensemble_result["ensemble_score"]
    df_feat["is_alert"] = ensemble_result["is_alert"].astype(int)
    df_feat["if_score"] = if_scores
    df_feat["rule_score"] = rule_scores

    # Step 4: Clustering
    logger.info("\n--- Step 4: Transaction Clustering ---")

    # K-Means
    km_params = config.get("models", "kmeans", default={})
    clusterer = TransactionClusterer(**km_params)
    cluster_labels = clusterer.fit_predict(X, feature_names=feature_cols)
    df_feat["cluster"] = cluster_labels

    # DBSCAN on a sample
    logger.info("Running DBSCAN outlier clustering...")
    dbscan_params = config.get("models", "dbscan", default={})
    outlier_clusterer = OutlierClusterer(**dbscan_params)
    dbscan_labels, dbscan_outliers = outlier_clusterer.fit_predict(X[sample_idx])

    # Step 5: Evaluation
    logger.info("\n--- Step 5: Evaluation ---")
    y_true = df_feat["is_suspicious"].values
    y_pred = df_feat["is_alert"].values

    detection_metrics = evaluate_detection(y_true, y_pred, df_feat["ensemble_score"].values)
    volume_analysis = alert_volume_analysis(df_feat["ensemble_score"].values)
    pattern_stats = evaluate_by_pattern(df_feat)

    # Step 6: Visualization
    logger.info("\n--- Step 6: Generate Visualizations ---")
    output_dir = config.get_path("visualization", "output_dir")
    generate_all_charts(
        df_feat,
        cluster_X=X[sample_idx],
        cluster_labels=cluster_labels[sample_idx],
        pattern_stats=pattern_stats,
        output_dir=output_dir,
    )

    # Save summary report
    reports_path = config.get_path("evaluation", "reports_path")
    Path(reports_path).mkdir(parents=True, exist_ok=True)

    summary = {
        "total_transactions": len(df),
        "suspicious_transactions": int(df["is_suspicious"].sum()),
        "suspicious_rate": float(df["is_suspicious"].mean()),
        "detection_metrics": detection_metrics,
        "alert_volume": volume_analysis.to_dict("records"),
        "pattern_detection": pattern_stats.to_dict() if not pattern_stats.empty else {},
        "n_clusters": int(clusterer.n_clusters),
    }

    with open(os.path.join(reports_path, "aml_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save processed features
    df_feat.to_csv(processed_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Precision: {detection_metrics['precision']:.3f}")
    logger.info(f"  Recall: {detection_metrics['recall']:.3f}")
    logger.info(f"  F1: {detection_metrics['f1_score']:.3f}")
    logger.info(f"  Total Alerts: {detection_metrics['total_alerts']}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    main()

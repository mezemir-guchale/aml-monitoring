"""Tests for AML anomaly detectors."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_dataset import generate_transactions
from src.features.engineering import engineer_all_features, get_feature_columns
from src.models.detector import (
    IsolationForestDetector,
    LOFDetector,
    RuleBasedDetector,
    EnsembleDetector,
)
from src.evaluation.metrics import evaluate_detection, alert_volume_analysis


@pytest.fixture(scope="module")
def feature_data():
    """Generate and engineer features for detector testing."""
    df, _ = generate_transactions(n_transactions=3000, n_accounts=200, random_state=42)
    df_feat = engineer_all_features(df)
    feature_cols = get_feature_columns(df_feat)
    X = df_feat[feature_cols].values
    return {"df": df_feat, "X": X, "feature_cols": feature_cols}


class TestIsolationForest:
    def test_fit_and_predict(self, feature_data):
        """IF should fit and produce predictions."""
        detector = IsolationForestDetector(n_estimators=50, contamination=0.05)
        detector.fit(feature_data["X"])
        preds = detector.predict(feature_data["X"])
        assert set(np.unique(preds)) <= {-1, 1}

    def test_score_samples_range(self, feature_data):
        """Anomaly scores should be in [0, 1]."""
        detector = IsolationForestDetector(n_estimators=50, contamination=0.05)
        detector.fit(feature_data["X"])
        scores = detector.score_samples(feature_data["X"])
        assert scores.min() >= 0
        assert scores.max() <= 1.001  # small tolerance


class TestLOF:
    def test_fit_predict(self, feature_data):
        """LOF should return labels and scores."""
        detector = LOFDetector(n_neighbors=10, contamination=0.05)
        labels, scores = detector.fit_predict(feature_data["X"][:1000])
        assert len(labels) == 1000
        assert len(scores) == 1000
        assert scores.min() >= 0
        assert scores.max() <= 1.001


class TestRuleBasedDetector:
    def test_detect_returns_scores(self, feature_data):
        """Rule-based detector should return scores array."""
        detector = RuleBasedDetector()
        scores, details = detector.detect(feature_data["df"])
        assert len(scores) == len(feature_data["df"])
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_rule_details_columns(self, feature_data):
        """Should return detail columns for each rule."""
        detector = RuleBasedDetector()
        scores, details = detector.detect(feature_data["df"])
        assert "rule_structuring" in details.columns
        assert "rule_rapid" in details.columns


class TestEnsembleDetector:
    def test_combine_scores(self):
        """Ensemble should combine multiple score arrays."""
        n = 100
        scores = {
            "isolation_forest": np.random.uniform(0, 1, n),
            "lof": np.random.uniform(0, 1, n),
            "rules": np.random.uniform(0, 1, n),
        }
        ensemble = EnsembleDetector(threshold=0.5)
        result = ensemble.combine_scores(scores)
        assert "ensemble_score" in result
        assert "is_alert" in result
        assert len(result["ensemble_score"]) == n

    def test_threshold_effect(self):
        """Higher threshold should produce fewer alerts."""
        n = 1000
        scores = {"rules": np.random.uniform(0, 1, n)}
        low_t = EnsembleDetector(weights={"rules": 1.0}, threshold=0.3)
        high_t = EnsembleDetector(weights={"rules": 1.0}, threshold=0.7)
        r_low = low_t.combine_scores(scores)
        r_high = high_t.combine_scores(scores)
        assert r_low["is_alert"].sum() >= r_high["is_alert"].sum()


class TestMetrics:
    def test_evaluate_detection(self):
        """Detection evaluation should return valid metrics."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        result = evaluate_detection(y_true, y_pred)
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert result["total_alerts"] == y_pred.sum()

    def test_alert_volume_analysis(self):
        """Alert volume should decrease with higher threshold."""
        scores = np.random.uniform(0, 1, 5000)
        result = alert_volume_analysis(scores)
        assert result["n_alerts"].is_monotonic_decreasing

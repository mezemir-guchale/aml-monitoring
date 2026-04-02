"""Anomaly detection models for AML transaction monitoring.

Implements Isolation Forest, Local Outlier Factor, rule-based detection,
and ensemble scoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class IsolationForestDetector:
    """Isolation Forest anomaly detector for transaction monitoring."""

    def __init__(self, n_estimators=200, contamination=0.03, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """Fit the Isolation Forest model.

        Args:
            X: Feature matrix.

        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info(f"Isolation Forest fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def predict(self, X):
        """Predict anomaly labels (-1 for anomaly, 1 for normal).

        Args:
            X: Feature matrix.

        Returns:
            Array of labels.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X):
        """Get anomaly scores (lower = more anomalous).

        Args:
            X: Feature matrix.

        Returns:
            Array of anomaly scores normalized to [0, 1] where 1 = most suspicious.
        """
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.score_samples(X_scaled)
        # Normalize: more negative = more anomalous, flip to 0-1 scale
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros_like(raw_scores)
        return normalized


class LOFDetector:
    """Local Outlier Factor detector for transaction monitoring."""

    def __init__(self, n_neighbors=20, contamination=0.03):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.scaler = StandardScaler()

    def fit_predict(self, X):
        """Fit LOF and predict anomaly labels.

        LOF is transductive so fit and predict happen together.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (labels array, scores array normalized to [0,1]).
        """
        X_scaled = self.scaler.fit_transform(X)
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )
        labels = lof.fit_predict(X_scaled)
        raw_scores = lof.negative_outlier_factor_

        # Normalize scores to [0, 1] where 1 = most suspicious
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros_like(raw_scores)

        logger.info(f"LOF detected {(labels == -1).sum()} anomalies out of {len(labels)}")
        return labels, normalized


class RuleBasedDetector:
    """Rule-based AML detection using domain knowledge."""

    def __init__(self):
        self.rules_triggered = {}

    def detect(self, df):
        """Apply AML rules to transaction data.

        Rules:
        1. Structuring: Amount between $8K-$10K with high frequency
        2. Large cash transactions: Cash deposits/withdrawals > $50K
        3. Rapid transfers: Multiple transfers within short time windows
        4. Night activity: High-value transactions during unusual hours
        5. Round amounts: Suspiciously round large amounts

        Args:
            df: DataFrame with transaction features.

        Returns:
            Tuple of (rule_scores array [0-1], rule_details DataFrame).
        """
        n = len(df)
        scores = np.zeros(n)
        details = pd.DataFrame(index=df.index)

        # Rule 1: Structuring indicator
        if "near_threshold" in df.columns and "txn_count_1d" in df.columns:
            r1 = ((df["near_threshold"] == 1) & (df["txn_count_1d"] >= 3)).astype(float)
        elif "near_threshold" in df.columns:
            r1 = df["near_threshold"].astype(float)
        else:
            r1 = pd.Series(np.zeros(n), index=df.index)
        details["rule_structuring"] = r1
        scores += r1.values * 0.3

        # Rule 2: Large cash transactions
        if "amount" in df.columns and "transaction_type" in df.columns:
            r2 = (
                (df["amount"] > 50000) &
                (df["transaction_type"].isin(["cash_deposit", "cash_withdrawal"]))
            ).astype(float)
        elif "amount" in df.columns:
            r2 = (df["amount"] > 50000).astype(float)
        else:
            r2 = pd.Series(np.zeros(n), index=df.index)
        details["rule_large_cash"] = r2
        scores += r2.values * 0.25

        # Rule 3: Rapid transfers
        if "is_rapid" in df.columns:
            r3 = df["is_rapid"].astype(float)
        elif "time_since_last_txn" in df.columns:
            r3 = (df["time_since_last_txn"] < 300).astype(float)
        else:
            r3 = pd.Series(np.zeros(n), index=df.index)
        details["rule_rapid"] = r3
        scores += r3.values * 0.2

        # Rule 4: Night + high value
        if "is_night" in df.columns and "amount" in df.columns:
            r4 = ((df["is_night"] == 1) & (df["amount"] > 10000)).astype(float)
        else:
            r4 = pd.Series(np.zeros(n), index=df.index)
        details["rule_night_activity"] = r4
        scores += r4.values * 0.15

        # Rule 5: Round large amounts
        if "is_round_amount" in df.columns and "amount" in df.columns:
            r5 = ((df["is_round_amount"] == 1) & (df["amount"] > 5000)).astype(float)
        else:
            r5 = pd.Series(np.zeros(n), index=df.index)
        details["rule_round_amount"] = r5
        scores += r5.values * 0.1

        # Clip to [0, 1]
        scores = np.clip(scores, 0, 1)
        logger.info(f"Rule-based detection: {(scores > 0.3).sum()} alerts triggered")

        return scores, details


class EnsembleDetector:
    """Combine multiple detection methods into an ensemble score."""

    def __init__(self, weights=None, threshold=0.5):
        """Initialize ensemble detector.

        Args:
            weights: Dict mapping detector name to weight.
            threshold: Alert threshold for ensemble score.
        """
        self.weights = weights or {
            "isolation_forest": 0.35,
            "lof": 0.25,
            "rules": 0.40,
        }
        self.threshold = threshold

    def combine_scores(self, score_dict):
        """Combine scores from multiple detectors.

        Args:
            score_dict: Dict mapping detector name to score array.

        Returns:
            Dict with ensemble_score array and alert boolean array.
        """
        total_weight = 0
        ensemble_score = None

        for name, scores in score_dict.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                continue
            if ensemble_score is None:
                ensemble_score = np.zeros_like(scores, dtype=float)
            ensemble_score += scores * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_score /= total_weight

        alerts = ensemble_score >= self.threshold

        logger.info(f"Ensemble: {alerts.sum()} alerts ({alerts.mean():.2%}) at threshold {self.threshold}")

        return {
            "ensemble_score": ensemble_score,
            "is_alert": alerts,
        }

"""Tests for AML feature engineering."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_dataset import generate_transactions
from src.features.engineering import (
    compute_amount_features,
    compute_network_features,
    compute_time_features,
    compute_velocity_features,
    engineer_all_features,
    get_feature_columns,
)


@pytest.fixture(scope="module")
def sample_data():
    """Generate small transaction dataset for testing."""
    df, accounts = generate_transactions(n_transactions=2000, n_accounts=100, random_state=42)
    return df


class TestAmountFeatures:
    def test_creates_zscore(self, sample_data):
        """Should create amount z-score column."""
        result = compute_amount_features(sample_data)
        assert "amount_zscore" in result.columns

    def test_creates_ratio(self, sample_data):
        """Should create amount-to-average ratio."""
        result = compute_amount_features(sample_data)
        assert "amount_to_avg_ratio" in result.columns
        assert (result["amount_to_avg_ratio"] >= 0).all()

    def test_near_threshold_flag(self, sample_data):
        """Should flag amounts near $10K threshold."""
        result = compute_amount_features(sample_data)
        assert "near_threshold" in result.columns
        flagged = result[result["near_threshold"] == 1]
        assert all(flagged["amount"].between(8000, 10000))

    def test_round_amount_flag(self, sample_data):
        """Should flag round amounts."""
        result = compute_amount_features(sample_data)
        assert "is_round_amount" in result.columns


class TestNetworkFeatures:
    def test_unique_receivers(self, sample_data):
        """Should count unique receivers per sender."""
        result = compute_network_features(sample_data)
        assert "n_unique_receivers" in result.columns
        assert (result["n_unique_receivers"] >= 1).all()

    def test_self_transfer_detection(self, sample_data):
        """Should flag self-transfers."""
        result = compute_network_features(sample_data)
        assert "is_self_transfer" in result.columns


class TestTimeFeatures:
    def test_hour_extraction(self, sample_data):
        """Should extract hour from timestamp."""
        result = compute_time_features(sample_data)
        assert "hour" in result.columns
        assert result["hour"].between(0, 23).all()

    def test_weekend_flag(self, sample_data):
        """Should flag weekend transactions."""
        result = compute_time_features(sample_data)
        assert "is_weekend" in result.columns
        assert set(result["is_weekend"].unique()) <= {0, 1}

    def test_rapid_flag(self, sample_data):
        """Should detect rapid consecutive transactions."""
        result = compute_time_features(sample_data)
        assert "is_rapid" in result.columns


class TestFullPipeline:
    def test_engineer_all_features(self, sample_data):
        """Full pipeline should produce valid feature set."""
        result = engineer_all_features(sample_data)
        assert len(result.columns) > len(sample_data.columns)
        feature_cols = get_feature_columns(result)
        assert len(feature_cols) > 10

    def test_no_nan_in_features(self, sample_data):
        """Feature columns should have no NaN."""
        result = engineer_all_features(sample_data)
        feature_cols = get_feature_columns(result)
        for col in feature_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

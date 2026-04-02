"""Feature engineering for AML transaction monitoring.

Computes velocity features, amount statistics, network features, and time patterns.
"""

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_velocity_features(df, windows=None):
    """Compute transaction velocity features per account over time windows.

    Velocity = number of transactions within a rolling window, which can
    indicate structuring or rapid movement patterns.

    Args:
        df: Transaction DataFrame with 'timestamp' and 'sender_account'.
        windows: List of window sizes in days (e.g., [1, 7, 30]).

    Returns:
        DataFrame with velocity features added.
    """
    if windows is None:
        windows = [1, 7, 30]

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values("timestamp")

    for window in windows:
        logger.info(f"Computing {window}-day velocity features...")

        # Count transactions per sender in rolling window
        counts = (
            result.groupby("sender_account")
            .rolling(f"{window}D", on="timestamp")["amount"]
            .count()
            .reset_index(level=0, drop=True)
        )
        result[f"txn_count_{window}d"] = counts.values

        # Sum of amounts in window
        sums = (
            result.groupby("sender_account")
            .rolling(f"{window}D", on="timestamp")["amount"]
            .sum()
            .reset_index(level=0, drop=True)
        )
        result[f"txn_sum_{window}d"] = sums.values

    return result


def compute_amount_features(df):
    """Compute amount-based statistical features per account.

    Detects unusual transaction amounts compared to account history.

    Args:
        df: Transaction DataFrame.

    Returns:
        DataFrame with amount features added.
    """
    result = df.copy()

    # Per-account statistics
    account_stats = result.groupby("sender_account")["amount"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).rename(columns={
        "mean": "acct_avg_amount",
        "std": "acct_std_amount",
        "median": "acct_median_amount",
        "min": "acct_min_amount",
        "max": "acct_max_amount",
        "count": "acct_total_txns",
    })

    result = result.merge(account_stats, left_on="sender_account", right_index=True, how="left")

    # Z-score of amount relative to account history
    result["amount_zscore"] = np.where(
        result["acct_std_amount"] > 0,
        (result["amount"] - result["acct_avg_amount"]) / result["acct_std_amount"],
        0,
    )

    # Ratio to account average
    result["amount_to_avg_ratio"] = np.where(
        result["acct_avg_amount"] > 0,
        result["amount"] / result["acct_avg_amount"],
        0,
    )

    # Is round amount (potential structuring indicator)
    result["is_round_amount"] = (result["amount"] % 1000 == 0).astype(int)

    # Near threshold (just below $10K)
    result["near_threshold"] = (
        (result["amount"] >= 8000) & (result["amount"] < 10000)
    ).astype(int)

    return result


def compute_network_features(df):
    """Compute network/graph-based features.

    Measures account connectivity and transaction patterns between entities.

    Args:
        df: Transaction DataFrame.

    Returns:
        DataFrame with network features added.
    """
    result = df.copy()

    # Unique counterparties per sender
    sender_counterparties = (
        result.groupby("sender_account")["receiver_account"]
        .nunique()
        .rename("n_unique_receivers")
    )
    result = result.merge(sender_counterparties, left_on="sender_account", right_index=True, how="left")

    # Unique senders per receiver
    receiver_counterparties = (
        result.groupby("receiver_account")["sender_account"]
        .nunique()
        .rename("n_unique_senders")
    )
    result = result.merge(receiver_counterparties, left_on="receiver_account", right_index=True, how="left")

    # Bidirectional flag: does this pair transact in both directions?
    pair_key = result.apply(
        lambda r: tuple(sorted([r["sender_account"], r["receiver_account"]])), axis=1
    )
    pair_counts = pair_key.value_counts()
    result["pair_txn_count"] = pair_key.map(pair_counts).fillna(0).astype(int)

    # Self-loop detection
    result["is_self_transfer"] = (
        result["sender_account"] == result["receiver_account"]
    ).astype(int)

    return result


def compute_time_features(df):
    """Compute time-based pattern features.

    Detects unusual timing patterns like late-night transactions
    or weekend activity.

    Args:
        df: Transaction DataFrame.

    Returns:
        DataFrame with time features added.
    """
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_night"] = ((result["hour"] < 6) | (result["hour"] >= 22)).astype(int)
    result["month"] = result["timestamp"].dt.month

    # Time since last transaction per account
    result = result.sort_values(["sender_account", "timestamp"])
    result["time_since_last_txn"] = (
        result.groupby("sender_account")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    # Flag rapid transactions (less than 5 minutes apart)
    result["is_rapid"] = (
        (result["time_since_last_txn"] > 0) & (result["time_since_last_txn"] < 300)
    ).astype(int)

    return result


def engineer_all_features(df):
    """Run the complete feature engineering pipeline.

    Args:
        df: Raw transaction DataFrame.

    Returns:
        DataFrame with all engineered features.
    """
    logger.info("Starting AML feature engineering pipeline...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time features first (needed for velocity)
    df_feat = compute_time_features(df)
    logger.info(f"  Time features: {len(df_feat.columns)} columns")

    # Amount features
    df_feat = compute_amount_features(df_feat)
    logger.info(f"  Amount features: {len(df_feat.columns)} columns")

    # Network features
    df_feat = compute_network_features(df_feat)
    logger.info(f"  Network features: {len(df_feat.columns)} columns")

    # Velocity features (can be slow on large datasets)
    df_feat = compute_velocity_features(df_feat, windows=[1, 7, 30])
    logger.info(f"  Velocity features: {len(df_feat.columns)} columns")

    # Fill NaN
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)

    logger.info(f"Feature engineering complete: {len(df_feat.columns)} total columns")
    return df_feat


def get_feature_columns(df):
    """Get the list of numeric feature columns for modeling.

    Args:
        df: Engineered feature DataFrame.

    Returns:
        List of feature column names.
    """
    exclude = [
        "transaction_id", "timestamp", "sender_account", "receiver_account",
        "transaction_type", "currency", "is_suspicious", "pattern_type",
    ]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]

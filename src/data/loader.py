"""Data loading utilities for AML Monitoring project."""

import pandas as pd
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_transactions(path=None):
    """Load transaction data from CSV.

    Args:
        path: Optional path override.

    Returns:
        pd.DataFrame with transaction data.
    """
    if path is None:
        config = Config()
        path = config.get_path("data", "raw_path")

    if not Path(path).exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} transactions from {path}")
    return df


def load_accounts(path=None):
    """Load account data from CSV.

    Args:
        path: Optional path override.

    Returns:
        pd.DataFrame with account data.
    """
    if path is None:
        config = Config()
        raw_path = config.get_path("data", "raw_path")
        path = raw_path.replace("transactions.csv", "accounts.csv")

    if not Path(path).exists():
        raise FileNotFoundError(f"Account file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} accounts from {path}")
    return df


def load_processed_features(path=None):
    """Load processed feature data.

    Args:
        path: Optional path override.

    Returns:
        pd.DataFrame with engineered features.
    """
    if path is None:
        config = Config()
        path = config.get_path("data", "processed_path")

    if not Path(path).exists():
        raise FileNotFoundError(f"Processed data not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} processed records from {path}")
    return df

"""Generate synthetic AML transaction dataset with suspicious patterns."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_accounts(n_accounts=5000, random_state=42):
    """Generate synthetic bank accounts.

    Args:
        n_accounts: Number of accounts.
        random_state: Random seed.

    Returns:
        DataFrame with account details.
    """
    np.random.seed(random_state)

    account_types = np.random.choice(
        ["personal", "business", "corporate"], n_accounts, p=[0.6, 0.3, 0.1]
    )
    countries = np.random.choice(
        ["US", "UK", "DE", "SG", "AE", "CH", "HK", "KY", "PA", "BS"],
        n_accounts,
        p=[0.35, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05],
    )
    risk_ratings = np.random.choice(
        ["low", "medium", "high"], n_accounts, p=[0.5, 0.35, 0.15]
    )

    return pd.DataFrame({
        "account_id": [f"ACC{i:06d}" for i in range(n_accounts)],
        "account_type": account_types,
        "country": countries,
        "risk_rating": risk_ratings,
        "opening_balance": np.random.lognormal(10, 1.5, n_accounts).round(2),
    })


def generate_transactions(n_transactions=200000, n_accounts=5000, suspicious_rate=0.03, random_state=42):
    """Generate synthetic financial transactions with embedded suspicious patterns.

    Suspicious patterns include:
    - Structuring: Multiple transactions just below reporting thresholds
    - Layering: Rapid movement through multiple accounts
    - Round-tripping: Money sent out and returned via different paths
    - Rapid movement: High-frequency transfers in short windows

    Args:
        n_transactions: Total number of transactions.
        n_accounts: Number of unique accounts.
        suspicious_rate: Fraction of transactions that are suspicious.
        random_state: Random seed.

    Returns:
        Tuple of (transactions DataFrame, accounts DataFrame).
    """
    np.random.seed(random_state)
    logger.info(f"Generating {n_transactions} transactions across {n_accounts} accounts...")

    accounts = generate_accounts(n_accounts, random_state)
    account_ids = accounts["account_id"].values

    n_suspicious = int(n_transactions * suspicious_rate)
    n_normal = n_transactions - n_suspicious

    # --- Normal transactions ---
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    normal_dates = [
        start_date + timedelta(days=int(d))
        for d in np.random.uniform(0, date_range_days, n_normal)
    ]
    normal_hours = np.random.choice(range(24), n_normal, p=_business_hour_probs())
    normal_minutes = np.random.randint(0, 60, n_normal)

    normal_timestamps = [
        d.replace(hour=int(h), minute=int(m))
        for d, h, m in zip(normal_dates, normal_hours, normal_minutes)
    ]

    normal_amounts = np.random.lognormal(6, 2, n_normal).clip(10, 500000).round(2)
    normal_types = np.random.choice(
        ["wire_transfer", "ach", "check", "cash_deposit", "cash_withdrawal", "internal_transfer"],
        n_normal,
        p=[0.20, 0.25, 0.10, 0.15, 0.10, 0.20],
    )

    normal_df = pd.DataFrame({
        "timestamp": normal_timestamps,
        "sender_account": np.random.choice(account_ids, n_normal),
        "receiver_account": np.random.choice(account_ids, n_normal),
        "amount": normal_amounts,
        "transaction_type": normal_types,
        "currency": np.random.choice(["USD", "EUR", "GBP", "CHF"], n_normal, p=[0.5, 0.25, 0.15, 0.1]),
        "is_suspicious": 0,
        "pattern_type": "normal",
    })

    # --- Suspicious transactions ---
    suspicious_dfs = []
    remaining = n_suspicious

    # Pattern 1: Structuring (just below $10K threshold)
    n_structuring = remaining // 4
    struct_accounts = np.random.choice(account_ids, n_structuring // 3 + 1)
    struct_txns = []
    for acc in struct_accounts:
        n_txn = np.random.randint(3, 8)
        if len(struct_txns) + n_txn > n_structuring:
            n_txn = n_structuring - len(struct_txns)
        if n_txn <= 0:
            break
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        for j in range(n_txn):
            struct_txns.append({
                "timestamp": base_date + timedelta(hours=np.random.randint(1, 48)),
                "sender_account": acc,
                "receiver_account": np.random.choice(account_ids),
                "amount": round(np.random.uniform(8000, 9999), 2),
                "transaction_type": "cash_deposit",
                "currency": "USD",
                "is_suspicious": 1,
                "pattern_type": "structuring",
            })
    if struct_txns:
        suspicious_dfs.append(pd.DataFrame(struct_txns))
        remaining -= len(struct_txns)

    # Pattern 2: Layering (rapid multi-hop transfers)
    n_layering = remaining // 3
    layer_txns = []
    for _ in range(n_layering // 5 + 1):
        chain_len = np.random.randint(3, 6)
        if len(layer_txns) + chain_len > n_layering:
            chain_len = n_layering - len(layer_txns)
        if chain_len <= 0:
            break
        chain_accounts = np.random.choice(account_ids, chain_len + 1, replace=False)
        base_amount = round(np.random.uniform(50000, 200000), 2)
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        for j in range(chain_len):
            layer_txns.append({
                "timestamp": base_date + timedelta(minutes=np.random.randint(5, 60) * (j + 1)),
                "sender_account": chain_accounts[j],
                "receiver_account": chain_accounts[j + 1],
                "amount": round(base_amount * np.random.uniform(0.95, 1.0), 2),
                "transaction_type": "wire_transfer",
                "currency": np.random.choice(["USD", "EUR", "CHF"]),
                "is_suspicious": 1,
                "pattern_type": "layering",
            })
    if layer_txns:
        suspicious_dfs.append(pd.DataFrame(layer_txns))
        remaining -= len(layer_txns)

    # Pattern 3: Round-tripping
    n_roundtrip = remaining // 2
    rt_txns = []
    for _ in range(n_roundtrip // 2 + 1):
        if len(rt_txns) + 2 > n_roundtrip:
            break
        acc_a, acc_b = np.random.choice(account_ids, 2, replace=False)
        amount = round(np.random.uniform(20000, 100000), 2)
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        rt_txns.append({
            "timestamp": base_date,
            "sender_account": acc_a,
            "receiver_account": acc_b,
            "amount": amount,
            "transaction_type": "wire_transfer",
            "currency": "USD",
            "is_suspicious": 1,
            "pattern_type": "round_tripping",
        })
        rt_txns.append({
            "timestamp": base_date + timedelta(days=np.random.randint(1, 10)),
            "sender_account": acc_b,
            "receiver_account": acc_a,
            "amount": round(amount * np.random.uniform(0.9, 1.05), 2),
            "transaction_type": "wire_transfer",
            "currency": np.random.choice(["USD", "EUR"]),
            "is_suspicious": 1,
            "pattern_type": "round_tripping",
        })
    if rt_txns:
        suspicious_dfs.append(pd.DataFrame(rt_txns))
        remaining -= len(rt_txns)

    # Pattern 4: Rapid movement (many transactions in short burst)
    rapid_txns = []
    for _ in range(remaining // 10 + 1):
        n_burst = min(np.random.randint(5, 12), remaining - len(rapid_txns))
        if n_burst <= 0:
            break
        acc = np.random.choice(account_ids)
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        for j in range(n_burst):
            rapid_txns.append({
                "timestamp": base_date + timedelta(minutes=np.random.randint(1, 30) * (j + 1)),
                "sender_account": acc,
                "receiver_account": np.random.choice(account_ids),
                "amount": round(np.random.uniform(1000, 50000), 2),
                "transaction_type": np.random.choice(["wire_transfer", "internal_transfer"]),
                "currency": "USD",
                "is_suspicious": 1,
                "pattern_type": "rapid_movement",
            })
    if rapid_txns:
        suspicious_dfs.append(pd.DataFrame(rapid_txns))

    # Combine all
    all_dfs = [normal_df] + suspicious_dfs
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i:08d}" for i in range(len(df))]

    logger.info(f"Generated {len(df)} transactions, {df['is_suspicious'].sum()} suspicious "
                f"({df['is_suspicious'].mean():.2%})")
    logger.info(f"Pattern distribution:\n{df[df['is_suspicious']==1]['pattern_type'].value_counts().to_string()}")

    return df, accounts


def _business_hour_probs():
    """Generate probability distribution favoring business hours."""
    probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 0-5
        0.02, 0.04, 0.07, 0.09, 0.09, 0.08,       # 6-11
        0.06, 0.08, 0.09, 0.08, 0.07, 0.05,        # 12-17
        0.04, 0.03, 0.03, 0.02, 0.015, 0.01,       # 18-23
    ])
    return probs / probs.sum()


def main():
    from src.utils.config import Config
    config = Config()
    n = config.get("data", "n_transactions", default=200000)
    seed = config.get("data", "random_state", default=42)

    df, accounts = generate_transactions(n_transactions=n, random_state=seed)

    raw_path = config.get_path("data", "raw_path")
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    accounts_path = raw_path.replace("transactions.csv", "accounts.csv")
    accounts.to_csv(accounts_path, index=False)
    logger.info(f"Saved {len(df)} transactions and {len(accounts)} accounts")

    return df, accounts


if __name__ == "__main__":
    main()

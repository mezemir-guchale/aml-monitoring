"""
AML Transaction Monitoring Dashboard
=====================================
Interactive Streamlit dashboard for Anti-Money Laundering detection and analysis.
Generates synthetic transaction data, runs anomaly detection models, and
visualizes suspicious patterns, risk scores, and transaction networks.

Author: Mezemir Neway Guchale
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AML Transaction Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {font-size: 0.85rem; color: #808495;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data generation (self-contained, no src imports needed for deployment)
# ---------------------------------------------------------------------------

def _business_hour_probs():
    probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,
        0.02, 0.04, 0.07, 0.09, 0.09, 0.08,
        0.06, 0.08, 0.09, 0.08, 0.07, 0.05,
        0.04, 0.03, 0.03, 0.02, 0.015, 0.01,
    ])
    return probs / probs.sum()


@st.cache_data(show_spinner="Generating synthetic transactions...")
def generate_transactions(n_transactions=50000, n_accounts=2000, suspicious_rate=0.03, seed=42):
    """Generate synthetic AML transaction data with embedded suspicious patterns."""
    np.random.seed(seed)

    # Accounts
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
    accounts = pd.DataFrame({
        "account_id": [f"ACC{i:06d}" for i in range(n_accounts)],
        "account_type": account_types,
        "country": countries,
        "risk_rating": risk_ratings,
        "opening_balance": np.random.lognormal(10, 1.5, n_accounts).round(2),
    })
    account_ids = accounts["account_id"].values

    n_suspicious = int(n_transactions * suspicious_rate)
    n_normal = n_transactions - n_suspicious

    # Normal transactions
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

    normal_df = pd.DataFrame({
        "timestamp": normal_timestamps,
        "sender_account": np.random.choice(account_ids, n_normal),
        "receiver_account": np.random.choice(account_ids, n_normal),
        "amount": np.random.lognormal(6, 2, n_normal).clip(10, 500000).round(2),
        "transaction_type": np.random.choice(
            ["wire_transfer", "ach", "check", "cash_deposit", "cash_withdrawal", "internal_transfer"],
            n_normal, p=[0.20, 0.25, 0.10, 0.15, 0.10, 0.20],
        ),
        "currency": np.random.choice(["USD", "EUR", "GBP", "CHF"], n_normal, p=[0.5, 0.25, 0.15, 0.1]),
        "is_suspicious": 0,
        "pattern_type": "normal",
    })

    # Suspicious patterns
    suspicious_dfs = []
    remaining = n_suspicious

    # Structuring
    n_struct = remaining // 4
    struct_txns = []
    struct_accounts = np.random.choice(account_ids, n_struct // 3 + 1)
    for acc in struct_accounts:
        n_txn = np.random.randint(3, 8)
        if len(struct_txns) + n_txn > n_struct:
            n_txn = n_struct - len(struct_txns)
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

    # Layering
    n_layer = remaining // 3
    layer_txns = []
    for _ in range(n_layer // 5 + 1):
        chain_len = np.random.randint(3, 6)
        if len(layer_txns) + chain_len > n_layer:
            chain_len = n_layer - len(layer_txns)
        if chain_len <= 0:
            break
        chain_accs = np.random.choice(account_ids, chain_len + 1, replace=False)
        base_amount = round(np.random.uniform(50000, 200000), 2)
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        for j in range(chain_len):
            layer_txns.append({
                "timestamp": base_date + timedelta(minutes=np.random.randint(5, 60) * (j + 1)),
                "sender_account": chain_accs[j],
                "receiver_account": chain_accs[j + 1],
                "amount": round(base_amount * np.random.uniform(0.95, 1.0), 2),
                "transaction_type": "wire_transfer",
                "currency": np.random.choice(["USD", "EUR", "CHF"]),
                "is_suspicious": 1,
                "pattern_type": "layering",
            })
    if layer_txns:
        suspicious_dfs.append(pd.DataFrame(layer_txns))
        remaining -= len(layer_txns)

    # Round-tripping
    n_rt = remaining // 2
    rt_txns = []
    for _ in range(n_rt // 2 + 1):
        if len(rt_txns) + 2 > n_rt:
            break
        acc_a, acc_b = np.random.choice(account_ids, 2, replace=False)
        amount = round(np.random.uniform(20000, 100000), 2)
        base_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        rt_txns.append({
            "timestamp": base_date,
            "sender_account": acc_a, "receiver_account": acc_b,
            "amount": amount, "transaction_type": "wire_transfer",
            "currency": "USD", "is_suspicious": 1, "pattern_type": "round_tripping",
        })
        rt_txns.append({
            "timestamp": base_date + timedelta(days=np.random.randint(1, 10)),
            "sender_account": acc_b, "receiver_account": acc_a,
            "amount": round(amount * np.random.uniform(0.9, 1.05), 2),
            "transaction_type": "wire_transfer",
            "currency": np.random.choice(["USD", "EUR"]),
            "is_suspicious": 1, "pattern_type": "round_tripping",
        })
    if rt_txns:
        suspicious_dfs.append(pd.DataFrame(rt_txns))
        remaining -= len(rt_txns)

    # Rapid movement
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
                "currency": "USD", "is_suspicious": 1, "pattern_type": "rapid_movement",
            })
    if rapid_txns:
        suspicious_dfs.append(pd.DataFrame(rapid_txns))

    df = pd.concat([normal_df] + suspicious_dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i:08d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df, accounts


# ---------------------------------------------------------------------------
# Feature engineering (lightweight, self-contained)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df):
    """Compute features for detection models."""
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])

    # Time features
    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_night"] = ((result["hour"] < 6) | (result["hour"] >= 22)).astype(int)
    result["month"] = result["timestamp"].dt.month

    # Amount features
    acct_stats = result.groupby("sender_account")["amount"].agg(
        ["mean", "std", "median", "count"]
    ).rename(columns={
        "mean": "acct_avg_amount", "std": "acct_std_amount",
        "median": "acct_median_amount", "count": "acct_total_txns",
    })
    result = result.merge(acct_stats, left_on="sender_account", right_index=True, how="left")
    result["amount_zscore"] = np.where(
        result["acct_std_amount"] > 0,
        (result["amount"] - result["acct_avg_amount"]) / result["acct_std_amount"], 0,
    )
    result["amount_to_avg_ratio"] = np.where(
        result["acct_avg_amount"] > 0, result["amount"] / result["acct_avg_amount"], 0,
    )
    result["is_round_amount"] = (result["amount"] % 1000 == 0).astype(int)
    result["near_threshold"] = ((result["amount"] >= 8000) & (result["amount"] < 10000)).astype(int)

    # Network features
    sender_cps = result.groupby("sender_account")["receiver_account"].nunique().rename("n_unique_receivers")
    result = result.merge(sender_cps, left_on="sender_account", right_index=True, how="left")
    receiver_cps = result.groupby("receiver_account")["sender_account"].nunique().rename("n_unique_senders")
    result = result.merge(receiver_cps, left_on="receiver_account", right_index=True, how="left")
    result["is_self_transfer"] = (result["sender_account"] == result["receiver_account"]).astype(int)

    # Time since last transaction per account
    result = result.sort_values(["sender_account", "timestamp"])
    result["time_since_last_txn"] = (
        result.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    result["is_rapid"] = ((result["time_since_last_txn"] > 0) & (result["time_since_last_txn"] < 300)).astype(int)

    # Fill NaN
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    return result


# ---------------------------------------------------------------------------
# Detection pipeline (self-contained)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "amount", "hour", "day_of_week", "is_weekend", "is_night", "month",
    "acct_avg_amount", "acct_std_amount", "acct_median_amount", "acct_total_txns",
    "amount_zscore", "amount_to_avg_ratio", "is_round_amount", "near_threshold",
    "n_unique_receivers", "n_unique_senders", "is_self_transfer",
    "time_since_last_txn", "is_rapid",
]


@st.cache_data(show_spinner="Running anomaly detection models...")
def run_detection(df_feat):
    """Run Isolation Forest, LOF, rule-based detection, and ensemble scoring."""
    available_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available_cols].values.astype(np.float64)

    # Isolation Forest
    scaler_if = StandardScaler()
    X_scaled = scaler_if.fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    raw_if = iso.score_samples(X_scaled)
    mn, mx = raw_if.min(), raw_if.max()
    if_scores = 1 - (raw_if - mn) / (mx - mn) if mx > mn else np.zeros_like(raw_if)

    # LOF
    scaler_lof = StandardScaler()
    X_lof = scaler_lof.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03, n_jobs=-1)
    lof.fit_predict(X_lof)
    raw_lof = lof.negative_outlier_factor_
    mn, mx = raw_lof.min(), raw_lof.max()
    lof_scores = 1 - (raw_lof - mn) / (mx - mn) if mx > mn else np.zeros_like(raw_lof)

    # Rule-based
    rule_scores = np.zeros(len(df_feat))
    if "near_threshold" in df_feat.columns:
        rule_scores += df_feat["near_threshold"].values * 0.3
    if "amount" in df_feat.columns:
        large_cash = (
            (df_feat["amount"] > 50000)
            & df_feat["transaction_type"].isin(["cash_deposit", "cash_withdrawal"])
        ).astype(float).values
        rule_scores += large_cash * 0.25
    if "is_rapid" in df_feat.columns:
        rule_scores += df_feat["is_rapid"].values * 0.2
    if "is_night" in df_feat.columns and "amount" in df_feat.columns:
        night_high = ((df_feat["is_night"] == 1) & (df_feat["amount"] > 10000)).astype(float).values
        rule_scores += night_high * 0.15
    if "is_round_amount" in df_feat.columns and "amount" in df_feat.columns:
        round_large = ((df_feat["is_round_amount"] == 1) & (df_feat["amount"] > 5000)).astype(float).values
        rule_scores += round_large * 0.1
    rule_scores = np.clip(rule_scores, 0, 1)

    # Ensemble
    w_if, w_lof, w_rules = 0.35, 0.25, 0.40
    total_w = w_if + w_lof + w_rules
    ensemble = (if_scores * w_if + lof_scores * w_lof + rule_scores * w_rules) / total_w

    return if_scores, lof_scores, rule_scores, ensemble


@st.cache_data(show_spinner="Clustering transaction groups...")
def run_clustering(df_feat):
    """Run K-Means and DBSCAN clustering on transaction features."""
    available_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available_cols].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    # DBSCAN on a sample for speed
    sample_size = min(15000, len(X_scaled))
    idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
    db = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
    db_labels_sample = db.fit_predict(X_scaled[idx])

    db_labels = np.full(len(X_scaled), -2)  # -2 = not sampled
    db_labels[idx] = db_labels_sample

    return km_labels, db_labels, idx


# ---------------------------------------------------------------------------
# Load data and run pipeline
# ---------------------------------------------------------------------------

df_raw, accounts = generate_transactions(n_transactions=50000, n_accounts=2000, seed=42)
df_feat = engineer_features(df_raw)
if_scores, lof_scores, rule_scores, ensemble_scores = run_detection(df_feat)
km_labels, db_labels, db_idx = run_clustering(df_feat)

df_feat["if_score"] = if_scores
df_feat["lof_score"] = lof_scores
df_feat["rule_score"] = rule_scores
df_feat["ensemble_score"] = ensemble_scores
df_feat["km_cluster"] = km_labels
df_feat["db_cluster"] = db_labels

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Filters")

min_date = df_feat["timestamp"].min().date()
max_date = df_feat["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    d_start, d_end = date_range
else:
    d_start, d_end = min_date, max_date

amount_min, amount_max = float(df_feat["amount"].min()), float(df_feat["amount"].max())
amount_range = st.sidebar.slider(
    "Amount range ($)",
    min_value=0.0,
    max_value=min(amount_max, 500000.0),
    value=(0.0, min(amount_max, 500000.0)),
    step=1000.0,
    format="$%,.0f",
)

pattern_options = ["All"] + sorted(df_feat["pattern_type"].unique().tolist())
selected_pattern = st.sidebar.selectbox("Pattern type", pattern_options)

threshold = st.sidebar.slider(
    "Alert threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
)

# Apply filters
mask = (
    (df_feat["timestamp"].dt.date >= d_start)
    & (df_feat["timestamp"].dt.date <= d_end)
    & (df_feat["amount"] >= amount_range[0])
    & (df_feat["amount"] <= amount_range[1])
)
if selected_pattern != "All":
    mask &= df_feat["pattern_type"] == selected_pattern

df_display = df_feat[mask].copy()
df_display["is_alert"] = (df_display["ensemble_score"] >= threshold).astype(int)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# AML Transaction Monitoring Dashboard")
st.markdown(
    "Real-time surveillance of financial transactions using ensemble anomaly detection "
    "(Isolation Forest + LOF + Rule-based scoring) on synthetic data."
)

# ---------------------------------------------------------------------------
# Key metrics row
# ---------------------------------------------------------------------------

n_total = len(df_display)
n_flagged = int(df_display["is_alert"].sum())
alert_rate = n_flagged / n_total if n_total > 0 else 0
n_patterns = df_display.loc[
    (df_display["is_suspicious"] == 1) & (df_display["is_alert"] == 1),
    "pattern_type"
].nunique()
total_value = df_display["amount"].sum()
avg_score = df_display["ensemble_score"].mean()

cols = st.columns(6)
cols[0].metric("Total Transactions", f"{n_total:,}")
cols[1].metric("Flagged Alerts", f"{n_flagged:,}", delta=f"{alert_rate:.2%} alert rate")
cols[2].metric("Alert Rate", f"{alert_rate:.2%}")
cols[3].metric("Patterns Detected", n_patterns)
cols[4].metric("Total Value", f"${total_value:,.0f}")
cols[5].metric("Avg Risk Score", f"{avg_score:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_alerts, tab_patterns, tab_network, tab_risk = st.tabs([
    "Overview", "Flagged Transactions", "Pattern Analysis", "Network & Clusters", "Risk Heatmap",
])

# ===== TAB 1: OVERVIEW =====
with tab_overview:
    col_left, col_right = st.columns(2)

    # Anomaly score distribution
    with col_left:
        st.subheader("Anomaly Score Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_display["ensemble_score"], nbinsx=80,
            marker_color="#636EFA", opacity=0.75, name="All Transactions",
        ))
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#EF553B",
                           annotation_text=f"Threshold ({threshold})")
        fig_dist.update_layout(
            xaxis_title="Ensemble Anomaly Score",
            yaxis_title="Count",
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Score comparison: normal vs suspicious
    with col_right:
        st.subheader("Score by Ground Truth Label")
        fig_box = go.Figure()
        for label, name, color in [(0, "Normal", "#636EFA"), (1, "Suspicious", "#EF553B")]:
            subset = df_display[df_display["is_suspicious"] == label]["ensemble_score"]
            fig_box.add_trace(go.Box(y=subset, name=name, marker_color=color))
        fig_box.update_layout(
            yaxis_title="Ensemble Score",
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Transactions over time
    st.subheader("Transaction Volume Over Time")
    daily = df_display.set_index("timestamp").resample("W").agg(
        total=("amount", "count"),
        flagged=("is_alert", "sum"),
        volume=("amount", "sum"),
    ).reset_index()
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(
        go.Bar(x=daily["timestamp"], y=daily["total"], name="Total Txns",
               marker_color="#636EFA", opacity=0.5),
        secondary_y=False,
    )
    fig_time.add_trace(
        go.Scatter(x=daily["timestamp"], y=daily["flagged"], name="Flagged",
                   mode="lines+markers", line=dict(color="#EF553B", width=2)),
        secondary_y=True,
    )
    fig_time.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_time.update_yaxes(title_text="Total Transactions", secondary_y=False)
    fig_time.update_yaxes(title_text="Flagged Alerts", secondary_y=True)
    st.plotly_chart(fig_time, use_container_width=True)

    # Model comparison
    st.subheader("Model Score Comparison")
    col_m1, col_m2, col_m3 = st.columns(3)
    for col, score_col, title, color in [
        (col_m1, "if_score", "Isolation Forest", "#636EFA"),
        (col_m2, "lof_score", "Local Outlier Factor", "#AB63FA"),
        (col_m3, "rule_score", "Rule-Based", "#00CC96"),
    ]:
        with col:
            fig = go.Figure(go.Histogram(
                x=df_display[score_col], nbinsx=50,
                marker_color=color, opacity=0.8,
            ))
            fig.update_layout(
                title=title, template="plotly_dark", height=260,
                margin=dict(l=30, r=10, t=40, b=30),
                xaxis_title="Score", yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)


# ===== TAB 2: FLAGGED TRANSACTIONS =====
with tab_alerts:
    st.subheader("Flagged Transactions")
    flagged = df_display[df_display["is_alert"] == 1].sort_values("ensemble_score", ascending=False)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flagged", f"{len(flagged):,}")
    if len(flagged) > 0:
        c2.metric("Avg Flagged Amount", f"${flagged['amount'].mean():,.2f}")
        c3.metric("Max Score", f"{flagged['ensemble_score'].max():.3f}")
        truly_sus = flagged["is_suspicious"].sum()
        c4.metric("True Positives", f"{truly_sus:,} / {len(flagged):,}")
    else:
        c2.metric("Avg Flagged Amount", "$0")
        c3.metric("Max Score", "0")
        c4.metric("True Positives", "0 / 0")

    # Display table
    display_cols = [
        "transaction_id", "timestamp", "sender_account", "receiver_account",
        "amount", "currency", "transaction_type", "pattern_type",
        "ensemble_score", "is_suspicious",
    ]
    available_display = [c for c in display_cols if c in flagged.columns]
    st.dataframe(
        flagged[available_display].head(500),
        use_container_width=True,
        height=450,
        column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%,.2f"),
            "ensemble_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.3f",
            ),
        },
    )


# ===== TAB 3: PATTERN ANALYSIS =====
with tab_patterns:
    st.subheader("Suspicious Pattern Breakdown")
    suspicious_only = df_display[df_display["is_suspicious"] == 1]

    if len(suspicious_only) > 0:
        col_p1, col_p2 = st.columns(2)

        # Pattern distribution pie
        with col_p1:
            pattern_counts = suspicious_only["pattern_type"].value_counts().reset_index()
            pattern_counts.columns = ["Pattern", "Count"]
            fig_pie = px.pie(
                pattern_counts, names="Pattern", values="Count",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4,
            )
            fig_pie.update_layout(
                title="Pattern Distribution",
                template="plotly_dark", height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Detection rate by pattern
        with col_p2:
            suspicious_only = suspicious_only.copy()
            suspicious_only["detected"] = (suspicious_only["ensemble_score"] >= threshold).astype(int)
            det_rate = suspicious_only.groupby("pattern_type").agg(
                total=("detected", "count"),
                detected=("detected", "sum"),
                avg_score=("ensemble_score", "mean"),
            ).reset_index()
            det_rate["detection_rate"] = (det_rate["detected"] / det_rate["total"] * 100).round(1)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=det_rate["pattern_type"], y=det_rate["detection_rate"],
                marker_color=["#EF553B" if r < 50 else "#00CC96" for r in det_rate["detection_rate"]],
                text=[f"{r:.1f}%" for r in det_rate["detection_rate"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Detection Rate by Pattern (%)",
                yaxis_title="Detection Rate (%)",
                template="plotly_dark", height=380,
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Amount distribution by pattern
        st.subheader("Amount Distribution by Pattern")
        fig_violin = px.violin(
            suspicious_only, x="pattern_type", y="amount", color="pattern_type",
            box=True, points=False,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_violin.update_layout(
            template="plotly_dark", height=380,
            margin=dict(l=40, r=20, t=30, b=40),
            showlegend=False,
            yaxis_title="Amount ($)",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

        # Score heatmap by pattern and hour
        st.subheader("Average Risk Score by Pattern and Hour")
        heat_data = suspicious_only.pivot_table(
            values="ensemble_score", index="pattern_type",
            columns="hour", aggfunc="mean",
        ).fillna(0)
        fig_heat_pattern = px.imshow(
            heat_data, color_continuous_scale="RdYlGn_r",
            labels=dict(x="Hour of Day", y="Pattern Type", color="Avg Score"),
        )
        fig_heat_pattern.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_heat_pattern, use_container_width=True)
    else:
        st.info("No suspicious transactions in the current filter selection.")


# ===== TAB 4: NETWORK & CLUSTERS =====
with tab_network:
    st.subheader("Transaction Network of Flagged Accounts")

    flagged_txns = df_display[df_display["is_alert"] == 1]

    if len(flagged_txns) > 0:
        # Build network from top flagged transactions
        top_flagged = flagged_txns.nlargest(200, "ensemble_score")

        # Get unique accounts involved
        all_accts = pd.concat([top_flagged["sender_account"], top_flagged["receiver_account"]]).unique()
        acct_to_idx = {a: i for i, a in enumerate(all_accts)}

        # Build edge list with amounts
        edge_x, edge_y = [], []
        np.random.seed(42)
        positions = {a: (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for a in all_accts}

        # Simple force-directed positioning using connected accounts
        for _, row in top_flagged.iterrows():
            s, r = row["sender_account"], row["receiver_account"]
            sx, sy = positions[s]
            rx, ry = positions[r]
            edge_x.extend([sx, rx, None])
            edge_y.extend([sy, ry, None])

        # Node properties
        node_x = [positions[a][0] for a in all_accts]
        node_y = [positions[a][1] for a in all_accts]

        # Color by avg score
        node_scores = []
        for a in all_accts:
            a_mask = (top_flagged["sender_account"] == a) | (top_flagged["receiver_account"] == a)
            node_scores.append(top_flagged.loc[a_mask, "ensemble_score"].mean() if a_mask.any() else 0)

        # Node degree
        node_sizes = []
        for a in all_accts:
            deg = ((top_flagged["sender_account"] == a) | (top_flagged["receiver_account"] == a)).sum()
            node_sizes.append(max(6, min(30, deg * 3)))

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#555"), hoverinfo="none",
        ))
        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=node_sizes, color=node_scores,
                colorscale="RdYlGn_r", colorbar=dict(title="Avg Score"),
                line=dict(width=1, color="#333"),
            ),
            text=[f"{a}<br>Score: {s:.3f}" for a, s in zip(all_accts, node_scores)],
            hoverinfo="text",
        ))
        fig_net.update_layout(
            template="plotly_dark", height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Cluster analysis
        st.subheader("K-Means Cluster Analysis")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            cluster_stats = df_display.groupby("km_cluster").agg(
                count=("amount", "size"),
                avg_amount=("amount", "mean"),
                avg_score=("ensemble_score", "mean"),
                flagged_pct=("is_alert", "mean"),
                suspicious_pct=("is_suspicious", "mean"),
            ).round(4).reset_index()

            fig_cluster = px.scatter(
                cluster_stats, x="avg_amount", y="avg_score",
                size="count", color="flagged_pct",
                color_continuous_scale="RdYlGn_r",
                hover_data=["km_cluster", "count", "suspicious_pct"],
                labels={
                    "avg_amount": "Avg Amount ($)",
                    "avg_score": "Avg Risk Score",
                    "flagged_pct": "Flagged %",
                },
            )
            fig_cluster.update_layout(
                title="Clusters: Amount vs Risk Score",
                template="plotly_dark", height=400,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        with col_c2:
            fig_cluster_bar = go.Figure()
            fig_cluster_bar.add_trace(go.Bar(
                x=cluster_stats["km_cluster"].astype(str),
                y=cluster_stats["count"],
                marker_color=[
                    "#EF553B" if p > 0.1 else "#636EFA"
                    for p in cluster_stats["flagged_pct"]
                ],
                text=[f"{p:.1%}" for p in cluster_stats["flagged_pct"]],
                textposition="outside",
            ))
            fig_cluster_bar.update_layout(
                title="Cluster Sizes (label = flagged %)",
                xaxis_title="Cluster", yaxis_title="Transaction Count",
                template="plotly_dark", height=400,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_cluster_bar, use_container_width=True)

        # DBSCAN outlier analysis
        st.subheader("DBSCAN Outlier Detection")
        db_sampled = df_display[df_display["db_cluster"] != -2]
        if len(db_sampled) > 0:
            db_sampled = db_sampled.copy()
            db_sampled["is_dbscan_outlier"] = (db_sampled["db_cluster"] == -1).astype(int)
            n_outliers = db_sampled["is_dbscan_outlier"].sum()
            n_sampled = len(db_sampled)
            c1_db, c2_db, c3_db = st.columns(3)
            c1_db.metric("Sampled Transactions", f"{n_sampled:,}")
            c2_db.metric("DBSCAN Outliers", f"{n_outliers:,}")
            c3_db.metric("Outlier Rate", f"{n_outliers/n_sampled:.2%}" if n_sampled > 0 else "0%")
    else:
        st.info("No flagged transactions in the current filter selection.")


# ===== TAB 5: RISK HEATMAP =====
with tab_risk:
    st.subheader("Risk Scoring Heatmap by Account")

    # Aggregate risk scores by account
    account_risk = df_display.groupby("sender_account").agg(
        avg_score=("ensemble_score", "mean"),
        max_score=("ensemble_score", "max"),
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        flagged_count=("is_alert", "sum"),
    ).reset_index()

    # Merge account metadata
    account_risk = account_risk.merge(accounts, left_on="sender_account", right_on="account_id", how="left")

    # Top risky accounts
    top_n = st.slider("Number of top risky accounts to display", 10, 100, 30, 5)
    top_risky = account_risk.nlargest(top_n, "avg_score")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("**Top Risky Accounts**")
        display_risk_cols = [
            "sender_account", "avg_score", "max_score", "txn_count",
            "total_amount", "flagged_count", "account_type", "country", "risk_rating",
        ]
        available_risk_cols = [c for c in display_risk_cols if c in top_risky.columns]
        st.dataframe(
            top_risky[available_risk_cols],
            use_container_width=True,
            height=400,
            column_config={
                "total_amount": st.column_config.NumberColumn("Total Amount", format="$%,.0f"),
                "avg_score": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=1, format="%.3f"),
                "max_score": st.column_config.ProgressColumn("Max Score", min_value=0, max_value=1, format="%.3f"),
            },
        )

    with col_r2:
        fig_risk_scatter = px.scatter(
            top_risky, x="txn_count", y="avg_score",
            size="total_amount", color="risk_rating",
            color_discrete_map={"low": "#00CC96", "medium": "#FFA15A", "high": "#EF553B"},
            hover_data=["sender_account", "country", "flagged_count"],
            labels={"txn_count": "Transaction Count", "avg_score": "Avg Risk Score"},
        )
        fig_risk_scatter.update_layout(
            title="Account Risk Profile",
            template="plotly_dark", height=420,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_risk_scatter, use_container_width=True)

    # Heatmap: country x account_type
    st.subheader("Risk Heatmap: Country x Account Type")
    heat_risk = account_risk.pivot_table(
        values="avg_score", index="country", columns="account_type", aggfunc="mean",
    ).fillna(0)
    fig_heat = px.imshow(
        heat_risk, color_continuous_scale="RdYlGn_r",
        labels=dict(x="Account Type", y="Country", color="Avg Risk Score"),
        text_auto=".3f",
    )
    fig_heat.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Risk by country bar chart
    st.subheader("Average Risk Score by Country")
    country_risk = account_risk.groupby("country").agg(
        avg_score=("avg_score", "mean"),
        n_accounts=("sender_account", "nunique"),
        total_flagged=("flagged_count", "sum"),
    ).reset_index().sort_values("avg_score", ascending=False)

    fig_country = go.Figure()
    fig_country.add_trace(go.Bar(
        x=country_risk["country"], y=country_risk["avg_score"],
        marker_color=[
            "#EF553B" if s > country_risk["avg_score"].median() else "#636EFA"
            for s in country_risk["avg_score"]
        ],
        text=[f"{s:.3f}" for s in country_risk["avg_score"]],
        textposition="outside",
    ))
    fig_country.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis_title="Avg Risk Score",
        yaxis=dict(range=[0, country_risk["avg_score"].max() * 1.2]),
    )
    st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "AML Transaction Monitoring Dashboard | Synthetic data generated for demonstration purposes. "
    "Models: Isolation Forest, Local Outlier Factor, Rule-Based Scoring, K-Means & DBSCAN Clustering. "
    "Built by Mezemir Neway Guchale."
)

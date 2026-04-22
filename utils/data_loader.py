"""
utils/data_loader.py
-------------------
FIX SUMMARY:
  OLD: generate_synthetic_price_data() → 49 fake data points, price ~$30k
  NEW: fetch_real_price_data()         → 365 real daily BTC prices from CoinGecko
       This fixes:
         1. Stale data problem (model was predicting $29k when BTC = $77k)
         2. Too few samples (49 → 365 data points)
         3. Distribution shift (scaler now fitted on actual recent prices)
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import requests
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────
# 1.  Elliptic Dataset Loader  (unchanged)
# ──────────────────────────────────────────

def load_elliptic_dataset(data_dir: str = "data/elliptic_bitcoin_dataset"):
    """
    Load the Elliptic Bitcoin Dataset from CSV files.

    Returns
    -------
    features_df : pd.DataFrame  – node feature matrix
    edges_df    : pd.DataFrame  – edge list
    classes_df  : pd.DataFrame  – node labels (1=illicit, 2=licit, unknown)
    """
    feat_path    = os.path.join(data_dir, "elliptic_txs_features.csv")
    edge_path    = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    features_df = pd.read_csv(feat_path, header=None)
    features_df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]

    edges_df   = pd.read_csv(edge_path)
    classes_df = pd.read_csv(classes_path)
    classes_df.columns = ["txId", "class"]

    print(f"[DataLoader] Loaded {len(features_df):,} transactions, "
          f"{len(edges_df):,} edges")
    return features_df, edges_df, classes_df


# ──────────────────────────────────────────
# 2.  Build PyG Graph Object  (unchanged)
# ──────────────────────────────────────────

def build_graph(features_df, edges_df, classes_df):
    """Convert raw DataFrames into a PyTorch Geometric Data object."""
    node_ids  = features_df["txId"].values
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
    X = features_df[feature_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    label_map  = {"1": 1, "2": 0, "unknown": -1}
    classes_df = classes_df.copy()
    classes_df["label"] = classes_df["class"].astype(str).map(label_map)
    label_dict = dict(zip(classes_df["txId"], classes_df["label"]))
    y = np.array([label_dict.get(nid, -1) for nid in node_ids], dtype=np.int64)

    src = edges_df.iloc[:, 0].map(id_to_idx).dropna().astype(int).values
    dst = edges_df.iloc[:, 1].map(id_to_idx).dropna().astype(int).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    labelled = y != -1
    indices  = np.where(labelled)[0]
    np.random.seed(42)
    np.random.shuffle(indices)
    split      = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    test_mask  = torch.zeros(len(node_ids), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = train_mask,
        test_mask  = test_mask,
    )
    data.node_ids = node_ids
    print(f"[DataLoader] Graph built → nodes={data.num_nodes:,}, "
          f"edges={data.num_edges:,}, "
          f"train={train_mask.sum().item():,}, test={test_mask.sum().item():,}")
    return data, scaler


# ──────────────────────────────────────────
# 3.  Fraud Ratio per Time-Step  (unchanged)
# ──────────────────────────────────────────

def compute_fraud_ratio(features_df, classes_df):
    """Compute fraud_ratio = illicit_txns / total_labeled_txns per time step."""
    merged = features_df[["txId", "time_step"]].merge(
        classes_df, on="txId", how="left"
    )
    merged["is_fraud"]   = (merged["class"].astype(str) == "1").astype(int)
    merged["is_labeled"] = (merged["class"].astype(str) != "unknown").astype(int)

    stats = merged.groupby("time_step").agg(
        total_labeled=("is_labeled", "sum"),
        total_fraud  =("is_fraud",   "sum"),
    ).reset_index()
    stats["fraud_ratio"] = (
        stats["total_fraud"] / stats["total_labeled"].replace(0, np.nan)
    ).fillna(0)

    print(f"[DataLoader] Fraud ratio computed for {len(stats)} time steps "
          f"(mean={stats['fraud_ratio'].mean():.4f})")
    return stats


# ──────────────────────────────────────────
# 4.  FIXED: Fetch Real BTC Price Data
# ──────────────────────────────────────────
# OLD PROBLEM: generate_synthetic_price_data() returned 49 fake rows
#              starting at $30,000.  MinMaxScaler fitted on $20k-$35k range.
#              When live price = $77,979, scaler output = 3.86 (way out of range).
#              Model predicted garbage.
#
# NEW FIX:     fetch_real_price_data() downloads 365 real daily OHLCV candles
#              from CoinGecko (free, no API key).
#              - 365 data points  vs  old 49  → 7x more training data
#              - Prices are current  → scaler fitted on $40k–$80k range
#              - Model trains on actual recent market behaviour
#              - Predictions will be in the correct price range

def fetch_real_price_data(days: int = 365, coin: str = "bitcoin",
                          vs_currency: str = "usd") -> pd.DataFrame:
    """
    Fetch real daily OHLCV data from CoinGecko free API.

    Parameters
    ----------
    days        : int  – how many past days to download (max 365 for free tier)
    coin        : str  – CoinGecko coin id  (default: "bitcoin")
    vs_currency : str  – quote currency     (default: "usd")

    Returns
    -------
    df : pd.DataFrame with columns [time_step, date, close, volume, high, low]
         One row per day, sorted oldest → newest.
         time_step is 1-indexed (1, 2, 3 ... N)
    """
    print(f"[DataLoader] Fetching {days} days of real {coin.upper()} price data ...")

    # ── CoinGecko OHLC endpoint ────────────────────────────────────────────
    # Returns [[timestamp_ms, open, high, low, close], ...]
    # Note: free tier returns 4-hourly for ≤90 days, daily for >90 days
    # We use market_chart for volume too, then merge with ohlc for high/low.

    try:
        # 1. Market chart → close price + volume (daily)
        chart_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        chart_r   = requests.get(
            chart_url,
            params={"vs_currency": vs_currency, "days": days, "interval": "daily"},
            timeout=15,
        )
        chart_r.raise_for_status()
        chart     = chart_r.json()

        prices  = pd.DataFrame(chart["prices"],  columns=["ts", "close"])
        volumes = pd.DataFrame(chart["total_volumes"], columns=["ts", "volume"])

        df = prices.merge(volumes, on="ts")
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date

        # 2. OHLC endpoint → high + low (daily, max 180 days on free tier)
        # Use min of requested days and 180 to stay within free tier limits
        ohlc_days = min(days, 180)
        time.sleep(1)  # be polite to free API

        ohlc_url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        ohlc_r   = requests.get(
            ohlc_url,
            params={"vs_currency": vs_currency, "days": ohlc_days},
            timeout=15,
        )
        ohlc_r.raise_for_status()
        ohlc_raw = ohlc_r.json()  # [[ts, open, high, low, close], ...]

        ohlc_df = pd.DataFrame(ohlc_raw, columns=["ts", "open", "high", "low", "ohlc_close"])
        ohlc_df["date"] = pd.to_datetime(ohlc_df["ts"], unit="ms").dt.date

        # Daily OHLC: keep one row per day (last candle of day)
        ohlc_daily = (
            ohlc_df.groupby("date")
            .agg(high=("high", "max"), low=("low", "min"))
            .reset_index()
        )

        # 3. Merge close+volume with high+low
        df = df.groupby("date").agg(
            close=("close", "last"),
            volume=("volume", "last"),
        ).reset_index()

        df = df.merge(ohlc_daily, on="date", how="left")

        # Fill missing high/low with close ± 0.5%
        df["high"] = df["high"].fillna(df["close"] * 1.005)
        df["low"]  = df["low"].fillna(df["close"]  * 0.995)

        df = df.sort_values("date").reset_index(drop=True)
        df["time_step"] = range(1, len(df) + 1)

        # Keep only what we need
        df = df[["time_step", "date", "close", "volume", "high", "low"]]

        print(f"[DataLoader] Real price data fetched: {len(df)} daily rows "
              f"| Price range: ${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
        return df

    except Exception as e:
        print(f"[DataLoader] CoinGecko API failed ({e}). Falling back to synthetic data.")
        return _generate_fallback_price_data(n_days=days)


def _generate_fallback_price_data(n_days: int = 365) -> pd.DataFrame:
    """
    Fallback synthetic data when CoinGecko is unavailable.
    FIX: Starts at a realistic current price (~$70,000) instead of $30,000.
         This prevents the distribution shift problem even in offline mode.
    """
    print(f"[DataLoader] Generating fallback synthetic price data ({n_days} days) ...")
    rng    = np.random.default_rng(42)

    # Start at realistic current BTC price
    START_PRICE = 70_000.0

    prices = [START_PRICE]
    for _ in range(n_days - 1):
        # ±3% daily volatility with slight upward drift
        change = rng.normal(0.001, 0.03)
        prices.append(max(prices[-1] * (1 + change), 1_000))  # floor at $1k
    prices = np.array(prices)

    dates = pd.date_range(
        end   = pd.Timestamp.today().normalize(),
        periods = n_days,
        freq  = "D",
    ).date

    df = pd.DataFrame({
        "time_step": range(1, n_days + 1),
        "date"     : dates,
        "close"    : prices,
        "volume"   : rng.integers(15_000_000_000, 60_000_000_000, size=n_days).astype(float),
        "high"     : prices * (1 + rng.uniform(0.002, 0.025, n_days)),
        "low"      : prices * (1 - rng.uniform(0.002, 0.025, n_days)),
    })
    return df


# ── Keep old name as alias so existing code doesn't break ─────────────────────
def generate_synthetic_price_data(n_timesteps: int = 49, seed: int = 42):
    """
    DEPRECATED — kept for backward compatibility.
    Calls fetch_real_price_data() instead, which returns real or
    realistic fallback data starting at current BTC prices.
    """
    print("[DataLoader] WARNING: generate_synthetic_price_data() is deprecated.")
    print("             Calling fetch_real_price_data() instead for better predictions.")
    df = fetch_real_price_data(days=365)
    # Align time_step numbers to match n_timesteps if needed
    return df.tail(n_timesteps).reset_index(drop=True).assign(
        time_step=range(1, min(n_timesteps, len(df)) + 1)
    )

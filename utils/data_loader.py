"""
utils/data_loader.py
-------------------
Handles loading and preprocessing the Elliptic Bitcoin Dataset
and cryptocurrency price data.

Dataset source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Download and place files in: data/elliptic_bitcoin_dataset/
  - elliptic_txs_features.csv
  - elliptic_txs_edgelist.csv
  - elliptic_txs_classes.csv
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────
# 1.  Elliptic Dataset Loader
# ──────────────────────────────────────────

def load_elliptic_dataset(data_dir: str = "data/elliptic_bitcoin_dataset"):
    """
    Load the Elliptic Bitcoin Dataset from CSV files.

    Returns
    -------
    features_df : pd.DataFrame  – node feature matrix
    edges_df    : pd.DataFrame  – edge list
    classes_df  : pd.DataFrame  – node labels (1=illicit/fraud, 2=licit, unknown)
    """
    feat_path    = os.path.join(data_dir, "elliptic_txs_features.csv")
    edge_path    = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    # 166 columns: txId + 1 time-step + 93 local + 72 aggregated features
    features_df = pd.read_csv(feat_path, header=None)
    features_df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]

    edges_df   = pd.read_csv(edge_path)
    classes_df = pd.read_csv(classes_path)
    classes_df.columns = ["txId", "class"]

    print(f"[DataLoader] Loaded {len(features_df):,} transactions, "
          f"{len(edges_df):,} edges")
    return features_df, edges_df, classes_df


# ──────────────────────────────────────────
# 2.  Build PyG Graph Object
# ──────────────────────────────────────────

def build_graph(features_df, edges_df, classes_df):
    """
    Convert raw DataFrames into a PyTorch Geometric Data object.

    Labels:
        1 (illicit/fraud) → 1
        2 (licit/normal)  → 0
        unknown           → -1  (masked during training)
    """
    # ── Node features ──────────────────────────────────────
    node_ids = features_df["txId"].values
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
    X = features_df[feature_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Labels ─────────────────────────────────────────────
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes_df = classes_df.copy()
    classes_df["label"] = classes_df["class"].astype(str).map(label_map)
    label_dict = dict(zip(classes_df["txId"], classes_df["label"]))
    y = np.array([label_dict.get(nid, -1) for nid in node_ids], dtype=np.int64)

    # ── Edges ──────────────────────────────────────────────
    src = edges_df.iloc[:, 0].map(id_to_idx).dropna().astype(int).values
    dst = edges_df.iloc[:, 1].map(id_to_idx).dropna().astype(int).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── Train / test masks (only labelled nodes) ───────────
    labelled = y != -1
    indices  = np.where(labelled)[0]
    np.random.seed(42)
    np.random.shuffle(indices)
    split   = int(0.8 * len(indices))
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
# 3.  Fraud Ratio per Time-Step
# ──────────────────────────────────────────

def compute_fraud_ratio(features_df, classes_df):
    """
    Compute fraud_ratio = illicit_txns / total_labeled_txns per time step.
    Used as an extra feature fed into the LSTM price predictor.
    """
    merged = features_df[["txId", "time_step"]].merge(
        classes_df, on="txId", how="left"
    )
    merged["is_fraud"] = (merged["class"].astype(str) == "1").astype(int)
    merged["is_labeled"] = (merged["class"].astype(str) != "unknown").astype(int)

    stats = merged.groupby("time_step").agg(
        total_labeled=("is_labeled", "sum"),
        total_fraud=("is_fraud", "sum"),
    ).reset_index()
    stats["fraud_ratio"] = stats["total_fraud"] / stats["total_labeled"].replace(0, np.nan)
    stats["fraud_ratio"] = stats["fraud_ratio"].fillna(0)

    print(f"[DataLoader] Fraud ratio computed for {len(stats)} time steps "
          f"(mean={stats['fraud_ratio'].mean():.4f})")
    return stats


# ──────────────────────────────────────────
# 4.  Synthetic / Demo Price Data
# ──────────────────────────────────────────

def generate_synthetic_price_data(n_timesteps: int = 49, seed: int = 42):
    """
    Generate synthetic Bitcoin-like price data aligned with the 49
    time steps in the Elliptic dataset.  Replace with real CoinGecko
    data in production (see dashboard/live_data.py).
    """
    rng = np.random.default_rng(seed)
    prices = [30000.0]
    for _ in range(n_timesteps - 1):
        change = rng.normal(0, 0.03)          # ±3 % daily volatility
        prices.append(prices[-1] * (1 + change))
    prices = np.array(prices)

    df = pd.DataFrame({
        "time_step": range(1, n_timesteps + 1),
        "close":     prices,
        "volume":    rng.integers(1_000, 50_000, size=n_timesteps).astype(float),
        "high":      prices * (1 + rng.uniform(0, 0.02, n_timesteps)),
        "low":       prices * (1 - rng.uniform(0, 0.02, n_timesteps)),
    })
    print("[DataLoader] Synthetic price data generated "
          f"({n_timesteps} time steps)")
    return df

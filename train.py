"""
train.py
--------
FIX SUMMARY:
  OLD: Used generate_synthetic_price_data(49) → 49 fake data points at ~$30k
       Model and scaler fitted on wrong price range.
       Dashboard predictions showed $29k when BTC = $77k.

  NEW: Uses fetch_real_price_data(365) → 365 real daily BTC prices
       from CoinGecko API. Model and scaler fitted on current price range.
       Predictions will now be in the correct range.

  OLD: seq_len = 10  (too small)
  NEW: seq_len = 30  (better trend capture)

Run:
    python train.py
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# ── Project imports ─────────────────────────────────────────────────────────
from utils.data_loader import (
    load_elliptic_dataset,
    build_graph,
    compute_fraud_ratio,
    fetch_real_price_data,           # FIX: was generate_synthetic_price_data
)
from models.gnn_model import train_gnn, evaluate_gnn, predict_fraud
from models.lstm_model import (
    prepare_lstm_data,
    train_lstm,
    evaluate_lstm,
    FEATURE_COLS,
)
from utils.explainability import explain_lstm_shap


# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR      = "data/elliptic_bitcoin_dataset"
USE_REAL_DATA = os.path.exists(DATA_DIR)

GNN_EPOCHS   = 100
GNN_HIDDEN   = 128
LSTM_SEQ_LEN = 30      # FIX: was 10. More context = better predictions.
LSTM_EPOCHS  = 150     # FIX: was 100. EarlyStopping will stop early if needed.
LSTM_UNITS   = 64

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs",      exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load / Build Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 – Data Loading")
print("="*60)

# ── FIX: Always fetch real price data, regardless of Elliptic availability ──
# This ensures the LSTM scaler is fitted on current price levels ($40k-$80k),
# not old synthetic data starting at $30k.
print("\n[Train] Fetching real BTC price data from CoinGecko ...")
price_df = fetch_real_price_data(days=365)
print(f"[Train] Price data: {len(price_df)} daily rows")
print(f"[Train] Price range: ${price_df['close'].min():,.0f} – ${price_df['close'].max():,.0f}")

if USE_REAL_DATA:
    features_df, edges_df, classes_df = load_elliptic_dataset(DATA_DIR)
    graph_data, feat_scaler           = build_graph(features_df, edges_df, classes_df)
    fraud_ratio_df                    = compute_fraud_ratio(features_df, classes_df)

    # Map fraud ratio time steps (1-49) to price time steps (1-365)
    # We spread 49 fraud ratios across 365 price days by repeating
    n_price_steps = len(price_df)
    n_fraud_steps = len(fraud_ratio_df)

    # Repeat each fraud ratio for ~7 days (365/49 ≈ 7.4)
    extended_ratios = np.interp(
        np.linspace(0, n_fraud_steps - 1, n_price_steps),
        np.arange(n_fraud_steps),
        fraud_ratio_df["fraud_ratio"].values,
    )
    fraud_ratio_df_extended = pd.DataFrame({
        "time_step"  : range(1, n_price_steps + 1),
        "fraud_ratio": extended_ratios,
    })

else:
    import pandas as pd
    print("\n[Warning] Elliptic dataset not found — using synthetic GNN fraud data.")
    print("  Download: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    print("  Place in: data/elliptic_bitcoin_dataset/\n")

    # Synthetic graph for GNN demo
    from torch_geometric.data import Data
    N = 500
    torch.manual_seed(42)
    x          = torch.randn(N, 164)
    edge_index = torch.randint(0, N, (2, 2000))
    y          = torch.randint(0, 2, (N,))
    mask       = torch.ones(N, dtype=torch.bool)
    train_mask = mask.clone(); train_mask[400:] = False
    test_mask  = ~train_mask
    graph_data = Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, test_mask=test_mask
    )

    # ── FIX: Synthetic fraud ratios now aligned to real price data length ──
    n_price_steps = len(price_df)
    np.random.seed(42)
    fraud_ratio_df_extended = pd.DataFrame({
        "time_step"  : range(1, n_price_steps + 1),
        "fraud_ratio": np.random.uniform(0.01, 0.2, n_price_steps),
    })
    fraud_ratio_df = fraud_ratio_df_extended  # for consistency


# Sync price_df time_step with fraud_ratio time_step
price_df = price_df.copy()
price_df["time_step"] = range(1, len(price_df) + 1)

# Use the extended fraud ratio df for LSTM
lstm_fraud_df = fraud_ratio_df_extended if USE_REAL_DATA else fraud_ratio_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – GNN Fraud Detection
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2 – GNN Fraud Detection Training")
print("="*60)

gnn_model, gnn_history = train_gnn(
    graph_data,
    epochs     = GNN_EPOCHS,
    hidden_dim = GNN_HIDDEN,
)
gnn_metrics = evaluate_gnn(gnn_model, graph_data)

torch.save(gnn_model.state_dict(), "saved_models/gnn_model.pt")
print("[Train] GNN saved → saved_models/gnn_model.pt")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(gnn_history["train_loss"], color="crimson")
plt.title("GNN Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(gnn_history["train_acc"], color="steelblue")
plt.title("GNN Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("outputs/gnn_training_curve.png", dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Fraud Ratio
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 – Fraud Ratio Feature")
print("="*60)
print(lstm_fraud_df.head(10))
print(f"[Train] Fraud ratio aligned to {len(lstm_fraud_df)} price time steps")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – LSTM Price Prediction
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4 – Fraud-Aware LSTM Training")
print("="*60)

X_train, X_test, y_train, y_test, scaler, merged_df = prepare_lstm_data(
    price_df,
    lstm_fraud_df,
    seq_len = LSTM_SEQ_LEN,   # FIX: 30 instead of 10
)

print(f"[Train] Training samples : {len(X_train)}")
print(f"[Train] Test samples     : {len(X_test)}")

lstm_model, lstm_history = train_lstm(
    X_train, y_train,
    seq_len    = LSTM_SEQ_LEN,
    n_features = len(FEATURE_COLS),
    epochs     = LSTM_EPOCHS,
    units      = LSTM_UNITS,
)
lstm_metrics = evaluate_lstm(lstm_model, X_test, y_test, scaler)

# Save LSTM + scaler
lstm_model.save("saved_models/lstm_model.h5")
with open("saved_models/lstm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ── FIX: Also save seq_len so dashboard can load the correct window size ──
with open("saved_models/lstm_config.pkl", "wb") as f:
    pickle.dump({"seq_len": LSTM_SEQ_LEN, "n_features": len(FEATURE_COLS)}, f)

print("[Train] LSTM + scaler + config saved to saved_models/")

# Plot predictions
plt.figure(figsize=(12, 5))
plt.plot(lstm_metrics["y_test_real"],  label="Actual",    color="steelblue", linewidth=2)
plt.plot(lstm_metrics["y_pred_real"],  label="Predicted", color="crimson", linestyle="--", linewidth=2)
plt.title("LSTM Price Prediction vs Actual (Real BTC Data)")
plt.xlabel("Test Days"); plt.ylabel("BTC Price (USD)")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/lstm_predictions.png", dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 5 – SHAP Explainability")
print("="*60)

shap_values = explain_lstm_shap(
    lstm_model,
    X_test,
    feature_names = FEATURE_COLS,
    save_path     = "outputs/shap_lstm_summary.png",
)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Train] ✅ All done! Summary:")
print(f"  GNN  Accuracy       : {gnn_metrics['accuracy']:.4f}")
print(f"  GNN  ROC-AUC        : {gnn_metrics['roc_auc']:.4f}")
print(f"  LSTM RMSE           : ${lstm_metrics['rmse']:,.2f}")
print(f"  LSTM MAE            : ${lstm_metrics['mae']:,.2f}")
print(f"  LSTM MAPE           : {lstm_metrics['mape']:.2f}%")
print(f"  LSTM NRMSE          : {lstm_metrics['nrmse']:.4f}")
print(f"  LSTM Dir. Accuracy  : {lstm_metrics['dir_accuracy']:.2%}")
print(f"\n  Price data used     : {len(price_df)} daily BTC prices")
print(f"  Price range trained : ${price_df['close'].min():,.0f} – ${price_df['close'].max():,.0f}")
print("\n  Saved files:")
print("    saved_models/gnn_model.pt")
print("    saved_models/lstm_model.h5")
print("    saved_models/lstm_scaler.pkl")
print("    saved_models/lstm_config.pkl   ← NEW: stores seq_len")
print("    outputs/gnn_training_curve.png")
print("    outputs/lstm_predictions.png")
print("    outputs/shap_lstm_summary.png")

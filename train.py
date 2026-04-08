"""
train.py
--------
End-to-end training pipeline:
  1. Load Elliptic dataset
  2. Train GCN fraud detector
  3. Compute fraud ratio
  4. Train fraud-aware LSTM price predictor
  5. Generate SHAP explanations
  6. Save models to disk

Run:
    python train.py
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Project imports ────────────────────────────────────────────────────────────
from utils.data_loader import (
    load_elliptic_dataset,
    build_graph,
    compute_fraud_ratio,
    generate_synthetic_price_data,
)
from models.gnn_model  import train_gnn, evaluate_gnn, predict_fraud
from models.lstm_model import (
    prepare_lstm_data,
    train_lstm,
    evaluate_lstm,
    FEATURE_COLS,
)
from utils.explainability import explain_lstm_shap


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/elliptic_bitcoin_dataset"
USE_REAL_DATA = os.path.exists(DATA_DIR)          # falls back to synthetic if absent

GNN_EPOCHS   = 100
GNN_HIDDEN   = 128
LSTM_SEQ_LEN = 10
LSTM_EPOCHS  = 100
LSTM_UNITS   = 64

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs",      exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load / Build Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 – Data Loading")
print("="*60)

if USE_REAL_DATA:
    features_df, edges_df, classes_df = load_elliptic_dataset(DATA_DIR)
    graph_data, feat_scaler = build_graph(features_df, edges_df, classes_df)
    fraud_ratio_df          = compute_fraud_ratio(features_df, classes_df)
    price_df                = generate_synthetic_price_data(
                                  n_timesteps=int(features_df["time_step"].max()))
    features_df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]
else:
    print("[Warning] Elliptic dataset not found – using synthetic data for demo.")
    print("  Download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    print("  Place in:      data/elliptic_bitcoin_dataset/\n")

    # Synthetic graph (small demo)
    from torch_geometric.data import Data
    N = 500
    torch.manual_seed(42)
    x          = torch.randn(N, 164)
    edge_index = torch.randint(0, N, (2, 2000))
    y          = torch.randint(0, 2, (N,))
    mask       = torch.ones(N, dtype=torch.bool)
    train_mask = mask.clone(); train_mask[400:] = False
    test_mask  = ~train_mask
    graph_data = Data(x=x, edge_index=edge_index, y=y,
                      train_mask=train_mask, test_mask=test_mask)

    import pandas as pd
    n_steps        = 49
    time_steps     = list(range(1, n_steps + 1))
    fraud_ratio_df = pd.DataFrame({
        "time_step":   time_steps,
        "fraud_ratio": np.random.uniform(0.01, 0.2, n_steps),
    })
    price_df = generate_synthetic_price_data(n_timesteps=n_steps)


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

# Save GNN
torch.save(gnn_model.state_dict(), "saved_models/gnn_model.pt")
print("[Train] GNN saved → saved_models/gnn_model.pt")

# Plot GNN training curve
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
print("[Train] GNN training curve → outputs/gnn_training_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Fraud Ratio (bridge feature)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 – Fraud Ratio Feature")
print("="*60)
print(fraud_ratio_df.head())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – LSTM Price Prediction
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4 – Fraud-Aware LSTM Training")
print("="*60)

X_train, X_test, y_train, y_test, scaler, merged_df = prepare_lstm_data(
    price_df, fraud_ratio_df, seq_len=LSTM_SEQ_LEN
)

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
print("[Train] LSTM saved → saved_models/lstm_model.h5")

# Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(lstm_metrics["y_test_real"],  label="Actual",    color="steelblue")
plt.plot(lstm_metrics["y_pred_real"],  label="Predicted", color="crimson",  linestyle="--")
plt.title("LSTM Price Prediction vs Actual")
plt.xlabel("Time Step"); plt.ylabel("BTC Price (USD)")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/lstm_predictions.png", dpi=150)
plt.close()
print("[Train] LSTM predictions plot → outputs/lstm_predictions.png")


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

print("\n[Train] All done! Summary:")
print(f"  GNN  Accuracy  : {gnn_metrics['accuracy']:.4f}")
print(f"  GNN  ROC-AUC   : {gnn_metrics['roc_auc']:.4f}")
print(f"  LSTM RMSE      : {lstm_metrics['rmse']:.2f}")
print(f"  LSTM NRMSE     : {lstm_metrics['nrmse']:.4f}")
print("\n  Saved files:")
print("    saved_models/gnn_model.pt")
print("    saved_models/lstm_model.h5")
print("    saved_models/lstm_scaler.pkl")
print("    outputs/gnn_training_curve.png")
print("    outputs/lstm_predictions.png")
print("    outputs/shap_lstm_summary.png")

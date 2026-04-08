"""
utils/explainability.py
-----------------------
SHAP-based explainability for both the GNN and LSTM models.

Produces:
  - Feature importance bar charts
  - Waterfall plots for per-prediction explanations
  - Summary plots for global feature importance
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works in VS Code)


# ──────────────────────────────────────────
# 1.  SHAP for GNN (tabular approximation)
# ──────────────────────────────────────────

def explain_gnn_shap(model, data, feature_names=None, n_background=100,
                     n_explain=50, save_path="shap_gnn_summary.png"):
    """
    Compute SHAP values for GNN fraud predictions using a
    KernelExplainer (model-agnostic, works with any GNN).

    Parameters
    ----------
    model        : trained GCNFraudDetector
    data         : torch_geometric.data.Data
    feature_names: list of str – names for the 164 features
    n_background : int – background samples for KernelSHAP
    n_explain    : int – samples to explain
    save_path    : str – output image path

    Returns
    -------
    shap_values  : np.ndarray
    explainer    : shap.KernelExplainer
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    # Convert node features to numpy for SHAP
    X_np = data.x.cpu().numpy()
    edge_index = data.edge_index

    # Prediction function: takes numpy array, returns fraud probabilities
    def predict_fn(X_batch):
        x_t = torch.tensor(X_batch, dtype=torch.float).to(device)
        # Replace only the nodes we're explaining; keep graph structure
        # (approximation: treat as fully independent for SHAP)
        with torch.no_grad():
            out  = model(x_t, edge_index.to(device)[:, :x_t.shape[0]])
            prob = torch.exp(out)[:, 1].cpu().numpy()
        return prob

    # Sample background and explain sets from labelled nodes
    labelled_idx = np.where(data.y.cpu().numpy() != -1)[0]
    np.random.seed(42)
    bg_idx  = np.random.choice(labelled_idx, min(n_background, len(labelled_idx)), replace=False)
    exp_idx = np.random.choice(labelled_idx, min(n_explain,    len(labelled_idx)), replace=False)

    background = X_np[bg_idx]
    explain_X  = X_np[exp_idx]

    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_X, nsamples=100)

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X_np.shape[1])]

    # ── Summary Plot ──────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explain_X,
                      feature_names=feature_names,
                      show=False, plot_type="bar")
    plt.title("GNN SHAP Feature Importance (Fraud Detection)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] GNN summary plot saved → {save_path}")

    return shap_values, explainer


# ──────────────────────────────────────────
# 2.  SHAP for LSTM
# ──────────────────────────────────────────

def explain_lstm_shap(lstm_model, X_test, feature_names=None,
                      n_background=50, save_path="shap_lstm_summary.png"):
    """
    Compute SHAP values for LSTM price predictions.
    Uses DeepExplainer (fast, native TF/Keras support).

    Note: X_test has shape (samples, seq_len, n_features).
    We flatten the last two dims for SHAP display.

    Returns
    -------
    shap_values  : np.ndarray (samples, seq_len, n_features)
    """
    from models.lstm_model import FEATURE_COLS

    background = X_test[:n_background]
    explain_X  = X_test[n_background: n_background + 30]

    if len(explain_X) == 0:
        explain_X = X_test[:30]

    explainer   = shap.GradientExplainer(lstm_model, background)
    shap_values = explainer.shap_values(explain_X)

    # Average importance across time steps
    mean_shap = np.abs(shap_values[0]).mean(axis=(0, 1))   # (n_features,)

    if feature_names is None:
        feature_names = FEATURE_COLS

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, mean_shap, color="#1f77b4")
    plt.xlabel("Mean |SHAP value|")
    plt.title("LSTM SHAP Feature Importance (Price Prediction)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] LSTM summary plot saved → {save_path}")

    return shap_values


# ──────────────────────────────────────────
# 3.  Waterfall plot for a single prediction
# ──────────────────────────────────────────

def waterfall_plot(shap_values, X_sample, feature_names=None,
                   expected_value=0.0, idx=0,
                   save_path="shap_waterfall.png"):
    """
    Waterfall plot showing feature contributions for a single prediction.

    Parameters
    ----------
    shap_values   : np.ndarray (samples, features) – from KernelExplainer
    X_sample      : np.ndarray (samples, features)
    expected_value: float – base value (model average output)
    idx           : int   – which sample to explain
    """
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X_sample.shape[1])]

    explanation = shap.Explanation(
        values    = shap_values[idx],
        base_values = expected_value,
        data      = X_sample[idx],
        feature_names = feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP Waterfall – Sample {idx}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] Waterfall plot saved → {save_path}")

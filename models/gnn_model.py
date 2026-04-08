"""
models/gnn_model.py
-------------------
Graph Convolutional Network (GCN) for binary fraud classification.

Architecture:
  Input → GCNConv(hidden) → ReLU → Dropout
        → GCNConv(hidden) → ReLU → Dropout
        → Linear(2)  →  Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np


# ──────────────────────────────────────────
# 1.  Model Definition
# ──────────────────────────────────────────

class GCNFraudDetector(nn.Module):
    """
    Two-layer Graph Convolutional Network.

    Parameters
    ----------
    in_channels  : int  – number of input node features
    hidden_dim   : int  – hidden layer size  (default 128)
    dropout      : float – dropout probability (default 0.5)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc      = nn.Linear(hidden_dim // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        """Return node embeddings (before final FC layer) for SHAP."""
        with torch.no_grad():
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        return x


# ──────────────────────────────────────────
# 2.  Training Loop
# ──────────────────────────────────────────

def train_gnn(data, epochs: int = 100, lr: float = 0.01,
              hidden_dim: int = 128, dropout: float = 0.5,
              weight_decay: float = 5e-4, verbose: bool = True):
    """
    Train the GCN on the Elliptic dataset graph.

    Parameters
    ----------
    data       : torch_geometric.data.Data  – full graph
    epochs     : int   – training epochs
    lr         : float – Adam learning rate
    hidden_dim : int   – GCN hidden size
    dropout    : float – dropout probability
    verbose    : bool  – print progress every 10 epochs

    Returns
    -------
    model      : trained GCNFraudDetector
    history    : dict with 'train_loss' and 'train_acc' lists
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN] Training on {device}")

    model = GCNFraudDetector(
        in_channels=data.x.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    data = data.to(device)

    data.x = data.x.float()
    data.y = data.y.long()
    

    # Class-weight for imbalanced dataset (fraud is rare)
    train_labels = data.y[data.train_mask].cpu().numpy()
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    history = {"train_loss": [], "train_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = out[data.train_mask].argmax(dim=1)
        acc  = (pred == data.y[data.train_mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc)

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss={loss.item():.4f} | Acc={acc:.4f}")

    print("[GNN] Training complete!")
    return model, history


# ──────────────────────────────────────────
# 3.  Evaluation
# ──────────────────────────────────────────

def evaluate_gnn(model, data):
    """
    Evaluate the trained GCN on the test split.

    Returns
    -------
    metrics : dict with accuracy, f1, roc_auc, report
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        out   = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.exp(out)                            # convert log-softmax → prob

    mask    = data.test_mask.to(device)
    y_true  = data.y[mask].cpu().numpy()
    y_pred  = probs[mask].argmax(dim=1).cpu().numpy()
    y_prob  = probs[mask][:, 1].cpu().numpy()            # fraud probability

    report  = classification_report(y_true, y_pred,
                                    target_names=["Normal", "Fraud"])
    roc_auc = roc_auc_score(y_true, y_prob)
    acc     = (y_pred == y_true).mean()

    print("\n[GNN Evaluation]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(report)

    return {
        "accuracy": acc,
        "roc_auc":  roc_auc,
        "report":   report,
        "y_true":   y_true,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
    }


# ──────────────────────────────────────────
# 4.  Predict on New / Full Graph
# ──────────────────────────────────────────

def predict_fraud(model, data):
    """
    Run inference on all nodes and return fraud probabilities.

    Returns
    -------
    fraud_probs : np.ndarray of shape (N,) – probability of fraud per node
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        out  = model(data.x.to(device), data.edge_index.to(device))
        prob = torch.exp(out)[:, 1].cpu().numpy()

    return prob

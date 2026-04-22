# Fraud-Aware Explainable GNN — MTech Project

> **GNN (fraud detection) + LSTM (price prediction) + SHAP (explainability) + Streamlit (dashboard)**

Requires:
Python 3.10.2: Download Link: https://www.python.org/ftp/python/3.10.2/python-3.10.2-amd64.exe

---

## Project Structure

```
gnn_fraud_detection/
│
├── data/
│   └── elliptic_bitcoin_dataset/      ← place dataset CSVs here
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_edgelist.csv
│       └── elliptic_txs_classes.csv
│
├── models/
│   ├── gnn_model.py                   ← GCN fraud detector
│   └── lstm_model.py                  ← Fraud-aware LSTM price predictor
│
├── utils/
│   ├── data_loader.py                 ← Dataset loading & preprocessing
│   └── explainability.py             ← SHAP explanations
│
├── dashboard/
│   └── app.py                         ← Streamlit real-time dashboard
│
├── saved_models/                      ← auto-created after training
├── outputs/                           ← plots saved here
│
├── train.py                           ← main training script
├── requirements.txt
└── README.md
```

---

## Step-by-Step Setup in VS Code

### STEP 1 — Install Python

1. Open your browser → go to https://www.python.org/downloads/
2. Download **Python 3.10** (recommended for TensorFlow compatibility)
3. Run the installer → **tick "Add Python to PATH"** → click Install Now
4. Open **Command Prompt** and verify:
   ```
   python --version
   ```
   You should see: `Python 3.10.x`

---

### STEP 2 — Install VS Code

1. Go to https://code.visualstudio.com/ → Download for Windows
2. Install it (all defaults are fine)
3. Open VS Code
4. Press `Ctrl+Shift+X` → search **"Python"** → install the Microsoft Python extension
5. Search **"Pylance"** → install it too

---

### STEP 3 — Open the Project in VS Code

1. Extract the project ZIP (or copy the folder) to somewhere like `C:\Users\YourName\gnn_fraud_detection`
2. In VS Code: **File → Open Folder** → select the `gnn_fraud_detection` folder
3. VS Code will ask "Do you trust this workspace?" → click **Yes**

---

### STEP 4 — Create a Virtual Environment

This keeps all libraries isolated for this project.

1. In VS Code press **Ctrl+` ** (backtick) to open the terminal
2. Run these commands one by one:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) appear at the start of the terminal line
```

> **Mac/Linux:** use `source venv/bin/activate` instead

---

### STEP 5 — Install All Libraries

With the virtual environment active, run:

```bash
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.4.0
pip install tensorflow==2.14.0
pip install scikit-learn pandas numpy shap streamlit matplotlib seaborn plotly requests
```

> **If you have an NVIDIA GPU:** replace the torch install line with the CUDA version from https://pytorch.org/get-started/locally/

Verify installations:
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import torch_geometric; print('PyG OK')"
python -c "import streamlit; print('Streamlit OK')"
```

---

### STEP 6 — Download the Elliptic Dataset

1. Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Create a free Kaggle account if you don't have one
3. Click **Download** → extract the ZIP
4. Copy these 3 files into `data/elliptic_bitcoin_dataset/`:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_classes.csv`

> **No dataset?** The code will automatically use **synthetic data** for demo purposes. You can still run everything and see the full pipeline working.

---

### STEP 7 — Select the Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` → type **"Python: Select Interpreter"**
2. Choose the one that says `venv` or shows the path `.\venv\Scripts\python.exe`
3. VS Code will now use your virtual environment automatically

---

### STEP 8 — Train the Models

In the terminal (make sure `(venv)` is active):

```bash
python train.py
```

You will see:
```
STEP 1 – Data Loading
[DataLoader] Loaded 203,769 transactions, 234,355 edges

STEP 2 – GNN Fraud Detection Training
[GNN] Training on cpu
  Epoch  10/100 | Loss=0.4821 | Acc=0.8763
  ...
  Epoch 100/100 | Loss=0.1234 | Acc=0.9740

STEP 3 – Fraud Ratio Feature
STEP 4 – Fraud-Aware LSTM Training
STEP 5 – SHAP Explainability

[Train] All done!
  GNN  Accuracy  : 0.9740
  GNN  ROC-AUC   : 0.9812
  LSTM RMSE      : ...
```

Training time: ~10–30 min on CPU, ~3–5 min on GPU.

**Output files created:**
- `saved_models/gnn_model.pt`
- `saved_models/lstm_model.h5`
- `saved_models/lstm_scaler.pkl`
- `outputs/gnn_training_curve.png`
- `outputs/lstm_predictions.png`
- `outputs/shap_lstm_summary.png`

---

### STEP 9 — Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

VS Code will show a link like:
```
  Local URL: http://localhost:8501
```

Your browser will open automatically showing the **live dashboard** with:
- 🪙 Live BTC price (if internet available)
- 📈 Price forecast chart
- 🚨 Fraud ratio alerts
- 🔍 SHAP feature importance
- 📋 Model performance comparison

---

### STEP 10 — Using VS Code Run Buttons (Alternative)

The `.vscode/launch.json` file is already set up. Use it like this:

1. Press `F5` or go to **Run → Start Debugging**
2. Select **"1. Train GNN + LSTM"** from the dropdown → trains the models
3. After training, select **"2. Run Streamlit Dashboard"** → opens dashboard

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: torch_geometric` | Run `pip install torch-geometric` with venv active |
| `No module named 'tensorflow'` | Run `pip install tensorflow==2.14.0` |
| `CUDA out of memory` | Add `os.environ["CUDA_VISIBLE_DEVICES"] = ""` at top of train.py |
| `FileNotFoundError: elliptic_txs_features.csv` | Place dataset files in `data/elliptic_bitcoin_dataset/` or delete the folder to use synthetic data |
| Streamlit not opening | Go to http://localhost:8501 manually in browser |
| `venv\Scripts\activate` not recognized | Use PowerShell and run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Architecture Summary

```
Elliptic Dataset (203K Bitcoin transactions)
        │
        ▼
  Graph Construction (PyTorch Geometric)
  Nodes = Transactions │ Edges = Fund Transfers
        │
        ▼
  GCN Fraud Detector ─────────────────────┐
  (2-layer GCN + FC)                      │
  Output: fraud/normal per node           │
        │                                 │
        ▼                                 ▼
  Fraud Ratio per Time Step ──────► LSTM Price Predictor
  (novel bridge feature)               [close, vol, high,
                                        low, fraud_ratio]
                                              │
                                              ▼
                                    SHAP Explainability
                                    (feature importance)
                                              │
                                              ▼
                                   Streamlit Dashboard
                                   (real-time alerts)
```

---

## References

- Elliptic Dataset: Weber et al., 2019
- GCN: Kipf & Welling, 2017
- SHAP: Lundberg & Lee, 2017
- PyTorch Geometric: Fey & Lenssen, 2019

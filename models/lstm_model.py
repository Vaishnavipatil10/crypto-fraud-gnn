"""
models/lstm_model.py
--------------------
LSTM time-series model for cryptocurrency price prediction.

Key novelty: includes the 'fraud_ratio' feature (computed from GNN output)
as an additional input to make predictions "fraud-aware".

Architecture:
  [close, volume, high, low, fraud_ratio]  →  LSTM(units) × n_layers
  →  Dense(1)  →  predicted_close
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────
# 1.  Data Preparation
# ──────────────────────────────────────────

FEATURE_COLS = ["close", "volume", "high", "low", "fraud_ratio"]
TARGET_COL   = "close"


def prepare_lstm_data(price_df: pd.DataFrame, fraud_ratio_df: pd.DataFrame,
                      seq_len: int = 10):
    """
    Merge price data with fraud ratio, scale, and create sliding windows.

    Parameters
    ----------
    price_df       : DataFrame with columns [time_step, close, volume, high, low]
    fraud_ratio_df : DataFrame with columns [time_step, fraud_ratio]
    seq_len        : look-back window length (default 10)

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted MinMaxScaler (for inverse transform)
    """
    df = price_df.merge(fraud_ratio_df[["time_step", "fraud_ratio"]],
                        on="time_step", how="left")
    df["fraud_ratio"] = df["fraud_ratio"].fillna(0)
    df = df.sort_values("time_step").reset_index(drop=True)

    # Scale all features to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])

    # Build sliding windows
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len: i])          # (seq_len, n_features)
        y.append(scaled[i, FEATURE_COLS.index(TARGET_COL)])

    X, y = np.array(X), np.array(y)

    split  = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[LSTM] Prepared data → "
          f"train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, df


# ──────────────────────────────────────────
# 2.  Model Definition
# ──────────────────────────────────────────

def build_lstm_model(seq_len: int, n_features: int,
                     units: int = 64, n_layers: int = 2,
                     dropout: float = 0.2):
    """
    Build a stacked LSTM model.

    Parameters
    ----------
    seq_len    : int   – look-back window
    n_features : int   – number of input features (5 with fraud_ratio)
    units      : int   – LSTM units per layer
    n_layers   : int   – number of stacked LSTM layers
    dropout    : float – dropout rate
    """
    model = Sequential(name="FraudAware_LSTM")

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)          # return sequences for all but last
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=(seq_len, n_features)))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))                           # single output: next close price

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    model.summary()
    return model


# ──────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────

def train_lstm(X_train, y_train, seq_len: int, n_features: int,
               epochs: int = 100, batch_size: int = 16,
               units: int = 64, n_layers: int = 2, dropout: float = 0.2):
    """
    Train the LSTM model with early stopping.

    Returns
    -------
    model   : trained Keras model
    history : Keras History object
    """
    model = build_lstm_model(seq_len, n_features, units, n_layers, dropout)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    print("[LSTM] Training complete!")
    return model, history


# ──────────────────────────────────────────
# 4.  Evaluation
# ──────────────────────────────────────────

def evaluate_lstm(model, X_test, y_test, scaler):
    """
    Evaluate LSTM on the test set; inverse-transform for real-price metrics.

    Returns
    -------
    metrics : dict with rmse, mae, y_pred_real, y_test_real
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform: rebuild full feature array for scaler
    close_idx = FEATURE_COLS.index(TARGET_COL)
    dummy     = np.zeros((len(y_pred_scaled), len(FEATURE_COLS)))
    dummy[:, close_idx] = y_pred_scaled
    y_pred_real = scaler.inverse_transform(dummy)[:, close_idx]

    dummy2 = np.zeros((len(y_test), len(FEATURE_COLS)))
    dummy2[:, close_idx] = y_test
    y_test_real = scaler.inverse_transform(dummy2)[:, close_idx]

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)

    # Normalized RMSE (NRMSE)
    nrmse = rmse / (y_test_real.max() - y_test_real.min())

    print(f"\n[LSTM Evaluation]")
    print(f"  RMSE  : {rmse:.2f}")
    print(f"  MAE   : {mae:.2f}")
    print(f"  NRMSE : {nrmse:.4f}")

    return {
        "rmse":        rmse,
        "mae":         mae,
        "nrmse":       nrmse,
        "y_pred_real": y_pred_real,
        "y_test_real": y_test_real,
    }


# ──────────────────────────────────────────
# 5.  Multi-Step Forecast
# ──────────────────────────────────────────

def forecast_future(model, last_sequence: np.ndarray, scaler,
                    steps: int = 7):
    """
    Iteratively predict `steps` future prices.

    Parameters
    ----------
    last_sequence : np.ndarray of shape (seq_len, n_features) – last known window
    scaler        : fitted MinMaxScaler
    steps         : number of future time steps to predict

    Returns
    -------
    future_prices : list of predicted prices (real scale)
    """
    close_idx = FEATURE_COLS.index(TARGET_COL)
    seq = last_sequence.copy()
    future_prices = []

    for _ in range(steps):
        pred_scaled = model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]

        # Build inverse-transform array
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, close_idx] = pred_scaled
        price = scaler.inverse_transform(dummy)[0, close_idx]
        future_prices.append(price)

        # Roll window forward: shift left and append new step
        new_step       = seq[-1].copy()
        new_step[close_idx] = pred_scaled
        seq = np.vstack([seq[1:], new_step])

    return future_prices

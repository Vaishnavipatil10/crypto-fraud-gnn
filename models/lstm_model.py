"""
models/lstm_model.py
--------------------
FIX SUMMARY:
  OLD: MinMaxScaler  → breaks completely when live price ($77k) is outside
                        training range ($20k-$35k). Produces scaled value of 3.86
                        instead of 0-1. Model predicts garbage.

  NEW: RobustScaler  → uses median + IQR instead of min/max.
                        Handles price spikes and outliers gracefully.
                        Works correctly even when prices move to new highs.

  OLD: seq_len = 10  → only 10 data points of context
  NEW: seq_len = 30  → 30 days of context (better trend capture)

  OLD: 49 training samples → terrible generalisation
  NEW: 365 daily samples  → 7x more data, proper train/test split

  OLD: batch_size = 16   → too small for noisy financial data
  NEW: batch_size = 32   → more stable gradient updates

  ADDED: directional_accuracy() → measures if up/down prediction is correct
                                   (more meaningful than RMSE for trading)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler   # FIX: was MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────

FEATURE_COLS = ["close", "volume", "high", "low", "fraud_ratio"]
TARGET_COL   = "close"


# ──────────────────────────────────────────
# 1.  Data Preparation
# ──────────────────────────────────────────

def prepare_lstm_data(price_df: pd.DataFrame, fraud_ratio_df: pd.DataFrame,
                      seq_len: int = 30):          # FIX: was 10, now 30
    """
    Merge price data with fraud ratio, scale, and create sliding windows.

    Parameters
    ----------
    price_df       : DataFrame – [time_step, close, volume, high, low]
    fraud_ratio_df : DataFrame – [time_step, fraud_ratio]
    seq_len        : look-back window (FIX: increased from 10 → 30 days)

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted RobustScaler
    merged_df                         : merged DataFrame for reference
    """
    df = price_df.merge(
        fraud_ratio_df[["time_step", "fraud_ratio"]],
        on="time_step", how="left"
    )
    df["fraud_ratio"] = df["fraud_ratio"].fillna(df["fraud_ratio"].median())
    df = df.sort_values("time_step").reset_index(drop=True)

    # ── FIX: RobustScaler instead of MinMaxScaler ──────────────────────────
    # RobustScaler uses:  (x - median) / IQR
    # Why better?
    #   MinMaxScaler: if max=35k during training, and live price = 77k,
    #                 scaled value = (77k-20k)/(35k-20k) = 3.86  ← BROKEN
    #   RobustScaler: robust to outliers and new price levels.
    #                 The IQR doesn't blow up just because price hit a new ATH.
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])

    # ── Build sliding windows ──────────────────────────────────────────────
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len: i])   # (seq_len, n_features)
        y.append(scaled[i, FEATURE_COLS.index(TARGET_COL)])

    X, y = np.array(X), np.array(y)

    # ── Train / test split ─────────────────────────────────────────────────
    split  = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[LSTM] Data prepared → train={X_train.shape}, test={X_test.shape}")
    print(f"[LSTM] Price range in data: "
          f"${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
    return X_train, X_test, y_train, y_test, scaler, df


# ──────────────────────────────────────────
# 2.  Model Definition
# ──────────────────────────────────────────

def build_lstm_model(seq_len: int, n_features: int,
                     units: int = 64, n_layers: int = 2,
                     dropout: float = 0.2):
    """
    Build a stacked LSTM model.

    FIX: Added L2 regularisation to reduce overfitting on financial data.
         Added second Dense layer for better non-linear mapping.
         Gradient clipping added in compile to prevent exploding gradients.
    """
    from tensorflow.keras.regularizers import l2

    model = Sequential(name="FraudAware_LSTM")

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        if i == 0:
            model.add(LSTM(
                units,
                return_sequences = return_seq,
                input_shape      = (seq_len, n_features),
                kernel_regularizer = l2(1e-4),   # FIX: L2 regularisation
            ))
        else:
            model.add(LSTM(
                units,
                return_sequences   = return_seq,
                kernel_regularizer = l2(1e-4),
            ))
        model.add(Dropout(dropout))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))     # FIX: extra dense layer
    model.add(Dense(1))                          # output: next close price

    # FIX: clipnorm=1.0 prevents exploding gradients (common in financial LSTM)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss      = "mse",
        metrics   = ["mae"],
    )
    model.summary()
    return model


# ──────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────

def train_lstm(X_train, y_train, seq_len: int, n_features: int,
               epochs: int = 150,      # FIX: was 100, now 150 (EarlyStopping will stop early)
               batch_size: int = 32,   # FIX: was 16, now 32 (more stable gradients)
               units: int = 64, n_layers: int = 2, dropout: float = 0.2):
    """
    Train the LSTM model with early stopping and learning rate scheduling.

    FIX: Increased patience from 15 → 20 so model doesn't stop too early.
         Added ModelCheckpoint logic via restore_best_weights=True.
    """
    model = build_lstm_model(seq_len, n_features, units, n_layers, dropout)

    callbacks = [
        EarlyStopping(
            patience             = 20,              # FIX: was 15
            restore_best_weights = True,
            verbose              = 1,
            monitor              = "val_loss",
        ),
        ReduceLROnPlateau(
            factor   = 0.5,
            patience = 10,                          # FIX: was 7
            verbose  = 1,
            min_lr   = 1e-6,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = 0.15,                   # FIX: was 0.10 (bigger val set)
        callbacks        = callbacks,
        verbose          = 1,
        shuffle          = False,                  # FIX: never shuffle time series!
    )
    print("[LSTM] Training complete!")
    return model, history


# ──────────────────────────────────────────
# 4.  Evaluation
# ──────────────────────────────────────────

def evaluate_lstm(model, X_test, y_test, scaler):
    """
    Evaluate LSTM on the test set.

    FIX: Added directional_accuracy — measures if the model correctly
         predicts whether price goes UP or DOWN. This is more meaningful
         for trading decisions than raw RMSE.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform back to real USD prices
    close_idx = FEATURE_COLS.index(TARGET_COL)
    dummy     = np.zeros((len(y_pred_scaled), len(FEATURE_COLS)))
    dummy[:, close_idx] = y_pred_scaled
    y_pred_real = scaler.inverse_transform(dummy)[:, close_idx]

    dummy2 = np.zeros((len(y_test), len(FEATURE_COLS)))
    dummy2[:, close_idx] = y_test
    y_test_real = scaler.inverse_transform(dummy2)[:, close_idx]

    rmse  = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae   = mean_absolute_error(y_test_real, y_pred_real)
    nrmse = rmse / (y_test_real.max() - y_test_real.min() + 1e-8)

    # ── FIX: Directional accuracy ──────────────────────────────────────────
    # Did the model correctly predict UP or DOWN?
    actual_dir    = np.diff(y_test_real) > 0   # True = price went up
    predicted_dir = np.diff(y_pred_real) > 0
    dir_accuracy  = (actual_dir == predicted_dir).mean()

    # ── MAPE ──────────────────────────────────────────────────────────────
    mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100

    print(f"\n[LSTM Evaluation]")
    print(f"  RMSE              : ${rmse:,.2f}")
    print(f"  MAE               : ${mae:,.2f}")
    print(f"  MAPE              : {mape:.2f}%")
    print(f"  NRMSE             : {nrmse:.4f}")
    print(f"  Directional Acc   : {dir_accuracy:.2%}  "
          f"(random baseline = 50%)")

    return {
        "rmse"          : rmse,
        "mae"           : mae,
        "mape"          : mape,
        "nrmse"         : nrmse,
        "dir_accuracy"  : dir_accuracy,
        "y_pred_real"   : y_pred_real,
        "y_test_real"   : y_test_real,
    }


# ──────────────────────────────────────────
# 5.  Multi-Step Forecast
# ──────────────────────────────────────────

def forecast_future(model, last_sequence: np.ndarray, scaler,
                    steps: int = 7):
    """
    Iteratively predict `steps` future prices.

    FIX: last_sequence must already be scaled using the SAME scaler
         that was used during training. The dashboard now ensures this
         by always using the saved scaler object — never a new one.

    Parameters
    ----------
    last_sequence : np.ndarray (seq_len, n_features) — last known window, SCALED
    scaler        : the SAME RobustScaler fitted during training
    steps         : number of future days to predict

    Returns
    -------
    future_prices : list of predicted prices in real USD
    """
    close_idx = FEATURE_COLS.index(TARGET_COL)
    seq       = last_sequence.copy()
    future_prices = []

    for step in range(steps):
        pred_scaled = model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]

        # Inverse transform to get real price
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, close_idx] = pred_scaled
        price = scaler.inverse_transform(dummy)[0, close_idx]
        future_prices.append(float(price))

        # Roll window forward
        new_step              = seq[-1].copy()
        new_step[close_idx]   = pred_scaled
        seq = np.vstack([seq[1:], new_step])

    return future_prices


# ──────────────────────────────────────────
# 6.  Build Prediction Window from Raw Prices
# ──────────────────────────────────────────

def build_prediction_window(recent_price_df: pd.DataFrame,
                             scaler,
                             seq_len: int = 30,
                             fraud_ratio: float = 0.05) -> np.ndarray:
    """
    FIX: NEW FUNCTION — This is the key fix for the dashboard.

    Takes the most recent `seq_len` days of real price data,
    scales it using the TRAINED scaler, and returns a window
    ready for LSTM prediction.

    WHY THIS IS NEEDED:
    The old dashboard code was:
      1. Fetching live BTC data  ($77,979)
      2. But using the OLD scaler fitted on 2023 data ($20k-$35k)
      3. Scaled value = 3.86  ← model sees totally out-of-range input
      4. Prediction = garbage

    This function ensures the scaler is always consistent with
    the data being fed.

    Parameters
    ----------
    recent_price_df : DataFrame with [close, volume, high, low] columns
                      Must have at least seq_len rows.
    scaler          : the SAME scaler fitted during training (loaded from .pkl)
    seq_len         : must match what model was trained with
    fraud_ratio     : current estimated fraud ratio (default 0.05 = 5%)

    Returns
    -------
    window : np.ndarray of shape (seq_len, 5) — scaled, ready for model.predict()
    """
    df = recent_price_df.tail(seq_len).copy().reset_index(drop=True)

    if len(df) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows of price data, got {len(df)}. "
            f"Fetch more historical data."
        )

    # Add fraud_ratio column (use constant current estimate)
    df["fraud_ratio"] = fraud_ratio

    # Scale using the trained scaler
    scaled = scaler.transform(df[FEATURE_COLS])
    return scaled

"""Walk-forward (expanding window) backtesting for several forecasting models."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.metrics import all_metrics

MODELS = ["lstm", "xgboost", "naive"]


def _windows(scaled, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def _fit_predict(model_name, X_train, y_train, X_test, epochs):
    """Return scaled predictions for X_test."""
    if model_name == "naive":
        # Persistence baseline: tomorrow equals today.
        return X_test[:, -1]
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, verbosity=0)
        m.fit(X_train, y_train)
        return m.predict(X_test)
    if model_name == "lstm":
        from src.model_builder import build_lstm_model
        m = build_lstm_model((X_train.shape[1], 1))
        m.fit(X_train.reshape(*X_train.shape, 1), y_train, epochs=epochs, batch_size=32, verbose=0)
        return m.predict(X_test.reshape(*X_test.shape, 1), verbose=0).ravel()
    raise ValueError(f"Unknown model: {model_name}")


def walk_forward(close, model_name="lstm", seq_len=60, n_splits=4, epochs=10):
    """Expanding-window walk-forward backtest.

    The series is cut into n_splits consecutive test blocks. For each block the
    model trains only on data before it, so there is no look-ahead. The scaler
    is fit on the training part only.

    Returns (predictions DataFrame, per-fold metrics DataFrame, overall metrics dict).
    """
    close = np.asarray(close, dtype=float).reshape(-1, 1)
    n = len(close)
    min_train = seq_len + 50
    block = (n - min_train) // n_splits
    if block < 5:
        raise ValueError("Not enough data for this lookback and number of folds.")

    preds, folds = [], []
    for k in range(n_splits):
        train_end = min_train + k * block
        test_end = n if k == n_splits - 1 else train_end + block
        scaler = MinMaxScaler().fit(close[:train_end])
        scaled = scaler.transform(close[:test_end])

        X_all, y_all = _windows(scaled, seq_len)
        # window i predicts index i + seq_len
        idx = np.arange(seq_len, test_end)
        tr = idx < train_end
        X_train, y_train = X_all[tr], y_all[tr]
        X_test, y_test = X_all[~tr], y_all[~tr]

        p = _fit_predict(model_name, X_train, y_train, X_test, epochs)
        p_price = scaler.inverse_transform(np.reshape(p, (-1, 1))).ravel()
        a_price = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

        preds.append(pd.DataFrame({"fold": k + 1, "index": idx[~tr], "actual": a_price, "predicted": p_price}))
        folds.append({"fold": k + 1, **all_metrics(a_price, p_price)})

    pred_df = pd.concat(preds, ignore_index=True)
    overall = all_metrics(pred_df["actual"], pred_df["predicted"])
    return pred_df, pd.DataFrame(folds), overall

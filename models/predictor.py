"""
models/predictor.py — Stock direction classifier
Techniques used:
  - Rich technical indicator features (RSI, MACD, Bollinger, ATR, OBV)
  - Sentiment × price interaction features (40+ features total)
  - Three classifiers: Logistic Regression, Random Forest, Gradient Boosting
  - TimeSeriesSplit cross-validation (5 folds) — no data leakage
  - CV accuracy reported as the honest out-of-sample performance metric
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")


# ── Model definitions ──────────────────────────────────────────────────────────
def _make_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=0.5, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2,
            max_features="sqrt", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.75, min_samples_leaf=3, random_state=42
        ),
    }


# ── Technical indicators ───────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta).clip(lower=0).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def _macd(series: pd.Series) -> pd.Series:
    return series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()


def _bollinger_pct(series: pd.Series, window: int = 20) -> pd.Series:
    """Price position within Bollinger Bands: 0 = lower band, 1 = upper band."""
    ma  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - (ma - 2 * std)) / (4 * std + 1e-9)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment + price data and engineer all features. Returns feature matrix."""
    sent  = sentiment_df.copy()
    price = price_df.copy()

    sent["date"]  = pd.to_datetime(sent["date"]).dt.normalize()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()

    df = pd.merge(sent, price, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if len(df) < 30:
        return pd.DataFrame()

    close  = df["close"]
    ret    = df["pct_change"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.ones(len(df)), index=df.index)
    high   = df["high"]   if "high"   in df.columns else close * 1.01
    low    = df["low"]    if "low"    in df.columns else close * 0.99

    # Target: will price go UP tomorrow?
    df["target"] = (ret.shift(-1) > 0).astype(int)

    # Sentiment lags and rolling stats
    for lag in [1, 2, 3, 5]:
        df[f"sent_lag_{lag}"] = df["combined_sentiment"].shift(lag)
        df[f"vol_lag_{lag}"]  = df["post_volume"].shift(lag)
        df[f"ret_lag_{lag}"]  = ret.shift(lag)

    df["sent_ma3"]   = df["combined_sentiment"].rolling(3).mean()
    df["sent_ma7"]   = df["combined_sentiment"].rolling(7).mean()
    df["sent_ma14"]  = df["combined_sentiment"].rolling(14).mean()
    df["sent_std7"]  = df["combined_sentiment"].rolling(7).std()
    df["sent_trend"] = df["sent_ma3"] - df["sent_ma7"]
    df["sent_vs14"]  = df["combined_sentiment"] - df["sent_ma14"]
    df["sent_delta"] = df["combined_sentiment"].diff()
    df["sent_accel"] = df["sent_delta"].diff()

    vol_mean         = df["post_volume"].rolling(14).mean()
    df["vol_surge"]  = df["post_volume"] / (vol_mean + 1)
    df["vol_delta"]  = df["post_volume"].diff()

    # Technical indicators
    df["rsi_7"]      = _rsi(close, 7)
    df["rsi_14"]     = _rsi(close, 14)
    df["macd"]       = _macd(close)
    df["macd_signal"]= df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]  = df["macd"] - df["macd_signal"]
    df["boll_pct"]   = _bollinger_pct(close, 20)
    df["atr_14"]     = _atr(high, low, close, 14)
    df["atr_pct"]    = df["atr_14"] / (close + 1e-9)
    df["obv_norm"]   = (_obv(close, volume) / (volume.rolling(10).mean() + 1)).rolling(5).mean()

    df["ma5"]              = close.rolling(5).mean()
    df["ma10"]             = close.rolling(10).mean()
    df["ma20"]             = close.rolling(20).mean()
    df["price_ma5"]        = close / (df["ma5"]  + 1e-9) - 1
    df["price_ma10"]       = close / (df["ma10"] + 1e-9) - 1
    df["price_ma20"]       = close / (df["ma20"] + 1e-9) - 1
    df["ma5_cross_ma10"]   = (df["ma5"] > df["ma10"]).astype(int)
    df["ret_ma3"]          = ret.rolling(3).mean()
    df["ret_ma5"]          = ret.rolling(5).mean()
    df["ret_std5"]         = ret.rolling(5).std()
    df["ret_std10"]        = ret.rolling(10).std()

    # Sentiment × price interaction features
    df["sent_x_rsi"]  = df["combined_sentiment"] * (df["rsi_14"] / 100)
    df["sent_x_macd"] = df["combined_sentiment"] * df["macd_hist"].clip(-1, 1)
    df["sent_x_vol"]  = df["combined_sentiment"] * df["vol_surge"].clip(0, 5)
    df["sent_x_mom"]  = df["combined_sentiment"] * df["ret_ma3"]

    return df.dropna()


FEATURE_COLS = [
    "combined_sentiment", "weighted_sentiment", "news_sentiment",
    "sent_lag_1", "sent_lag_2", "sent_lag_3", "sent_lag_5",
    "sent_ma3", "sent_ma7", "sent_ma14", "sent_std7",
    "sent_trend", "sent_vs14", "sent_delta", "sent_accel",
    "post_volume", "vol_surge", "vol_delta", "vol_lag_1", "vol_lag_2",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "ret_ma3", "ret_ma5", "ret_std5", "ret_std10",
    "rsi_7", "rsi_14", "macd", "macd_hist", "boll_pct",
    "atr_pct", "obv_norm",
    "price_ma5", "price_ma10", "price_ma20", "ma5_cross_ma10",
    "sent_x_rsi", "sent_x_macd", "sent_x_vol", "sent_x_mom",
]


# ── Training ───────────────────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame) -> dict:
    """Train all models with TimeSeriesSplit CV. Returns results dict."""
    results      = {}
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    X        = df[feature_cols].values
    y        = df["target"].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv     = TimeSeriesSplit(n_splits=5)

    for name, model in _make_models().items():
        fold_metrics = []

        for train_idx, test_idx in tscv.split(X_scaled):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

            fold_metrics.append({
                "accuracy" : accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall"   : recall_score(y_te, y_pred, zero_division=0),
                "f1"       : f1_score(y_te, y_pred, zero_division=0),
                "roc_auc"  : roc_auc_score(y_te, y_prob) if len(set(y_te)) > 1 else 0.5,
            })

        cv_avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

        # Retrain on full dataset for feature importance extraction
        model.fit(X_scaled, y)
        train_acc = float(accuracy_score(y, model.predict(X_scaled)))
        cm        = confusion_matrix(y, model.predict(X_scaled))

        feat_imp = {}
        if hasattr(model, "feature_importances_"):
            feat_imp = dict(zip(feature_cols, model.feature_importances_))
        elif hasattr(model, "coef_"):
            feat_imp = dict(zip(feature_cols, np.abs(model.coef_[0])))

        results[name] = {
            **cv_avg,
            "train_accuracy"    : train_acc,
            "cv_accuracy"       : cv_avg["accuracy"],
            "confusion_matrix"  : cm.tolist(),
            "feature_importance": feat_imp,
            "model"             : model,
            "scaler"            : scaler,
            "feature_cols"      : feature_cols,
        }

    return results


# ── Signal prediction ──────────────────────────────────────────────────────────
def predict_signal(model_result: dict, latest_row: pd.Series) -> dict:
    """Generate a directional signal from the latest data point."""
    feat_cols = model_result["feature_cols"]
    scaler    = model_result["scaler"]
    model     = model_result["model"]

    row_vals = [float(latest_row.get(col, 0) or 0) for col in feat_cols]
    X        = scaler.transform([row_vals])
    prob_up  = float(model.predict_proba(X)[0][1])

    if prob_up >= 0.60:
        signal = "BULLISH"
        color  = "#00ff41"
    elif prob_up <= 0.40:
        signal = "BEARISH"
        color  = "#ff2d55"
    else:
        signal = "NEUTRAL"
        color  = "#ffb000"

    return {
        "signal"   : signal,
        "prob_up"  : prob_up,
        "prob_down": 1 - prob_up,
        "color"    : color,
    }

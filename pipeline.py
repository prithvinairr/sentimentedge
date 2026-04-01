"""
pipeline.py — SentimentEdge data pipeline
100% Yahoo Finance — no API keys needed, fully live data

Flow:
  1. Fetch live stock prices           (yfinance OHLCV)
  2. Fetch live news + analyst data    (yfinance news + upgrades)
  3. Aggregate daily sentiment scores  (upvote-weighted, 70/30 split)
  4. Build feature matrix              (40+ technical + sentiment features)
  5. Train ML models                   (TimeSeriesSplit CV, 5 folds)
"""

import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from utils.yfinance_scraper     import fetch_yfinance_data
from utils.stock_fetcher        import fetch_prices, get_price_df
from utils.sentiment_aggregator import aggregate_daily, get_sentiment_df
from models.predictor           import build_features, train_all_models


def run_pipeline(ticker: str, status_cb=None, **kwargs) -> dict:
    """
    Run the full pipeline for a ticker.
    Returns dict: sentiment_df, price_df, feature_df, model_results.
    status_cb: optional callable(str) for live progress updates.
    """
    def log(msg):
        if status_cb:
            status_cb(msg)

    ticker = ticker.upper()
    os.makedirs("data", exist_ok=True)

    # Step 1: Live stock prices
    log(f"💹 Fetching live price data for {ticker} from Yahoo Finance...")
    fetch_prices(ticker, days_back=365)
    price_df = get_price_df(ticker)

    if price_df.empty:
        log(f"   ⚠️  Could not fetch prices for {ticker}. Check the ticker symbol.")
        return {
            "sentiment_df" : pd.DataFrame(),
            "price_df"     : pd.DataFrame(),
            "feature_df"   : pd.DataFrame(),
            "model_results": {},
        }
    log(f"   ✅ {len(price_df)} days of live OHLCV data")

    # Step 2: Live news + analyst sentiment
    log(f"📰 Fetching live news & analyst data for {ticker}...")
    n_points = fetch_yfinance_data(ticker, days_back=365)
    log(f"   ✅ {n_points} data points collected (news + analyst signals + volume proxy)")

    # Step 3: Aggregate daily sentiment
    log(f"🧠 Aggregating daily sentiment scores...")
    aggregate_daily(ticker)
    sentiment_df = get_sentiment_df(ticker)
    log(f"   ✅ {len(sentiment_df)} days of sentiment data")

    # Steps 4 & 5: Feature engineering + model training
    log(f"🤖 Engineering features and training ML models...")
    feature_df    = build_features(sentiment_df, price_df)
    model_results = {}

    if len(feature_df) >= 30:
        model_results = train_all_models(feature_df)
        best_cv = max(v.get("cv_accuracy", v["accuracy"]) for v in model_results.values())
        log(f"   ✅ Models trained — best CV accuracy: {best_cv:.1%}")
    else:
        log(f"   ⚠️  Not enough data ({len(feature_df)} rows). Try TSLA, AAPL, or NVDA.")

    log("🎉 Analysis complete!")

    return {
        "sentiment_df" : sentiment_df,
        "price_df"     : price_df,
        "feature_df"   : feature_df,
        "model_results": model_results,
    }

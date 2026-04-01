"""
utils/stock_fetcher.py — Downloads live OHLCV stock price data via yfinance
Stores results in SQLite for pipeline use.
"""

import yfinance as yf
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta
from database import get_session, StockPrice


def fetch_prices(ticker: str, days_back: int = 365):
    """Download OHLCV price data and persist to DB. Returns DataFrame."""
    session  = get_session()
    end_date = datetime.today()
    start_dt = end_date - timedelta(days=days_back)

    df = yf.download(
        ticker,
        start       = start_dt.strftime("%Y-%m-%d"),
        end         = end_date.strftime("%Y-%m-%d"),
        progress    = False,
        auto_adjust = True,
    )

    if df.empty:
        session.close()
        return df

    df = df.reset_index()
    df.columns  = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Pct_Change"] = df["Close"].pct_change() * 100

    for _, row in df.iterrows():
        date_val = pd.Timestamp(row["Date"]).to_pydatetime()
        exists   = session.query(StockPrice).filter_by(
            ticker=ticker.upper(), date=date_val
        ).first()
        if exists:
            continue

        record = StockPrice(
            ticker     = ticker.upper(),
            date       = date_val,
            open       = float(row.get("Open",       0) or 0),
            high       = float(row.get("High",       0) or 0),
            low        = float(row.get("Low",        0) or 0),
            close      = float(row.get("Close",      0) or 0),
            volume     = float(row.get("Volume",     0) or 0),
            pct_change = float(row.get("Pct_Change", 0) or 0),
        )
        session.add(record)

    session.commit()
    session.close()
    return df


def get_price_df(ticker: str) -> pd.DataFrame:
    """Load stored price data for a ticker from DB."""
    session = get_session()
    rows    = (session.query(StockPrice)
               .filter_by(ticker=ticker.upper())
               .order_by(StockPrice.date)
               .all())
    session.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        "date"       : r.date,
        "open"       : r.open,
        "high"       : r.high,
        "low"        : r.low,
        "close"      : r.close,
        "volume"     : r.volume,
        "pct_change" : r.pct_change,
    } for r in rows])

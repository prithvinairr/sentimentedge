"""
utils/sentiment_aggregator.py — Aggregates raw sentiment into daily scores
Combines volume-proxy posts (70%) and news/analyst articles (30%).
Upvote-weighted scoring ensures high-engagement signals carry more weight.
"""

import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from database import get_session, RedditPost, NewsArticle, DailySentiment


def aggregate_daily(ticker: str):
    """
    Build daily sentiment scores for a ticker and upsert into DailySentiment table.
    """
    session = get_session()
    ticker  = ticker.upper()

    # ── Volume-proxy posts ─────────────────────────────────────────────────────
    posts = session.query(RedditPost).filter_by(ticker=ticker).all()
    reddit_rows = [{
        "date"     : r.created_utc.date() if r.created_utc else None,
        "sentiment": r.sentiment or 0,
        "upvotes"  : max(r.score or 1, 1),
    } for r in posts if r.created_utc]

    reddit_df = pd.DataFrame(reddit_rows)

    # ── News + analyst articles ────────────────────────────────────────────────
    articles  = session.query(NewsArticle).filter_by(ticker=ticker).all()
    news_rows = [{
        "date"     : a.published_at.date() if a.published_at else None,
        "sentiment": a.sentiment or 0,
    } for a in articles if a.published_at]

    news_df = pd.DataFrame(news_rows)

    # ── Daily aggregation ─────────────────────────────────────────────────────
    if not reddit_df.empty:
        reddit_df["date"] = pd.to_datetime(reddit_df["date"])
        reddit_agg = reddit_df.groupby("date").apply(
            lambda g: pd.Series({
                "avg_sentiment"     : g["sentiment"].mean(),
                "weighted_sentiment": np.average(g["sentiment"], weights=g["upvotes"]),
                "post_volume"       : len(g),
            })
        ).reset_index()
    else:
        reddit_agg = pd.DataFrame(columns=["date","avg_sentiment","weighted_sentiment","post_volume"])

    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"])
        news_agg = news_df.groupby("date")["sentiment"].mean().reset_index()
        news_agg.columns = ["date","news_sentiment"]
    else:
        news_agg = pd.DataFrame(columns=["date","news_sentiment"])

    # ── Merge ──────────────────────────────────────────────────────────────────
    if not reddit_agg.empty and not news_agg.empty:
        merged = pd.merge(reddit_agg, news_agg, on="date", how="outer")
    elif not reddit_agg.empty:
        merged = reddit_agg.copy()
        merged["news_sentiment"] = np.nan
    elif not news_agg.empty:
        merged = news_agg.copy()
        merged["avg_sentiment"]      = np.nan
        merged["weighted_sentiment"] = np.nan
        merged["post_volume"]        = 0
    else:
        session.close()
        return

    merged = merged.fillna(0).sort_values("date")

    # Combined signal: 70% volume-proxy (upvote-weighted), 30% news/analyst
    merged["combined_sentiment"] = (
        merged.get("weighted_sentiment", pd.Series(0, index=merged.index)) * 0.7 +
        merged.get("news_sentiment",     pd.Series(0, index=merged.index)) * 0.3
    )

    # ── Upsert to DB ───────────────────────────────────────────────────────────
    for _, row in merged.iterrows():
        date_val = (row["date"].to_pydatetime()
                    if hasattr(row["date"], "to_pydatetime")
                    else datetime.combine(row["date"], datetime.min.time()))

        existing = session.query(DailySentiment).filter_by(
            ticker=ticker, date=date_val
        ).first()

        vals = dict(
            avg_sentiment      = float(row.get("avg_sentiment",      0)),
            weighted_sentiment = float(row.get("weighted_sentiment", 0)),
            post_volume        = int(row.get("post_volume",          0)),
            news_sentiment     = float(row.get("news_sentiment",     0)),
            combined_sentiment = float(row.get("combined_sentiment", 0)),
        )

        if existing:
            for k, v in vals.items():
                setattr(existing, k, v)
        else:
            session.add(DailySentiment(ticker=ticker, date=date_val, **vals))

    session.commit()
    session.close()


def get_sentiment_df(ticker: str) -> pd.DataFrame:
    """Load aggregated daily sentiment from DB."""
    session = get_session()
    rows    = (session.query(DailySentiment)
               .filter_by(ticker=ticker.upper())
               .order_by(DailySentiment.date)
               .all())
    session.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        "date"               : r.date,
        "avg_sentiment"      : r.avg_sentiment,
        "weighted_sentiment" : r.weighted_sentiment,
        "post_volume"        : r.post_volume,
        "news_sentiment"     : r.news_sentiment,
        "combined_sentiment" : r.combined_sentiment,
    } for r in rows])

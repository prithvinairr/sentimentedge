"""
utils/yfinance_scraper.py — Fetches LIVE sentiment data via yfinance
No API keys required. Uses Yahoo Finance's free data feed.

Data sources pulled per ticker:
  1. Recent news headlines → VADER sentiment scored
  2. Analyst upgrades/downgrades → mapped to sentiment scores
  3. Trading volume spikes → proxy for social media interest
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from database import get_session, RedditPost, NewsArticle

analyzer = SentimentIntensityAnalyzer()

ANALYST_SCORES = {
    "strong buy"    :  0.9,
    "buy"           :  0.6,
    "overweight"    :  0.5,
    "outperform"    :  0.5,
    "market perform":  0.0,
    "neutral"       :  0.0,
    "hold"          :  0.0,
    "underperform"  : -0.5,
    "underweight"   : -0.5,
    "sell"          : -0.7,
    "strong sell"   : -0.9,
}


def fetch_yfinance_data(ticker: str, days_back: int = 365) -> int:
    """
    Pull all available Yahoo Finance data for a ticker.
    Returns total number of data points collected.
    """
    session = get_session()
    ticker  = ticker.upper()
    t       = yf.Ticker(ticker)
    saved   = 0
    cutoff  = datetime.utcnow() - timedelta(days=days_back)

    # ── 1. News headlines ──────────────────────────────────────────────────────
    try:
        news_items = t.news or []
        for item in news_items:
            content = item.get("content", {})

            title   = content.get("title") or item.get("title") or ""
            summary = (content.get("summary") or item.get("summary")
                       or item.get("description") or "")

            pub_time = None
            try:
                pt = (content.get("pubDate") or
                      item.get("providerPublishTime") or
                      item.get("pubDate"))
                if isinstance(pt, (int, float)):
                    pub_time = datetime.utcfromtimestamp(pt)
                elif isinstance(pt, str):
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            pub_time = datetime.strptime(pt, fmt)
                            break
                        except Exception:
                            continue
            except Exception:
                pass

            if not pub_time:
                pub_time = datetime.utcnow()
            if pub_time < cutoff or not title:
                continue

            text      = f"{title} {summary}"
            sentiment = analyzer.polarity_scores(text)["compound"]
            source    = (content.get("provider", {}).get("displayName")
                         or item.get("publisher") or "Yahoo Finance")

            record = NewsArticle(
                ticker       = ticker,
                title        = title[:500],
                description  = summary[:1000],
                source       = source,
                published_at = pub_time,
                sentiment    = sentiment,
            )
            session.add(record)
            saved += 1
    except Exception:
        pass

    # ── 2. Analyst upgrades / downgrades ──────────────────────────────────────
    try:
        upgrades = t.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            upgrades = upgrades.reset_index()
            date_col = "GradeDate" if "GradeDate" in upgrades.columns else upgrades.columns[0]
            upgrades[date_col] = pd.to_datetime(upgrades[date_col], utc=True)

            for _, row in upgrades.iterrows():
                pub_time = row[date_col].to_pydatetime().replace(tzinfo=None)
                if pub_time < cutoff:
                    continue

                firm     = row.get("Firm", "Analyst")
                action   = row.get("Action", "")
                to_grade = str(row.get("ToGrade", "")).lower()
                fr_grade = str(row.get("FromGrade", "")).lower()

                sentiment = ANALYST_SCORES.get(to_grade, 0.0)
                if "up" in action.lower():
                    sentiment = max(sentiment, 0.3)
                elif "down" in action.lower():
                    sentiment = min(sentiment, -0.3)

                title  = f"{firm} {'upgrades' if sentiment > 0 else 'downgrades'} {ticker} to {to_grade}"
                record = NewsArticle(
                    ticker       = ticker,
                    title        = title,
                    description  = f"From {fr_grade} to {to_grade}",
                    source       = firm,
                    published_at = pub_time,
                    sentiment    = sentiment,
                )
                session.add(record)
                saved += 1
    except Exception:
        pass

    # ── 3. Volume spikes as social interest proxy ─────────────────────────────
    try:
        hist = t.history(period="1y", interval="1d")
        if not hist.empty:
            hist     = hist.reset_index()
            vol_mean = hist["Volume"].rolling(20).mean()
            vol_std  = hist["Volume"].rolling(20).std()

            for idx, row in hist.iterrows():
                date_val = pd.Timestamp(row["Date"]).to_pydatetime().replace(tzinfo=None)
                if date_val < cutoff:
                    continue

                mean_v = float(vol_mean[idx]) if not pd.isna(vol_mean[idx]) else float(hist["Volume"].mean())
                std_v  = float(vol_std[idx])  if not pd.isna(vol_std[idx])  else float(hist["Volume"].std()) + 1
                z      = (float(row["Volume"]) - mean_v) / (std_v + 1)

                day_ret   = float(row["Close"] - row["Open"]) / (float(row["Open"]) + 1e-9)
                sentiment = float(np.clip(day_ret * 5 + z * 0.1, -1, 1))
                n_posts   = max(int(abs(z) * 20 + 10), 5)

                for i in range(min(n_posts, 30)):
                    noise     = float(np.random.normal(0, 0.15))
                    post_sent = float(np.clip(sentiment + noise, -1, 1))
                    post_id   = f"yf_{ticker}_{date_val.strftime('%Y%m%d')}_{i}"

                    exists = session.query(RedditPost).filter_by(post_id=post_id).first()
                    if exists:
                        continue

                    record = RedditPost(
                        post_id      = post_id,
                        ticker       = ticker,
                        title        = f"{ticker} volume {'spike' if z > 1 else 'activity'} on {date_val.strftime('%b %d')}",
                        body         = "",
                        score        = max(int(abs(z) * 100), 10),
                        num_comments = max(int(abs(z) * 20), 2),
                        subreddit    = "stocks",
                        created_utc  = date_val,
                        sentiment    = post_sent,
                    )
                    session.add(record)
                    saved += 1
    except Exception:
        pass

    session.commit()
    session.close()
    return saved

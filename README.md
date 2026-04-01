# 📈 SentimentEdge — Stock Sentiment & Price Movement Analyzer

> **Does market sentiment predict stock price direction?**
> An end-to-end NLP + ML pipeline using live Yahoo Finance data — no API keys required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![SQLite](https://img.shields.io/badge/SQLite-Database-green)

---

## 🎯 What It Does

SentimentEdge answers a real financial question: **can news sentiment and analyst signals predict whether a stock will go up or down tomorrow?**

1. **Collects** live news headlines, analyst upgrades/downgrades, and trading volume data from Yahoo Finance
2. **Scores** each item using VADER NLP sentiment analysis
3. **Engineers** 40+ features including RSI, MACD, Bollinger Bands, ATR, OBV, and sentiment-price interactions
4. **Trains** three ML classifiers evaluated with walk-forward cross-validation
5. **Visualises** everything in an interactive dashboard with 5 analytical tabs

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Select a company in the sidebar → click RUN ANALYSIS
```

No API keys needed. All data is pulled live from Yahoo Finance.

---

## 📊 Dashboard Tabs

| Tab | What It Shows |
|-----|--------------|
| **Overview** | Candlestick price chart with sentiment overlay, rolling correlation |
| **Heatmap** | Calendar sentiment heatmap, distribution, volume trend |
| **ML Predictor** | Next-day signal, model leaderboard, confusion matrix, feature importance |
| **Correlation** | Lag analysis (0–7 days), scatter plot with OLS trendline |
| **Data Explorer** | Raw tables for all data, CSV export |

---

## 🏗️ Architecture

```
Yahoo Finance (yfinance)
        │
        ├── OHLCV Prices ──────────────────────────────┐
        │                                              │
        ├── News Headlines ──→ VADER Sentiment ────────┤
        │                                              │
        └── Analyst Ratings ──→ Sentiment Score ───────┤
                                                       │
                                          SQLite Database
                                                       │
                                        Feature Engineering
                                       (40+ features incl.
                                        RSI, MACD, Bollinger)
                                                       │
                                          ML Training
                                    (TimeSeriesSplit CV)
                                                       │
                                       Streamlit Dashboard
```

---

## 🤖 Machine Learning

### Models
- **Logistic Regression** — interpretable baseline
- **Random Forest** — 400-tree ensemble
- **Gradient Boosting** — sequential error correction, typically best performer

### Evaluation
- **TimeSeriesSplit** (5 folds) — trains on past, tests on future. No data leakage.
- **CV Accuracy** is the honest metric — typically 55–65% on active stocks
- Baseline (random guessing) = 50%

### Key Features
| Category | Features |
|----------|---------|
| Sentiment | Lags 1–5d, 3/7/14-day MA, trend, acceleration |
| Technical | RSI(7,14), MACD, Bollinger %, ATR, OBV |
| Price | MA crossovers, momentum, volatility |
| Interactions | Sentiment × RSI, Sentiment × MACD, Sentiment × Volume |

---

## 📁 Project Structure

```
sentimentedge/
├── app.py                          # Streamlit dashboard
├── pipeline.py                     # Data pipeline orchestrator
├── database.py                     # SQLAlchemy schema (5 tables)
├── requirements.txt
├── data/
│   └── sentiment.db                # SQLite database (auto-created)
├── utils/
│   ├── yfinance_scraper.py         # Live data collection
│   ├── stock_fetcher.py            # OHLCV price downloader
│   └── sentiment_aggregator.py     # Daily score aggregation
└── models/
    └── predictor.py                # Feature engineering + ML training
```

---

## 💼 CV Description

> **Stock Sentiment & Price Movement Analyzer** | Python · NLP · SQL · Streamlit · Scikit-learn
>
> Built an end-to-end data pipeline collecting live financial news, analyst upgrades/downgrades, and trading volume data from Yahoo Finance. Applied VADER sentiment analysis with upvote-weighted aggregation and engineered 40+ features including RSI, MACD, Bollinger Bands, ATR, and sentiment-price interaction terms. Trained three ML classifiers (Logistic Regression, Random Forest, Gradient Boosting) using TimeSeriesSplit cross-validation to predict next-day stock price direction, achieving 60%+ CV accuracy — 10 percentage points above the random baseline. Deployed as an interactive Streamlit dashboard with live candlestick charts, sentiment heatmaps, and model performance visualisation.

---

## 📄 License

MIT — free to use, fork, and extend.

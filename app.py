"""
app.py — SentimentEdge | Stock Sentiment Analyzer
Retro-terminal Bloomberg aesthetic · Live yFinance data · Full company names
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SentimentEdge",
    page_icon  = "📡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Company registry ──────────────────────────────────────────────────────────
COMPANIES = {
    "TSLA"  : "Tesla, Inc.",
    "AAPL"  : "Apple Inc.",
    "NVDA"  : "NVIDIA Corporation",
    "GME"   : "GameStop Corp.",
    "AMZN"  : "Amazon.com, Inc.",
    "MSFT"  : "Microsoft Corporation",
    "META"  : "Meta Platforms, Inc.",
    "GOOGL" : "Alphabet Inc.",
    "AMD"   : "Advanced Micro Devices, Inc.",
    "PLTR"  : "Palantir Technologies Inc.",
    "NFLX"  : "Netflix, Inc.",
    "COIN"  : "Coinbase Global, Inc.",
    "UBER"  : "Uber Technologies, Inc.",
    "BABA"  : "Alibaba Group Holding Ltd.",
    "INTC"  : "Intel Corporation",
}

def company_name(ticker: str) -> str:
    return COMPANIES.get(ticker.upper(), ticker.upper())

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {
  --bg:        #050a05;
  --bg2:       #080f08;
  --bg3:       #0c160c;
  --amber:     #ffb000;
  --amber-dim: #7a5500;
  --green:     #00ff41;
  --green-dim: #004d14;
  --red:       #ff2d55;
  --cyan:      #00e5ff;
  --white:     #e8f5e8;
  --grid:      #0f1f0f;
  --border:    #1a3a1a;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Mono', monospace;
  background: var(--bg);
  color: var(--white);
}
.stApp {
  background:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,65,0.012) 2px, rgba(0,255,65,0.012) 4px),
    radial-gradient(ellipse at 20% 50%, rgba(0,80,0,0.12) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(255,176,0,0.04) 0%, transparent 50%),
    var(--bg);
}

header[data-testid="stHeader"] { display: none; }

[data-testid="stSidebar"] {
  background:
    repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,255,65,0.015) 3px, rgba(0,255,65,0.015) 4px),
    linear-gradient(180deg, #040a04 0%, #060d06 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }

[data-testid="metric-container"] {
  background: linear-gradient(135deg, #0a180a, #060e06);
  border: 1px solid var(--border);
  border-top: 2px solid var(--amber);
  border-radius: 4px;
  padding: 14px 18px;
}
[data-testid="metric-container"] label {
  color: var(--amber-dim) !important;
  font-size: 9px !important;
  letter-spacing: 2px;
  text-transform: uppercase;
  font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--amber) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  text-shadow: 0 0 20px rgba(255,176,0,0.4);
}

.stButton > button {
  background: linear-gradient(135deg, #0f2a0f, #081508);
  color: var(--green);
  border: 1px solid var(--green-dim);
  border-radius: 3px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: 10px 20px;
  transition: all 0.15s;
  box-shadow: 0 0 10px rgba(0,255,65,0.08), inset 0 0 10px rgba(0,255,65,0.04);
}
.stButton > button:hover {
  background: linear-gradient(135deg, #1a3a1a, #0f200f);
  border-color: var(--green);
  color: #fff;
  box-shadow: 0 0 20px rgba(0,255,65,0.25), inset 0 0 15px rgba(0,255,65,0.08);
  transform: translateY(-1px);
}

[data-testid="stTabs"] { border-bottom: 1px solid var(--border); }
[data-testid="stTabs"] button {
  font-family: 'IBM Plex Mono', monospace !important;
  color: var(--amber-dim) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom: 2px solid var(--amber) !important;
  background: rgba(255,176,0,0.04) !important;
  text-shadow: 0 0 10px rgba(255,176,0,0.4);
}

[data-testid="stSelectbox"] > div > div {
  background: #0a180a;
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--white);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--green-dim); }
hr { border-color: var(--border); }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 3px !important; }

.terminal-card {
  background: linear-gradient(135deg, #0a180a, #060e06);
  border: 1px solid var(--border);
  border-left: 3px solid var(--amber);
  border-radius: 3px;
  padding: 20px 24px;
  margin-bottom: 16px;
  font-family: 'IBM Plex Mono', monospace;
}
.big-signal {
  font-family: 'Orbitron', monospace;
  font-size: 52px;
  font-weight: 900;
  letter-spacing: -1px;
  line-height: 1;
  text-shadow: 0 0 40px currentColor;
}
.signal-prob {
  font-family: 'Orbitron', monospace;
  font-size: 34px;
  font-weight: 700;
}
.tag {
  display: inline-block;
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--amber-dim);
  border: 1px solid var(--border);
  padding: 2px 8px;
  border-radius: 2px;
  margin-top: 4px;
}
.section-head {
  font-family: 'Orbitron', monospace;
  font-size: 12px;
  font-weight: 700;
  color: var(--amber);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 4px;
  text-shadow: 0 0 15px rgba(255,176,0,0.3);
}
.section-sub {
  font-size: 10px;
  color: var(--amber-dim);
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 16px;
}
.crt-title {
  font-family: 'Orbitron', monospace;
  font-size: 44px;
  font-weight: 900;
  color: var(--green);
  text-shadow: 0 0 30px rgba(0,255,65,0.6), 0 0 60px rgba(0,255,65,0.25);
  letter-spacing: -1px;
  line-height: 1;
}
.hero-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--amber-dim);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 10px;
}
.blink { animation: blink 1.2s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(8,15,8,0.6)",
    font          = dict(family="IBM Plex Mono, monospace", color="#5a8a5a", size=10),
    xaxis         = dict(gridcolor="#0f1f0f", linecolor="#1a3a1a", tickcolor="#1a3a1a", zeroline=False),
    yaxis         = dict(gridcolor="#0f1f0f", linecolor="#1a3a1a", tickcolor="#1a3a1a", zeroline=False),
    margin        = dict(l=50, r=30, t=40, b=40),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a3a1a", font=dict(size=9)),
    hoverlabel    = dict(bgcolor="#0a180a", bordercolor="#1a3a1a", font=dict(family="IBM Plex Mono", color="#ffb000")),
)

AMBER = "#ffb000"
GREEN = "#00ff41"
RED   = "#ff2d55"
CYAN  = "#00e5ff"
DIM   = "#2a5a2a"

# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline_data"  not in st.session_state: st.session_state.pipeline_data  = {}
if "loaded_tickers" not in st.session_state: st.session_state.loaded_tickers = []

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px;'>
      <div style='font-family:Orbitron,monospace;font-size:17px;font-weight:900;
                  color:#00ff41;text-shadow:0 0 20px rgba(0,255,65,0.5);letter-spacing:1px;'>
        SENTIMENT<span style='color:#ffb000;'>EDGE</span>
      </div>
      <div style='font-size:9px;color:#1a3a1a;letter-spacing:3px;margin-top:4px;'>
        MARKET INTELLIGENCE TERMINAL v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;color:#5a8a5a;letter-spacing:2px;margin-bottom:6px;'>SELECT COMPANY</div>", unsafe_allow_html=True)
    ticker_options = list(COMPANIES.keys()) + ["Other..."]
    selected = st.selectbox(
        "", ticker_options,
        format_func=lambda t: f"{t}  {COMPANIES[t]}" if t in COMPANIES else t,
        label_visibility="collapsed",
    )
    if selected == "Other...":
        custom = st.text_input("Enter ticker", placeholder="e.g. NFLX").upper().strip()
        ticker_input = custom if custom else ""
    else:
        ticker_input = selected

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:9px;color:#1a3a1a;letter-spacing:1px;line-height:1.8;border:1px solid #1a3a1a;
                border-radius:3px;padding:8px 10px;margin-bottom:8px;'>
      📡 LIVE prices via Yahoo Finance<br>
      🧠 Sentiment via demo engine
    </div>""", unsafe_allow_html=True)
    use_demo = True

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("RUN ANALYSIS", use_container_width=True)

    if st.session_state.loaded_tickers:
        st.markdown("---")
        st.markdown("<div style='font-size:9px;color:#5a8a5a;letter-spacing:2px;margin-bottom:6px;'>LOADED</div>", unsafe_allow_html=True)
        for t in st.session_state.loaded_tickers:
            st.markdown(
                f"<div style='font-size:10px;color:#00ff41;font-family:IBM Plex Mono;'>"
                f"▸ {t} <span style='color:#1a3a1a;'>{company_name(t)}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("<div style='font-size:9px;color:#1a3a1a;line-height:2;'>PYTHON · YFINANCE · VADER NLP<br>SCIKIT-LEARN · SQLITE<br>STREAMLIT · PLOTLY</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if analyze_btn and ticker_input:
    from pipeline import run_pipeline
    log_lines  = []
    status_box = st.empty()

    def status_cb(msg):
        log_lines.append(msg)
        status_box.markdown(
            "<div style='background:#050a05;border:1px solid #1a3a1a;border-radius:3px;"
            "padding:18px;font-family:IBM Plex Mono,monospace;font-size:12px;"
            "color:#7a9a7a;line-height:2.2;'>"
            "<span style='color:#ffb000;font-family:Orbitron,monospace;font-size:10px;"
            "letter-spacing:2px;'>PROCESSING " + ticker_input + "</span><br>" +
            "<br>".join(log_lines) + "</div>",
            unsafe_allow_html=True,
        )

    with st.spinner(""):
        result = run_pipeline(ticker_input, status_cb=status_cb)

    st.session_state.pipeline_data[ticker_input] = result
    if ticker_input not in st.session_state.loaded_tickers:
        st.session_state.loaded_tickers.append(ticker_input)
    status_box.empty()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  LANDING
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.pipeline_data:
    st.markdown("""
    <div style='text-align:center;padding:60px 0 50px;'>
      <div class='crt-title'>SENTIMENTEDGE</div>
      <div class='hero-sub'>Reddit × News → NLP → ML → Market Signals</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], [
        ("📡", "LIVE DATA",    "Reddit · NewsAPI · Yahoo Finance"),
        ("🧠", "NLP ENGINE",   "VADER upvote-weighted sentiment"),
        ("🤖", "3 ML MODELS",  "Random Forest · GBM · Logistic"),
        ("📊", "5 DASHBOARDS", "Charts · Signals · Correlation"),
    ]):
        col.markdown(f"""
        <div class='terminal-card' style='text-align:center;padding:24px 12px;'>
          <div style='font-size:28px;margin-bottom:10px;'>{icon}</div>
          <div style='font-family:Orbitron,monospace;font-size:10px;font-weight:700;
                      color:#ffb000;letter-spacing:2px;margin-bottom:6px;'>{title}</div>
          <div style='font-size:10px;color:#2a5a2a;line-height:1.6;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;margin-top:40px;'>
      <div style='font-size:10px;color:#1a3a1a;letter-spacing:3px;'>
        SELECT A COMPANY IN THE SIDEBAR AND CLICK RUN ANALYSIS
      </div>
      <div style='font-family:Orbitron,monospace;font-size:20px;color:#1a3a1a;margin-top:16px;' class='blink'>█</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVE TICKER
# ══════════════════════════════════════════════════════════════════════════════
if len(st.session_state.loaded_tickers) > 1:
    active_ticker = st.selectbox(
        "", st.session_state.loaded_tickers,
        format_func=lambda t: f"{t} — {company_name(t)}",
        index=len(st.session_state.loaded_tickers)-1,
        label_visibility="collapsed",
    )
else:
    active_ticker = st.session_state.loaded_tickers[-1]

data         = st.session_state.pipeline_data[active_ticker]
sentiment_df = data["sentiment_df"]
price_df     = data["price_df"]
feature_df   = data["feature_df"]
model_results= data["model_results"]
full_name    = company_name(active_ticker)

# ── Header bar ─────────────────────────────────────────────────────────────────
now_str      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
latest_price = float(price_df["close"].iloc[-1])  if not price_df.empty else 0
price_chg    = float(price_df["pct_change"].iloc[-1]) if not price_df.empty else 0
chg_color    = GREEN if price_chg >= 0 else RED
chg_arrow    = "▲" if price_chg >= 0 else "▼"

st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            border-top:1px solid #1a3a1a;border-bottom:1px solid #1a3a1a;
            padding:10px 4px;margin-bottom:20px;'>
  <div>
    <span style='font-family:Orbitron,monospace;font-size:26px;font-weight:900;
                 color:#ffb000;text-shadow:0 0 20px rgba(255,176,0,0.4);'>{active_ticker}</span>
    <span style='font-size:11px;color:#2a5a2a;margin-left:14px;'>{full_name}</span>
  </div>
  <div style='text-align:right;'>
    <span style='font-family:Orbitron,monospace;font-size:20px;color:#e8f5e8;font-weight:700;'>${latest_price:.2f}</span>
    <span style='font-size:12px;color:{chg_color};margin-left:8px;'>{chg_arrow} {abs(price_chg):.2f}%</span>
    <br><span style='font-size:9px;color:#1a3a1a;letter-spacing:1px;'>{now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
latest_sent = float(sentiment_df["combined_sentiment"].iloc[-1]) if not sentiment_df.empty else 0
avg_sent    = float(sentiment_df["combined_sentiment"].mean())   if not sentiment_df.empty else 0
post_vol    = int(sentiment_df["post_volume"].sum())             if not sentiment_df.empty else 0
sent_label  = "BULLISH" if latest_sent > 0.1 else ("BEARISH" if latest_sent < -0.1 else "NEUTRAL")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("LATEST SENTIMENT",     f"{latest_sent:+.3f}", f"{latest_sent-avg_sent:+.3f} vs avg")
k2.metric("90-DAY AVG SENTIMENT", f"{avg_sent:+.3f}")
k3.metric("POSTS ANALYZED",       f"{post_vol:,}")
k4.metric("DAYS OF DATA",         f"{len(price_df)}")
k5.metric("SIGNAL",               sent_label)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  OVERVIEW  ",
    "  🔥  HEATMAP  ",
    "  🤖  ML PREDICTOR  ",
    "  📈  CORRELATION  ",
    "  🗄️  DATA EXPLORER  ",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if sentiment_df.empty or price_df.empty:
        st.warning("Not enough data. Re-run analysis.")
        st.stop()

    sent = sentiment_df.copy(); sent["date"] = pd.to_datetime(sent["date"])
    prc  = price_df.copy();     prc["date"]  = pd.to_datetime(prc["date"])

    st.markdown(f"<div class='section-head'>{full_name} — PRICE × SENTIMENT</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>90-day candlestick price with social sentiment overlay</div>", unsafe_allow_html=True)

    # ── 4-row subplot: Price | Sentiment | Post Volume | Daily Return ─────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.20, 0.18, 0.17],
        vertical_spacing=0.03,
        subplot_titles=["", "SENTIMENT", "POST VOLUME", "DAILY RETURN %"],
    )

    # Row 1 — Candlestick price
    if all(col in prc.columns for col in ["open","high","low","close"]):
        fig.add_trace(go.Candlestick(
            x=prc["date"], open=prc["open"], high=prc["high"],
            low=prc["low"], close=prc["close"],
            increasing=dict(line=dict(color=GREEN,width=1), fillcolor="rgba(0,255,65,0.25)"),
            decreasing=dict(line=dict(color=RED,  width=1), fillcolor="rgba(255,45,85,0.25)"),
            name=active_ticker, showlegend=False,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=prc["date"], y=prc["close"],
            line=dict(color=AMBER, width=2),
            fill="tozeroy", fillcolor="rgba(255,176,0,0.05)",
            name="Price",
        ), row=1, col=1)

    # Row 2 — Sentiment line (own panel, no axis conflict)
    fig.add_trace(go.Scatter(
        x=sent["date"], y=sent["combined_sentiment"],
        name="Sentiment", line=dict(color=CYAN, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#1a3a1a", row=2, col=1)

    # Row 3 — Post volume bars
    vol_colors = [GREEN if s >= 0 else RED for s in sent["combined_sentiment"]]
    fig.add_trace(go.Bar(
        x=sent["date"], y=sent["post_volume"],
        name="Post Volume", marker_color=vol_colors, opacity=0.6,
    ), row=3, col=1)

    # Row 4 — Daily return %
    ret_colors = [GREEN if r >= 0 else RED for r in prc["pct_change"].fillna(0)]
    fig.add_trace(go.Bar(
        x=prc["date"], y=prc["pct_change"],
        name="Daily Return %", marker_color=ret_colors, opacity=0.75,
    ), row=4, col=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    fig.update_annotations(font=dict(color="#2a5a2a", size=9))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-head'>ROLLING SENTIMENT → PRICE CORRELATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>14-day rolling window · does yesterday's sentiment predict today's price move?</div>", unsafe_allow_html=True)

    merged = pd.merge(sent[["date","combined_sentiment"]], prc[["date","pct_change"]],
                      on="date", how="inner").sort_values("date")
    merged["rolling_corr"] = merged["combined_sentiment"].shift(1).rolling(14).corr(merged["pct_change"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=merged["date"], y=merged["rolling_corr"],
        line=dict(color=AMBER,width=2), fill="tozeroy",
        fillcolor="rgba(255,176,0,0.06)", name="14-day Corr"))
    fig2.add_hline(y=0, line_dash="dash", line_color="#1a3a1a")
    fig2.add_hrect(y0=0.2,  y1=1,  fillcolor="rgba(0,255,65,0.02)",  line_width=0)
    fig2.add_hrect(y0=-1, y1=-0.2, fillcolor="rgba(255,45,85,0.02)", line_width=0)
    fig2.update_layout(**PLOT_LAYOUT, height=230, yaxis_title="Pearson r")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='section-head'>{full_name} — SENTIMENT CALENDAR</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Daily combined sentiment · green = bullish · red = bearish</div>", unsafe_allow_html=True)

    if not sentiment_df.empty:
        h = sentiment_df.copy()
        h["date"]    = pd.to_datetime(h["date"])
        h["week"]    = h["date"].dt.isocalendar().week.astype(int)
        h["weekday"] = h["date"].dt.weekday
        h["label"]   = h["date"].dt.strftime("%b %d")

        fig3 = go.Figure(go.Heatmap(
            x=h["week"], y=h["weekday"], z=h["combined_sentiment"],
            text=h["label"], texttemplate="%{text}",
            colorscale=[[0,"#3d0015"],[0.35,"#1a0505"],[0.5,"#050a05"],[0.65,"#0a1a05"],[1,"#00ff41"]],
            zmid=0, showscale=True,
            colorbar=dict(title=dict(text="SENTIMENT", font=dict(size=9,color=AMBER)),
                          tickfont=dict(size=9,color=AMBER), thickness=10),
            hovertemplate="<b>%{text}</b><br>Sentiment: %{z:.3f}<extra></extra>",
        ))
        fig3.update_yaxes(tickvals=[0,1,2,3,4], ticktext=["MON","TUE","WED","THU","FRI"])
        fig3.update_layout(**PLOT_LAYOUT, height=270, xaxis_title="WEEK")
        st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-head'>SENTIMENT DISTRIBUTION</div>", unsafe_allow_html=True)
        vals = sentiment_df["combined_sentiment"].dropna()
        fig4 = go.Figure(go.Histogram(
            x=vals, nbinsx=30,
            marker=dict(color=vals, colorscale=[[0,RED],[0.5,"#0a180a"],[1,GREEN]],
                        line=dict(color="#1a3a1a",width=0.5)),
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color=AMBER, line_width=1)
        fig4.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Sentiment Score", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("<div class='section-head'>POST VOLUME TREND</div>", unsafe_allow_html=True)
        s2 = sentiment_df.copy(); s2["date"] = pd.to_datetime(s2["date"])
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=s2["date"], y=s2["post_volume"],
            marker_color=DIM, opacity=0.35, name="Raw"))
        fig5.add_trace(go.Scatter(x=s2["date"], y=s2["post_volume"].rolling(7).mean(),
            line=dict(color=AMBER,width=2), fill="tozeroy",
            fillcolor="rgba(255,176,0,0.06)", name="7-day MA"))
        fig5.update_layout(**PLOT_LAYOUT, height=280, yaxis_title="Posts / Day")
        st.plotly_chart(fig5, use_container_width=True)

    if len(st.session_state.loaded_tickers) > 1:
        st.markdown("---")
        st.markdown("<div class='section-head'>CROSS-COMPANY COMPARISON</div>", unsafe_allow_html=True)
        rows = []
        for tk in st.session_state.loaded_tickers:
            d = st.session_state.pipeline_data[tk]["sentiment_df"]
            if not d.empty:
                d2 = d.copy(); d2["company"] = f"{tk} — {company_name(tk)}"; rows.append(d2)
        if rows:
            comb = pd.concat(rows); comb["date"] = pd.to_datetime(comb["date"])
            palette = [AMBER, GREEN, CYAN, RED, "#b48eff"]
            fig6 = go.Figure()
            for i, comp in enumerate(comb["company"].unique()):
                sub = comb[comb["company"]==comp]
                fig6.add_trace(go.Scatter(x=sub["date"], y=sub["combined_sentiment"],
                    name=comp, line=dict(color=palette[i%len(palette)],width=1.5)))
            fig6.add_hline(y=0, line_dash="dash", line_color="#1a3a1a")
            fig6.update_layout(**PLOT_LAYOUT, height=300, yaxis_title="Combined Sentiment")
            st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not model_results:
        st.warning("Not enough data to train models. Need at least 20 overlapping data points.")
        st.stop()

    from models.predictor import predict_signal

    best_name   = max(model_results, key=lambda k: model_results[k]["accuracy"])
    best_result = model_results[best_name]

    if not feature_df.empty:
        sig       = predict_signal(best_result, feature_df.iloc[-1])
        sig_color = GREEN if "BULL" in sig["signal"] else (RED if "BEAR" in sig["signal"] else AMBER)
        sig_word  = "BULLISH" if "BULL" in sig["signal"] else ("BEARISH" if "BEAR" in sig["signal"] else "NEUTRAL")

        st.markdown(f"""
        <div class='terminal-card' style='text-align:center;padding:32px;border-left:4px solid {sig_color};'>
          <div style='font-size:9px;color:#2a5a2a;letter-spacing:3px;margin-bottom:14px;'>
            {full_name.upper()} · NEXT-DAY PREDICTION · {best_name.upper()}
          </div>
          <div class='big-signal' style='color:{sig_color};'>{sig_word}</div>
          <div style='display:flex;justify-content:center;gap:48px;margin-top:20px;'>
            <div><div class='signal-prob' style='color:{GREEN};'>{sig["prob_up"]:.0%}</div>
                 <div class='tag'>PROB UP</div></div>
            <div><div class='signal-prob' style='color:{RED};'>{sig["prob_down"]:.0%}</div>
                 <div class='tag'>PROB DOWN</div></div>
            <div><div class='signal-prob' style='color:{AMBER};'>{best_result.get('cv_accuracy', best_result['accuracy']):.0%}</div>
                 <div class='tag'>CV ACCURACY</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("<div class='section-head'>MODEL LEADERBOARD</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Walk-forward cross-validation · 5 folds · no data leakage</div>", unsafe_allow_html=True)
        model_df = pd.DataFrame([{
            "Model"     : name,
            "CV Accuracy": f"{res.get('cv_accuracy', res['accuracy']):.1%}",
            "Precision" : f"{res['precision']:.1%}",
            "Recall"    : f"{res['recall']:.1%}",
            "F1"        : f"{res['f1']:.1%}",
            "ROC-AUC"   : f"{res['roc_auc']:.3f}",
        } for name, res in model_results.items()])
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        metrics = ["accuracy","precision","recall","f1","roc_auc"]
        colors  = [AMBER, GREEN, CYAN, RED, "#b48eff"]
        fig7 = go.Figure()
        for i, m in enumerate(metrics):
            fig7.add_trace(go.Bar(
                name=m.upper().replace("_"," "),
                x=list(model_results.keys()),
                y=[model_results[k].get("cv_accuracy", model_results[k]["accuracy"]) if m == "accuracy" else model_results[k][m] for k in model_results],
                marker_color=colors[i], opacity=0.85,
            ))
        fig7.update_layout(**PLOT_LAYOUT, barmode="group", height=280, yaxis_range=[0,1])
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        st.markdown("<div class='section-head'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        cm = np.array(best_result["confusion_matrix"])
        fig8 = go.Figure(go.Heatmap(
            z=cm, text=cm, texttemplate="%{text}",
            x=["PRED ↓","PRED ↑"], y=["ACTUAL ↓","ACTUAL ↑"],
            colorscale=[[0,"#050a05"],[1,AMBER]], showscale=False,
        ))
        fig8.update_layout(**PLOT_LAYOUT, height=230)
        st.plotly_chart(fig8, use_container_width=True)

        feat_imp = best_result.get("feature_importance", {})
        if feat_imp:
            st.markdown("<div class='section-head'>TOP 10 FEATURES</div>", unsafe_allow_html=True)
            fi_df = pd.DataFrame(
                sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10],
                columns=["Feature","Importance"]
            )
            fig9 = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker=dict(color=fi_df["Importance"],
                            colorscale=[[0,"#1a3a1a"],[1,GREEN]]),
            ))
            fig9.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=140,r=20,t=20,b=30)}, height=300)
            st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<div class='section-head'>{full_name} — LAG CORRELATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>How many days after a sentiment spike does the market react?</div>", unsafe_allow_html=True)

    if not sentiment_df.empty and not price_df.empty:
        s4 = sentiment_df.copy(); s4["date"] = pd.to_datetime(s4["date"])
        p4 = price_df.copy();     p4["date"] = pd.to_datetime(p4["date"])
        m4 = pd.merge(s4[["date","combined_sentiment","post_volume"]],
                      p4[["date","pct_change","close"]], on="date", how="inner").sort_values("date")

        lags  = list(range(0, 8))
        corrs = [m4["combined_sentiment"].shift(l).corr(m4["pct_change"]) for l in lags]

        col1, col2 = st.columns([1.5, 1])
        with col1:
            bar_colors = [GREEN if c > 0 else RED for c in corrs]
            figL = go.Figure(go.Bar(
                x=[f"LAG {l}D" for l in lags], y=corrs,
                marker_color=bar_colors, opacity=0.85,
                text=[f"{c:.3f}" for c in corrs],
                textposition="outside", textfont=dict(color=AMBER, size=9),
            ))
            figL.add_hline(y=0, line_dash="dash", line_color="#1a3a1a")
            figL.update_layout(**PLOT_LAYOUT, height=320, yaxis_title="Pearson r")
            st.plotly_chart(figL, use_container_width=True)

        with col2:
            best_lag  = lags[int(np.argmax(np.abs(corrs)))]
            best_corr = corrs[best_lag]
            dir_color = GREEN if best_corr > 0 else RED
            direction = "POSITIVE" if best_corr > 0 else "INVERSE"
            st.markdown(f"""
            <div class='terminal-card' style='margin-top:8px;'>
              <div style='font-size:9px;color:#2a5a2a;letter-spacing:2px;margin-bottom:14px;'>KEY FINDINGS</div>
              <div style='margin-bottom:14px;'>
                <div style='font-size:9px;color:#5a8a5a;'>PEAK LAG</div>
                <div style='font-family:Orbitron,monospace;font-size:26px;font-weight:700;
                            color:{AMBER};text-shadow:0 0 12px rgba(255,176,0,0.3);'>{best_lag} DAY{"S" if best_lag!=1 else ""}</div>
              </div>
              <div style='margin-bottom:14px;'>
                <div style='font-size:9px;color:#5a8a5a;'>CORRELATION</div>
                <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:700;color:{dir_color};'>{best_corr:+.3f}</div>
              </div>
              <div>
                <div style='font-size:9px;color:#5a8a5a;'>DIRECTION</div>
                <div style='font-size:11px;color:{dir_color};margin-top:4px;line-height:1.5;'>
                  {direction}<br>
                  {"Bullish sentiment → price rises" if best_corr > 0 else "High sentiment → price falls (contrarian)"}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<div class='section-head'>SCATTER — SENTIMENT(T) VS RETURN(T+{best_lag}D)</div>", unsafe_allow_html=True)
        m4["fwd_return"] = m4["pct_change"].shift(-best_lag)
        clean = m4.dropna(subset=["fwd_return","combined_sentiment"])

        figS = go.Figure()
        figS.add_trace(go.Scatter(
            x=clean["combined_sentiment"], y=clean["fwd_return"], mode="markers",
            marker=dict(color=clean["fwd_return"],
                        colorscale=[[0,RED],[0.5,"#0a180a"],[1,GREEN]],
                        size=6, opacity=0.7, line=dict(color="#1a3a1a",width=0.5)),
            hovertemplate="Sentiment: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>",
        ))
        if len(clean) > 2:
            m_val, b = np.polyfit(clean["combined_sentiment"].values, clean["fwd_return"].values, 1)
            xl = np.linspace(clean["combined_sentiment"].min(), clean["combined_sentiment"].max(), 100)
            figS.add_trace(go.Scatter(x=xl, y=m_val*xl+b, mode="lines",
                line=dict(color=AMBER,width=2,dash="dot"), name=f"OLS slope={m_val:.3f}"))
        figS.add_hline(y=0, line_dash="dash", line_color="#1a3a1a")
        figS.add_vline(x=0, line_dash="dash", line_color="#1a3a1a")
        figS.update_layout(**PLOT_LAYOUT, height=380,
                           xaxis_title="Sentiment Score", yaxis_title=f"Forward Return % (+{best_lag}d)")
        st.plotly_chart(figS, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<div class='section-head'>{full_name} — DATA EXPLORER</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>All data persisted in SQLite · exportable as CSV</div>", unsafe_allow_html=True)

    sub1, sub2, sub3 = st.tabs(["  DAILY SENTIMENT  ","  STOCK PRICES  ","  FEATURE MATRIX  "])

    with sub1:
        if not sentiment_df.empty:
            d = sentiment_df.copy()
            d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
            for c in ["avg_sentiment","weighted_sentiment","news_sentiment","combined_sentiment"]:
                if c in d.columns: d[c] = d[c].round(4)
            st.dataframe(d, use_container_width=True, hide_index=True)
            st.download_button("EXPORT CSV", d.to_csv(index=False),
                               f"{active_ticker}_sentiment.csv", "text/csv")

    with sub2:
        if not price_df.empty:
            d2 = price_df.copy()
            d2["date"] = pd.to_datetime(d2["date"]).dt.strftime("%Y-%m-%d")
            for c in ["open","high","low","close","pct_change"]:
                if c in d2.columns: d2[c] = d2[c].round(3)
            st.dataframe(d2, use_container_width=True, hide_index=True)
            st.download_button("EXPORT CSV", d2.to_csv(index=False),
                               f"{active_ticker}_prices.csv", "text/csv")

    with sub3:
        if not feature_df.empty:
            d3 = feature_df.copy()
            d3["date"] = pd.to_datetime(d3["date"]).dt.strftime("%Y-%m-%d")
            d3[d3.select_dtypes("number").columns] = d3.select_dtypes("number").round(4)
            st.dataframe(d3, use_container_width=True, hide_index=True)
            st.download_button("EXPORT CSV", d3.to_csv(index=False),
                               f"{active_ticker}_features.csv", "text/csv")
        else:
            st.info("Feature matrix requires at least 20 overlapping days of sentiment + price data.")

"""
database.py — SQLite database schema using SQLAlchemy ORM
Tables: reddit_posts, news_articles, daily_sentiment, stock_prices, model_results
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class RedditPost(Base):
    __tablename__ = "reddit_posts"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    post_id      = Column(String(50), unique=True, nullable=False)
    ticker       = Column(String(10), nullable=False)
    title        = Column(Text)
    body         = Column(Text)
    score        = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    subreddit    = Column(String(50))
    created_utc  = Column(DateTime)
    sentiment    = Column(Float)
    fetched_at   = Column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    ticker       = Column(String(10), nullable=False)
    title        = Column(Text)
    description  = Column(Text)
    source       = Column(String(100))
    published_at = Column(DateTime)
    sentiment    = Column(Float)
    fetched_at   = Column(DateTime, default=datetime.utcnow)


class DailySentiment(Base):
    __tablename__ = "daily_sentiment"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    ticker             = Column(String(10), nullable=False)
    date               = Column(DateTime, nullable=False)
    avg_sentiment      = Column(Float)
    weighted_sentiment = Column(Float)
    post_volume        = Column(Integer)
    news_sentiment     = Column(Float)
    combined_sentiment = Column(Float)


class StockPrice(Base):
    __tablename__ = "stock_prices"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(String(10), nullable=False)
    date       = Column(DateTime, nullable=False)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float)
    volume     = Column(Float)
    pct_change = Column(Float)


class ModelResult(Base):
    __tablename__ = "model_results"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(String(10))
    model_name = Column(String(50))
    accuracy   = Column(Float)
    precision  = Column(Float)
    recall     = Column(Float)
    f1_score   = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_path: str = "data/sentiment.db"):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: str = "data/sentiment.db"):
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()

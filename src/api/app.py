# src/api/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import requests
from dotenv import load_dotenv
import logging

# load environment variables from .env
load_dotenv()

# set up logging for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_ai_platform")

from src.utils.config import AnomalyScoreRequest, NewsSentimentRequest, RebalanceRequest
from src.anomaly.detector import SimpleAnomalyDetector
from src.nlp.news_sentiment import NewsSentiment
from src.rl.portfolio_agent import rebalance_opt
from src.forecasting.forecaster import forecast_intraday

# ================================
# 🚀 Initialize FastAPI App
# ================================
app = FastAPI(title="Finance AI Platform")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ================================
# 🚀 Bootstrapping anomaly detector
# ================================
bootstrap_df = pd.DataFrame({
    "amount":   [50, 100, 200, 500, 800, 1500, 5000, 20000, 50000, 120000],
    "hour":     [10, 12, 14, 18,  9,   21,   16,   20,    2,      3],
    "merchant": ["Flipkart", "Amazon", "Walmart", "Reliance", "Uber", "Zomato", "Myntra", "IRCTC", "Unknown", "Unknown"],
    "category": ["Shopping", "Electronics", "Groceries", "Travel", "Transport", "Food", "Fashion", "Travel", "Luxury", "Bills"],
    "location": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Delhi", "Pune", "Hyderabad", "Kolkata", "Delhi", "Pune"],
    "device_id": ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "dev8", "dev9", "dev10"],
    "currency": ["INR"] * 10,
    "channel": ["card", "card", "upi", "upi", "card", "upi", "card", "card", "wire", "wire"],
    "account_id": ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]
})
_detector = SimpleAnomalyDetector().fit(bootstrap_df)

# ✅ Sentiment Analyzer
_sentiment = NewsSentiment()

# ✅ News API Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

# ================================
# 🚀 Routes
# ================================

@app.get("/", response_class=HTMLResponse)
def dashboard():
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html")
    return FileResponse(file_path, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_anomaly(req: AnomalyScoreRequest):
    record = {
        "txn_id": req.txn_id,
        "account_id": req.account_id,
        "amount": req.amount,
        "currency": req.currency,
        "merchant": req.merchant,
        "category": req.category,
        "hour": req.hour,
        "channel": req.channel,
        "location": req.location,
        "device_id": req.device_id,
    }
    result = _detector.predict(record, threshold=req.threshold)
    return {"score": float(result["score"]), "status": result["status"]}


@app.post("/sentiment")
def news_sentiment(req: NewsSentimentRequest):
    return {"results": _sentiment.infer(req.texts)}


@app.get("/news-today")
def fetch_today_news():
    """
    Fetch today's business news and analyze sentiment automatically.
    Also groups sentiment by major stock/company names.
    """
    logger.info("Request /news-today received")

    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY not set in environment")
        raise HTTPException(
            status_code=500,
            detail="NEWS_API_KEY is not set. Put NEWS_API_KEY=your_key in .env and restart server."
        )

    params = {
        "category": "business",
        "language": "en",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }

    try:
        logger.info("Calling NewsAPI...")
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        logger.info(f"NewsAPI returned status {resp.status_code}")
    except Exception as e:
        logger.exception("Exception while calling NewsAPI")
        raise HTTPException(status_code=500, detail=f"Exception while fetching news: {str(e)}")

    try:
        data = resp.json()
    except Exception as e:
        logger.exception("Failed to parse JSON from NewsAPI")
        raise HTTPException(status_code=500, detail=f"Invalid JSON from NewsAPI (status={resp.status_code}): {str(e)}")

    if resp.status_code != 200:
        text_preview = (resp.text or "")[:300]
        logger.error(f"NewsAPI error: status={resp.status_code}, body={text_preview}")
        raise HTTPException(status_code=500, detail=f"NewsAPI error: status={resp.status_code}, body_preview={text_preview}")

    articles = data.get("articles", [])
    if not articles:
        logger.warning("No articles returned from NewsAPI")
        return {"news": [], "summary": {}}

    headlines = [a.get("title") for a in articles if a.get("title")]
    if not headlines:
        logger.warning("No headlines extracted")
        return {"news": [], "summary": {}}

    try:
        logger.info(f"Analyzing {len(headlines)} headlines")
        results = _sentiment.infer(headlines)
        combined = [{"headline": h, "sentiment": r} for h, r in zip(headlines, results)]

        # 🔥 Stock/company keywords to track
        keywords = ["Tesla", "Amazon", "Apple", "Microsoft", "NVIDIA", "Google", "Meta", "SoftBank", "Intel"]
        summary = {}

        for h, r in zip(headlines, results):
            for k in keywords:
                if k.lower() in h.lower():
                    if k not in summary:
                        summary[k] = {"positive": 0, "negative": 0}
                    if r["label"] == "POSITIVE":
                        summary[k]["positive"] += 1
                    else:
                        summary[k]["negative"] += 1

        return {"news": combined, "summary": summary}

    except Exception as e:
        logger.exception("Sentiment analysis failed")
        raise HTTPException(status_code=500, detail=f"Sentiment inference failed: {str(e)}")


@app.get("/forecast/{ticker}")
def forecast_stock(ticker: str):
    """
    Forecast next 10 hours (15m steps) for a stock ticker.
    """
    try:
        preds, actual = forecast_intraday(ticker)
        return {"ticker": ticker, "history": actual, "forecast": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebalance")
def rebalance(req: RebalanceRequest):
    return rebalance_opt(req.tickers, req.risk_target)

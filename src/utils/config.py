from pydantic import BaseModel
from typing import Optional, List


class AnomalyScoreRequest(BaseModel):
    txn_id: Optional[str] = None
    account_id: Optional[str] = None
    amount: float
    currency: str = "INR"
    merchant: str = "UNKNOWN"
    category: str = "UNKNOWN"
    hour: int = 12
    channel: str = "card"
    location: str = "UNKNOWN"     # ✅ New: transaction location
    device_id: str = "UNKNOWN"    # ✅ New: device identifier
    threshold: float = 0.8        # ✅ New: configurable threshold


class NewsSentimentRequest(BaseModel):
    texts: List[str]


class RebalanceRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    risk_target: float = 0.10

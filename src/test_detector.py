import pandas as pd
from anomaly.detector import SimpleAnomalyDetector

# Load your CSV
df = pd.read_csv("transactions.csv")

# Train the detector
detector = SimpleAnomalyDetector().fit(df)

# Test prediction
print(detector.predict({
    "amount": 99999,
    "hour": 3,
    "merchant": "BTCEX",
    "category": "CRYPTO",
    "channel": "wire",
    "currency": "INR",
    "account_id": "1003"
}))

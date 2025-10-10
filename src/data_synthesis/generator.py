# generator.py

import numpy as np
import pandas as pd


def synthesize_transactions(n=1000, fraud_ratio=0.02, seed=42):
    rng = np.random.default_rng(seed)

    # --- Continuous features ---
    amounts = np.exp(rng.normal(6, 1.0, size=n))  # log-normal distribution
    hours = rng.integers(0, 24, size=n)

    # --- Fraud labels ---
    is_fraud = (rng.random(size=n) < fraud_ratio).astype(int)

    # --- Categorical features ---
    merchants = rng.choice(
        ["Amazon", "Walmart", "Target", "Flipkart", "Uber", "Starbucks"],
        size=n
    )
    categories = rng.choice(
        ["groceries", "electronics", "fashion", "fuel", "entertainment", "travel"],
        size=n
    )
    locations = rng.choice(
        ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune"],
        size=n
    )
    device_ids = rng.choice(
        [f"device_{i}" for i in range(1, 21)],
        size=n
    )

    df = pd.DataFrame({
        "amount": amounts,
        "hour": hours,
        "merchant": merchants,
        "category": categories,
        "location": locations,
        "device_id": device_ids,
        "is_fraud": is_fraud
    })

    # --- Fraud amplification rules ---
    # Fraudulent transactions often involve unusual amounts or odd context
    df.loc[df["is_fraud"] == 1, "amount"] *= rng.uniform(3, 10)  # big spike
    df.loc[df["is_fraud"] == 1, "hour"] = rng.choice([1, 2, 3, 4])  # odd hours
    df.loc[df["is_fraud"] == 1, "location"] = rng.choice(["Unknown", "Offshore", "RandomCity"])
    df.loc[df["is_fraud"] == 1, "device_id"] = rng.choice(["stolen_device", "new_device", "proxy_device"])

    return df

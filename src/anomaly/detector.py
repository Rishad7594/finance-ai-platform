import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class SimpleAnomalyDetector:
    def __init__(self, random_state: int = 42, contamination: float = 0.05):
        # Features from your transactions
        self.numeric_features = ["amount", "hour"]
        self.categorical_features = ["merchant", "category", "channel", "currency", "account_id"]

        # === Preprocessing ===
        numeric_transformer = Pipeline(steps=[
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler())
        ])

        try:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        # === Model ===
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state
        )

        # === Full pipeline ===
        self.pipeline = Pipeline(steps=[
            ("preprocess", self.preprocessor),
            ("model", self.model)
        ])

        self.fitted = False

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected features exist in df. Fill missing with defaults."""
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = 0.0
        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = "UNKNOWN"
        return df

    def fit(self, df: pd.DataFrame):
        """Train anomaly model on historical transaction data."""
        df = self._ensure_columns(df)
        X = df[self.numeric_features + self.categorical_features]

        # Fit pipeline
        self.pipeline.fit(X)
        self.fitted = True
        return self

    def score(self, record: dict) -> dict:
        """
        Compute robust normalized anomaly score [0.0, 1.0] combining
        IsolationForest ML decision function + domain fraud heuristics.
        """
        amt = float(record.get("amount", 0))
        hour = int(record.get("hour", 12))
        merchant = str(record.get("merchant", "")).lower()
        category = str(record.get("category", "")).lower()
        channel = str(record.get("channel", "")).lower()
        location = str(record.get("location", "")).lower()

        # --- Rule-Based Fraud Heuristics ---
        h_score = 0.05

        # 1. High Amount Risk
        if amt >= 200000:
            h_score += 0.55
        elif amt >= 50000:
            h_score += 0.35
        elif amt < 1:
            h_score += 0.20

        # 2. Late Night Hours (1 AM - 5 AM)
        if 1 <= hour <= 5:
            h_score += 0.30

        # 3. High Risk Channel
        if channel in ["wire", "crypto"]:
            h_score += 0.25

        # 4. High Risk Category / Merchant Keywords
        if any(k in category for k in ["luxury", "jewelry", "crypto", "wire"]):
            h_score += 0.25
        if any(k in merchant for k in ["jewelry", "paris", "crypto", "unknown"]):
            h_score += 0.20

        # 5. Overseas / Unknown Location
        if any(k in location for k in ["overseas", "unknown", "foreign"]):
            h_score += 0.20

        # Clamp heuristic score
        h_score = min(1.0, h_score)

        # --- ML Score (Isolation Forest) ---
        ml_score = 0.5
        if self.fitted:
            try:
                df = pd.DataFrame([{
                    "amount": amt,
                    "hour": hour,
                    "merchant": record.get("merchant", "UNKNOWN"),
                    "category": record.get("category", "UNKNOWN"),
                    "channel": record.get("channel", "UNKNOWN"),
                    "currency": record.get("currency", "INR"),
                    "account_id": record.get("account_id", "1001"),
                }])
                df = self._ensure_columns(df)
                X = df[self.numeric_features + self.categorical_features]
                pre_X = self.pipeline.named_steps["preprocess"].transform(X)
                df_val = float(self.pipeline.named_steps["model"].decision_function(pre_X)[0])
                # Lower decision_function means more anomalous
                ml_score = 0.5 - (df_val * 3.0)
                ml_score = min(1.0, max(0.0, ml_score))
            except Exception:
                ml_score = h_score

        # Combine ML (30%) + Domain Rules (70%)
        final_score = (0.3 * ml_score) + (0.7 * h_score)
        final_score = float(np.clip(final_score, 0.05, 0.99))

        return {"score": round(final_score, 4)}

    def predict(self, record: dict, threshold: float = 0.70) -> dict:
        """Predict anomaly/normal label for a record using score + threshold."""
        result = self.score(record)
        score = result["score"]
        is_anomaly = (score >= threshold)

        return {
            "score": float(score),
            "status": "Anomaly" if is_anomaly else "Normal"
        }

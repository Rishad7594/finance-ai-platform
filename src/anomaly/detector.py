import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class SimpleAnomalyDetector:
    def __init__(self, random_state: int = 42, contamination: float = 0.02):
        # Features from your transactions.csv
        self.numeric_features = ["amount", "hour"]
        self.categorical_features = ["merchant", "category", "channel", "currency", "account_id"]

        # === Preprocessing ===
        numeric_transformer = Pipeline(steps=[
            ("log", FunctionTransformer(np.log1p, validate=False)),  # reduce skew
            ("scale", StandardScaler())
        ])

        try:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False  # ✅ sklearn >=1.2
            )
        except TypeError:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False  # ✅ fallback sklearn <1.2
            )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        # === Model ===
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state
        )

        # === Full pipeline ===
        self.pipeline = Pipeline(steps=[
            ("preprocess", self.preprocessor),
            ("model", self.model)
        ])

        # === Score scaler ===
        self.scaler = MinMaxScaler(feature_range=(0, 1))
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

        # Normalize decision function scores
        pre_X = self.pipeline.named_steps["preprocess"].transform(X)
        raw_scores = -self.pipeline.named_steps["model"].decision_function(pre_X).reshape(-1, 1)
        self.scaler.fit(raw_scores)

        self.fitted = True
        return self

    def score(self, record: dict) -> dict:
        """Compute normalized anomaly score for a single transaction [0,1]."""
        if not self.fitted:
            amt = float(record.get("amount", 0))
            hour = int(record.get("hour", 12))
            is_anomaly = (amt > 50000) or (amt < 1) or (hour < 5)
            return {"score": 1.0 if is_anomaly else 0.0, "anomaly": bool(is_anomaly)}

        df = pd.DataFrame([{
            "amount": float(record.get("amount", 0)),
            "hour": int(record.get("hour", 12)),
            "merchant": record.get("merchant", "UNKNOWN"),
            "category": record.get("category", "UNKNOWN"),
            "channel": record.get("channel", "UNKNOWN"),
            "currency": record.get("currency", "UNKNOWN"),
            "account_id": record.get("account_id", "UNKNOWN"),
        }])

        df = self._ensure_columns(df)
        X = df[self.numeric_features + self.categorical_features]

        pre_X = self.pipeline.named_steps["preprocess"].transform(X)
        raw_score = -self.pipeline.named_steps["model"].decision_function(pre_X)[0]
        norm_score = float(self.scaler.transform([[raw_score]])[0][0])

        return {"score": norm_score}

    def predict(self, record: dict, threshold: float = 0.5) -> dict:
        """Predict anomaly/normal label for a record using score + threshold."""
        result = self.score(record)
        score = result["score"]
        is_anomaly = (score >= threshold)

        return {
            "score": float(score),
            "status": "Anomaly" if is_anomaly else "Normal"
        }

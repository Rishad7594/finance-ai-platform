import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

from typing import List, Dict

# Use VADER — a lightweight, rule-based sentiment analyzer designed for news/social media.
# Uses ~2MB RAM instead of DistilBERT's ~340MB (PyTorch + model weights).
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NewsSentiment:
    def __init__(self, model_name: str = "vader"):
        self.analyzer = SentimentIntensityAnalyzer()

    def infer(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            scores = self.analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "POSITIVE"
            elif compound <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            results.append({
                "label": label,
                "score": round(abs(compound), 4)
            })
        return results

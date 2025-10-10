import os
os.environ["TRANSFORMERS_NO_TF"] = "1"    # 🚫 Disable TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  # 🚫 Disable Flax
os.environ["TRANSFORMERS_NO_JAX"] = "1"   # 🚫 Disable JAX


from typing import List, Dict
from transformers import pipeline

class NewsSentiment:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # explicitly choose PyTorch backend
        self.pipe = pipeline("sentiment-analysis", model=model_name, framework="pt")

    def infer(self, texts: List[str]) -> List[Dict]:
        return self.pipe(texts)

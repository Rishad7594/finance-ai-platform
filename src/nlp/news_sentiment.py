import os
os.environ["TRANSFORMERS_NO_TF"] = "1"    # 🚫 Disable TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  # 🚫 Disable Flax
os.environ["TRANSFORMERS_NO_JAX"] = "1"   # 🚫 Disable JAX


from typing import List, Dict
from transformers import pipeline

class NewsSentiment:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.pipe = None

    def _load_model(self):
        if self.pipe is None:
            self.pipe = pipeline("sentiment-analysis", model=self.model_name, framework="pt")

    def infer(self, texts: List[str]) -> List[Dict]:
        self._load_model()
        return self.pipe(texts)

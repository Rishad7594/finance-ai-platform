# Holistic AI-Driven Financial Risk & Fraud Platform (Starter Kit)

An end-to-end, modular AI system for **fraud/anomaly detection**, **news/NLP sentiment**, and **reinforcement-learning-based portfolio risk management**.

## Modules
- `data_synthesis/` — GAN/VAE-based synthetic transaction generator.
- `anomaly/` — Hybrid anomaly detection (Isolation Forest + Autoencoder).
- `nlp/` — Financial news sentiment via Transformers.
- `rl/` — Portfolio environment + PPO agent (stable-baselines3).
- `api/` — FastAPI service exposing `/score`, `/sentiment`, `/rebalance` endpoints.

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download a small Transformer model once
python -m nltk.downloader punkt
python -c "import spacy; spacy.download('en_core_web_sm')"

# Run API
uvicorn api.app:app --reload --port 8000
```

## Minimal Workplan
1) **MVP**
   - Rule + IsolationForest for anomaly scoring on sample transactions.
   - FinBERT (or any small sentiment model) pipeline for news sentiment.
   - Stub RL env returning random actions, wire PPO later.
   - FastAPI with three endpoints + pydantic schemas.
2) **Phase 2**
   - Add Autoencoder; calibrate thresholds with precision-recall.
   - Replace generic sentiment with domain model; add entity linking.
   - Implement PPO or SAC; reward = risk-adjusted return + anomaly penalty.
3) **Phase 3**
   - Synthetic data (GAN/VAE) for rare fraud patterns.
   - Model registry, experiment tracking, monitoring dashboards.

## Safety & Compliance
- PII minimization, encryption at rest/in transit, role-based auth.
- Model bias testing; human-in-the-loop overrides.
- Clear audit logs for decisions.

## Folder Layout
```
src/
  anomaly/detector.py
  data_synthesis/generator.py
  nlp/news_sentiment.py
  rl/portfolio_agent.py
  api/app.py
  utils/config.py
sample_data/transactions.csv
tests/
docker/Dockerfile
.env.example
```

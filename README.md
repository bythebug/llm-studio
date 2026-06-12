# LLM Studio

A production-ready backend system for fine-tuning large language models. Upload training data, kick off a training job, evaluate the results, and serve predictions — all through a clean REST API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        REST API (FastAPI)                │
├──────────────┬──────────────┬──────────────┬────────────┤
│ Job Manager  │ Data Upload  │  Evaluation  │ Inference  │
├──────────────┴──────────────┴──────────────┴────────────┤
│              Training Pipeline (PyTorch + HuggingFace)  │
├─────────────────────────────────────────────────────────┤
│   PostgreSQL DB   │   MLflow Tracking   │   Prometheus  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-studio/
├── llm_studio/
│   ├── models.py            # SQLAlchemy ORM (User, Job, TrainingData, ModelVersion, Prediction)
│   ├── config.py            # Base models, hyperparameters, storage, MLflow
│   ├── data_loader.py       # CSV/JSON ingestion, validation, cleaning, splitting
│   ├── preprocessor.py      # Tokenization, padding, normalization pipeline
│   ├── trainer.py           # Training loop with MLflow integration
│   ├── loss_functions.py    # Cross-entropy loss, perplexity, metrics tracker
│   ├── optimizer.py         # AdamW + linear warmup / cosine decay scheduler
│   ├── evaluator.py         # Generation & classification evaluation
│   ├── metrics.py           # BLEU, accuracy, F1, confusion matrix, interpretation
│   ├── comparator.py        # Version comparison + statistical significance
│   ├── model_loader.py      # LRU in-memory model cache
│   ├── inference.py         # Single & batch prediction with confidence scoring
│   ├── mlflow_integration.py# Experiment tracking wrapper
│   └── monitoring.py        # Prometheus metrics, drift detection, prediction audit log
├── db/
│   └── schema.sql           # PostgreSQL DDL
├── tests/
│   ├── test_data_loading.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   ├── test_mlflow.py
│   ├── test_edge_cases.py
│   └── test_integration.py
├── app.py                   # FastAPI application
├── docker-compose.yml       # MLflow + App + Prometheus + Grafana + PostgreSQL
├── Dockerfile
├── prometheus.yml
├── deploy.sh                # AWS ECR/ECS deployment
├── TRAINING_GUIDE.md
├── API_DOCS.md
└── requirements.txt
```

---

## Supported Base Models

| Model | HuggingFace ID | Context | Family |
|---|---|---|---|
| `gpt2` | `gpt2` | 1024 tokens | Causal LM |
| `gpt2-medium` | `gpt2-medium` | 1024 tokens | Causal LM |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B` | 8192 tokens | Causal LM |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | 4096 tokens | Causal LM |
| `t5-small` | `t5-small` | 512 tokens | Seq2Seq |

---

## Quick Start

### 1. Local development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL=sqlite:///dev.db
uvicorn app:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 2. Docker (full stack)

```bash
docker-compose up -d
# API:        http://localhost:8000
# MLflow UI:  http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### 3. Apply database schema

```bash
psql $DATABASE_URL < db/schema.sql
```

---

## Training Pipeline Walkthrough

### Step 1 — Create a user and job

```bash
# Create a job for user 1 using GPT-2
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "model_name": "gpt2"}'
# → {"job_id": 1, "status": "queued", "model_name": "gpt2"}
```

### Step 2 — Upload training data

```bash
# CSV format: input_text, expected_output
curl -X POST http://localhost:8000/jobs/1/upload_data \
  -F "file=@training_data.csv"
# → {"job_id": 1, "rows_added": 500}
```

### Step 3 — Start training

```bash
curl -X POST http://localhost:8000/jobs/1/start_training
# → {"job_id": 1, "message": "Training started"}
```

### Step 4 — Monitor progress

```bash
curl http://localhost:8000/jobs/1/status
# → {"status": "training", "loss_curves": {"train_loss": [...], "val_loss": [...]}}

curl http://localhost:8000/jobs/1/logs
# → {"logs": [{"timestamp": "...", "message": "Epoch 1 complete — train_loss=2.41"}]}
```

### Step 5 — Evaluate

```bash
curl http://localhost:8000/jobs/1/metrics
# → {"versions": [{"version_num": 1, "training_loss": 1.82, "evaluation": {...}}]}
```

### Step 6 — Run inference

```bash
curl -X POST http://localhost:8000/jobs/1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "Translate to French: Hello world"}'
# → {"output": "Bonjour le monde", "confidence": 0.87, "latency_ms": 142}
```

---

## API Reference

See [API_DOCS.md](API_DOCS.md) for the full reference with request/response examples.

| Category | Endpoint | Method |
|---|---|---|
| Jobs | `/jobs` | POST |
| Jobs | `/jobs/{id}/start_training` | POST |
| Jobs | `/jobs/{id}/status` | GET |
| Jobs | `/jobs/{id}/logs` | GET |
| Data | `/jobs/{id}/upload_data` | POST |
| Data | `/jobs/{id}/data_preview` | GET |
| Data | `/jobs/{id}/data_stats` | GET |
| Evaluation | `/jobs/{id}/metrics` | GET |
| Evaluation | `/jobs/{id}/model_comparison` | GET |
| Evaluation | `/models/{version_id}/confusion_matrix` | GET |
| Inference | `/jobs/{id}/predict` | POST |
| Inference | `/jobs/{id}/predict_batch` | POST |
| Inference | `/jobs/{id}/models` | GET |
| Experiments | `/experiments` | GET |
| Experiments | `/experiments/{id}` | GET |
| Experiments | `/experiments/compare` | POST |
| Monitoring | `/jobs/{id}/monitoring` | GET |
| Monitoring | `/metrics` | GET |

---

## Running Tests

```bash
# Full suite (141 tests)
venv/bin/pytest tests/ -q

# By category
venv/bin/pytest tests/test_data_loading.py   # Data pipeline
venv/bin/pytest tests/test_training.py        # Training loop
venv/bin/pytest tests/test_evaluation.py      # Metrics & comparison
venv/bin/pytest tests/test_inference.py       # Inference & caching
venv/bin/pytest tests/test_mlflow.py          # Experiment tracking
venv/bin/pytest tests/test_edge_cases.py      # Edge cases
venv/bin/pytest tests/test_integration.py     # End-to-end
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://localhost/llm_studio` | SQLAlchemy DB connection |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `MLFLOW_EXPERIMENT_NAME` | `llm-studio` | Experiment name |
| `STORAGE_BASE` | `./storage` | Model/checkpoint/log root |

---

## Data Formats

**CSV** (`input_text`, `expected_output` columns required):
```csv
input_text,expected_output
"Translate: Hello","Bonjour"
"Summarize: Long article...","Short summary."
```

**JSON** (array of `{input, output}` objects):
```json
[
  {"input": "Translate: Hello", "output": "Bonjour"},
  {"input": "Summarize: Long article...", "output": "Short summary."}
]
```

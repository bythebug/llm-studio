# LLM Studio

A production-ready backend system for fine-tuning large language models. Upload training data, run a training job, evaluate results, and serve predictions — all through a REST API with experiment tracking and production monitoring built in.

---

## Features

- **Data ingestion** — upload CSV or JSON training pairs with validation, cleaning, and 80/10/10 splitting
- **Fine-tuning** — full PyTorch training loop with AdamW + warmup/cosine LR scheduling, gradient clipping, and gradient accumulation
- **Experiment tracking** — every run logged to MLflow: hyperparameters, per-epoch metrics, model artifacts
- **Evaluation** — BLEU, perplexity, accuracy, F1, confusion matrix; statistical significance testing across versions
- **Inference** — single and batch prediction with LRU model cache and per-token confidence scoring
- **Monitoring** — Prometheus metrics, latency percentiles (p50/p95/p99), confidence drift detection, full prediction audit log
- **Deployment** — Docker Compose stack + AWS ECR/ECS deploy script

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.136, Uvicorn |
| ML | PyTorch 2.12, HuggingFace Transformers 5.12 |
| Database | PostgreSQL (prod) / SQLite (dev), SQLAlchemy 2.0 |
| Experiment tracking | MLflow 3.13 (SQLite backend) |
| Metrics | Prometheus client, Grafana |
| Data | Pandas 3.0, NLTK, sacreBLEU |
| Evaluation | scikit-learn 1.9, SciPy 1.17 |
| Testing | pytest, 141 tests across 7 files |
| Deployment | Docker, AWS ECR + ECS |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI  (app.py)                     │
├────────────┬─────────────┬──────────────┬────────────────────┤
│ Job Mgmt   │ Data Upload │  Evaluation  │ Inference          │
├────────────┴─────────────┴──────────────┴────────────────────┤
│              Training Pipeline  (trainer.py)                 │
│   PyTorch loop · AdamW · LR scheduler · MLflow logging       │
├──────────────────────┬───────────────────────────────────────┤
│  PostgreSQL / SQLite │  MLflow (experiments + artifacts)     │
├──────────────────────┴───────────────────────────────────────┤
│         Prometheus metrics  ←  Grafana dashboards            │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-studio/
├── llm_studio/
│   ├── models.py             # SQLAlchemy ORM — User, FineTuningJob, TrainingData,
│   │                         #   ModelVersion, Prediction
│   ├── config.py             # Base models registry, TrainingConfig, StorageConfig, MLflow
│   ├── data_loader.py        # CSV/JSON ingestion, cleaning, validation, 80/10/10 split
│   ├── preprocessor.py       # Normalize → tokenize → encode → pad pipeline
│   ├── trainer.py            # Training loop with MLflow integration
│   ├── loss_functions.py     # Cross-entropy (causal LM shift), perplexity, MetricsTracker
│   ├── optimizer.py          # AdamW with param splitting + warmup/cosine scheduler
│   ├── evaluator.py          # Run model on test set, persist results as JSON
│   ├── metrics.py            # BLEU, accuracy, F1, confusion matrix, interpretation guides
│   ├── comparator.py         # Rank versions, Welch's t-test significance
│   ├── model_loader.py       # LRU in-memory cache (OrderedDict, max 3 models)
│   ├── inference.py          # Greedy decode, batch inference, per-token confidence
│   ├── mlflow_integration.py # ExperimentTracker context manager + query helpers
│   └── monitoring.py         # Prometheus counters/histograms, DriftDetector, PredictionLogger
├── db/
│   └── schema.sql            # PostgreSQL DDL — tables, indexes, updated_at trigger
├── tests/
│   ├── test_data_loading.py  # 27 tests — CSV/JSON loading, split, tokenization
│   ├── test_training.py      # 13 tests — training loop, loss, checkpointing
│   ├── test_evaluation.py    # 27 tests — metrics, comparison, significance
│   ├── test_inference.py     # 18 tests — prediction, LRU cache, latency < 100ms
│   ├── test_mlflow.py        # 16 tests — param/metric/artifact logging
│   ├── test_edge_cases.py    # 31 tests — empty data, unicode, long text, null bytes
│   └── test_integration.py   # 9 tests  — full pipeline, status transitions
├── app.py                    # FastAPI application (19 endpoints)
├── docker-compose.yml        # PostgreSQL + MLflow + App + Prometheus + Grafana
├── Dockerfile
├── prometheus.yml
├── deploy.sh                 # AWS ECR → ECS deploy with pre-flight test run
├── TRAINING_GUIDE.md         # Data prep, hyperparameter tuning, loss curve guide
├── API_DOCS.md               # Full endpoint reference with request/response examples
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

### Local development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL=sqlite:///dev.db
uvicorn app:app --reload
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Interactive docs | http://localhost:8000/docs |

### Docker — full stack

```bash
docker-compose up -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

### Apply database schema (PostgreSQL)

```bash
psql $DATABASE_URL < db/schema.sql
```

---

## Training Pipeline Walkthrough

```bash
# 1. Create a fine-tuning job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "model_name": "gpt2"}'
# → {"job_id": 1, "status": "queued"}

# 2. Upload training data (CSV or JSON)
curl -X POST http://localhost:8000/jobs/1/upload_data \
  -F "file=@training_data.csv"
# → {"rows_added": 500}

# 3. Start training (runs in background)
curl -X POST http://localhost:8000/jobs/1/start_training
# → {"message": "Training started"}

# 4. Check status + live loss curves
curl http://localhost:8000/jobs/1/status
# → {"status": "training", "loss_curves": {"train_loss": [3.1, 2.4], "val_loss": [3.3, 2.6]}}

# 5. View evaluation metrics
curl http://localhost:8000/jobs/1/metrics
# → {"versions": [{"version_num": 1, "evaluation": {"bleu": 28.4, "perplexity": 42.1}}]}

# 6. Run inference against best model version
curl -X POST http://localhost:8000/jobs/1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "Translate to French: Hello world"}'
# → {"output": "Bonjour le monde", "confidence": 0.87, "latency_ms": 142}
```

---

## API Reference

Full reference with request/response examples: [API_DOCS.md](API_DOCS.md)

| Category | Endpoint | Method | Description |
|---|---|---|---|
| Jobs | `/jobs` | POST | Create job |
| Jobs | `/jobs/{id}/start_training` | POST | Start training |
| Jobs | `/jobs/{id}/status` | GET | Status + loss curves |
| Jobs | `/jobs/{id}/logs` | GET | Training logs |
| Data | `/jobs/{id}/upload_data` | POST | Upload CSV or JSON |
| Data | `/jobs/{id}/data_preview` | GET | First N rows |
| Data | `/jobs/{id}/data_stats` | GET | Row count, vocab size, split sizes |
| Evaluation | `/jobs/{id}/metrics` | GET | Metrics for all versions |
| Evaluation | `/jobs/{id}/model_comparison` | GET | Rank versions, recommendation |
| Evaluation | `/models/{version_id}/confusion_matrix` | GET | Classification matrix |
| Inference | `/jobs/{id}/predict` | POST | Single prediction |
| Inference | `/jobs/{id}/predict_batch` | POST | Batch predictions |
| Inference | `/jobs/{id}/models` | GET | List versions + cache status |
| Experiments | `/experiments` | GET | List MLflow experiments |
| Experiments | `/experiments/{id}` | GET | Runs in an experiment |
| Experiments | `/experiments/compare` | POST | Diff runs side-by-side |
| Monitoring | `/jobs/{id}/monitoring` | GET | Latency stats + drift alerts |
| Monitoring | `/metrics` | GET | Prometheus scrape endpoint |

---

## Running Tests

```bash
# Full suite
venv/bin/pytest tests/ -q
# 141 passed

# By file
venv/bin/pytest tests/test_data_loading.py  # Data pipeline (27)
venv/bin/pytest tests/test_training.py       # Training loop (13)
venv/bin/pytest tests/test_evaluation.py     # Metrics & comparison (27)
venv/bin/pytest tests/test_inference.py      # Inference & caching (18)
venv/bin/pytest tests/test_mlflow.py         # Experiment tracking (16)
venv/bin/pytest tests/test_edge_cases.py     # Edge cases — unicode, empty, long text (31)
venv/bin/pytest tests/test_integration.py    # End-to-end pipeline (9)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://localhost/llm_studio` | SQLAlchemy connection string |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `MLFLOW_EXPERIMENT_NAME` | `llm-studio` | Experiment name |
| `STORAGE_BASE` | `./storage` | Root for model weights, logs, eval results |

---

## Data Formats

**CSV** — columns `input_text` and `expected_output`:
```csv
input_text,expected_output
"Translate: Hello","Bonjour"
"Summarize: Long article...","Short summary."
```

**JSON** — array of `{ "input", "output" }` objects:
```json
[
  {"input": "Translate: Hello", "output": "Bonjour"},
  {"input": "Summarize: Long article...", "output": "Short summary."}
]
```

---

## AWS Deployment

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
./deploy.sh          # CPU
./deploy.sh --gpu    # GPU (uses Dockerfile.gpu)
```

The script runs the full test suite, builds and pushes to ECR, registers a new ECS task definition, and waits for the service to stabilise.

---

## Documentation

| File | Contents |
|---|---|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Data prep checklist, hyperparameter tuning, loss curve patterns, early stopping |
| [API_DOCS.md](API_DOCS.md) | All endpoints with full request/response JSON and error codes |

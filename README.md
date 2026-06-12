# LLM Studio

A full-stack platform for fine-tuning large language models. Upload training data, run training jobs locally or on a remote GPU over SSH, evaluate results, and serve predictions — with a Next.js dashboard, REST API, experiment tracking, and production monitoring built in.

---

## Features

- **Data ingestion** — upload CSV or JSON training pairs with validation, cleaning, and 80/10/10 splitting
- **Fine-tuning** — full PyTorch training loop with AdamW + warmup/cosine LR scheduling, gradient clipping, and gradient accumulation
- **Remote compute** — SSH into any GPU machine; training data and script are uploaded automatically, logs stream back live, model artifacts are downloaded when done
- **Experiment tracking** — every run logged to MLflow: hyperparameters, per-epoch metrics, model artifacts
- **Evaluation** — BLEU, perplexity, accuracy, F1, confusion matrix; statistical significance testing across versions
- **Inference** — single and batch prediction with LRU model cache and per-token confidence scoring
- **Monitoring** — Prometheus metrics, latency percentiles (p50/p95/p99), confidence drift detection, full prediction audit log
- **Frontend** — Next.js dashboard: job management, data upload, live loss curves, training logs, inference playground, compute instance management
- **Sample jobs** — one-click "Load Samples" seeds two pre-loaded jobs (translation + summarisation) with training data ready to go
- **Model download** — download any trained model version as a zip archive; standard HuggingFace checkpoint, usable anywhere
- **Deployment** — Docker Compose stack (6 services) + AWS ECR/ECS deploy script

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React 19, Tailwind CSS, lucide-react |
| API | FastAPI 0.136, Uvicorn, Pydantic v2 |
| ML | PyTorch 2.12, HuggingFace Transformers 5.12, Accelerate |
| Remote compute | paramiko (SSH + SFTP) |
| Database | PostgreSQL 16 (prod) / SQLite (dev), SQLAlchemy 2.0 |
| Experiment tracking | MLflow 3.13 |
| Metrics | Prometheus client, Grafana |
| Data | Pandas, NLTK, sacreBLEU |
| Evaluation | scikit-learn 1.9, SciPy 1.17 |
| Testing | pytest — 141 tests across 7 files |
| Deployment | Docker Compose, AWS ECR + ECS |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Next.js Frontend  (:3001)                     │
│  Jobs · Data · Training · Evaluation · Inference · Compute      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST
┌───────────────────────────▼─────────────────────────────────────┐
│                    FastAPI  (app.py)  (:8000)                    │
├──────────┬──────────┬───────────┬─────────────┬─────────────────┤
│ Job Mgmt │   Data   │ Inference │ Experiments │ Compute (SSH)   │
├──────────┴──────────┴───────────┴─────────────┴─────────────────┤
│           Training Pipeline  (trainer.py / remote_runner.py)    │
│    Local: PyTorch loop · AdamW · MLflow logging                 │
│    Remote: paramiko SSH → SFTP upload → stream logs → download  │
├─────────────────────────┬───────────────────────────────────────┤
│  PostgreSQL / SQLite    │  MLflow  (:5001)                      │
├─────────────────────────┴───────────────────────────────────────┤
│         Prometheus  (:9090)  ←  Grafana dashboards  (:3000)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-studio/
├── frontend/                     # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx              # Jobs list
│   │   ├── jobs/[id]/page.tsx    # Job detail (6 tabs)
│   │   ├── compute/page.tsx      # SSH compute instances
│   │   ├── experiments/page.tsx  # MLflow experiments
│   │   └── layout.tsx            # Nav + footer
│   ├── components/
│   │   ├── LossCurveChart.tsx
│   │   └── StatusBadge.tsx
│   └── lib/api.ts                # Typed fetch wrappers
├── llm_studio/
│   ├── models.py                 # SQLAlchemy ORM — User, FineTuningJob, TrainingData,
│   │                             #   ModelVersion, Prediction, ComputeInstance
│   ├── config.py                 # Base models registry, TrainingConfig, StorageConfig
│   ├── data_loader.py            # CSV/JSON ingestion, cleaning, validation, split
│   ├── preprocessor.py           # Normalize → tokenize → encode → pad pipeline
│   ├── trainer.py                # Training loop with MLflow integration
│   ├── loss_functions.py         # Cross-entropy (causal LM shift), perplexity
│   ├── optimizer.py              # AdamW with param splitting + warmup/cosine scheduler
│   ├── evaluator.py              # Run model on test set, persist results as JSON
│   ├── metrics.py                # BLEU, accuracy, F1, confusion matrix
│   ├── comparator.py             # Rank versions, Welch's t-test significance
│   ├── model_loader.py           # LRU in-memory cache (OrderedDict, max 3 models)
│   ├── inference.py              # Greedy decode, batch inference, per-token confidence
│   ├── mlflow_integration.py     # ExperimentTracker context manager + query helpers
│   ├── monitoring.py             # Prometheus counters/histograms, DriftDetector
│   ├── remote_runner.py          # paramiko SSH: upload → stream → download artifacts
│   └── scripts/
│       └── remote_train.py       # Self-contained training script run on remote GPU
├── db/
│   └── schema.sql                # PostgreSQL DDL — tables, indexes, updated_at trigger
├── tests/
│   ├── test_data_loading.py      # 27 tests — CSV/JSON loading, split, tokenization
│   ├── test_training.py          # 13 tests — training loop, loss, checkpointing
│   ├── test_evaluation.py        # 27 tests — metrics, comparison, significance
│   ├── test_inference.py         # 18 tests — prediction, LRU cache, latency < 100ms
│   ├── test_mlflow.py            # 16 tests — param/metric/artifact logging
│   ├── test_edge_cases.py        # 31 tests — empty data, unicode, long text, null bytes
│   └── test_integration.py       # 9 tests  — full pipeline, status transitions
├── app.py                        # FastAPI application (28 endpoints)
├── docker-compose.yml            # 6-service stack
├── Dockerfile                    # Python 3.11-slim API image
├── prometheus.yml
├── deploy.sh                     # AWS ECR → ECS deploy with pre-flight tests
├── TRAINING_GUIDE.md
├── API_DOCS.md
└── requirements.txt
```

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
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3001 |
| API | http://localhost:8000 |
| MLflow UI | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

> `NEXT_PUBLIC_API_URL` is passed as a Docker build arg (default `http://localhost:8000`) and baked into the frontend bundle at build time. If you run the API on a different port, update the `args` value in `docker-compose.yml` and rebuild the frontend image.

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

# 3a. Start training locally
curl -X POST http://localhost:8000/jobs/1/start_training_remote \
  -H "Content-Type: application/json" \
  -d '{"compute_id": null}'

# 3b. Or train on a remote GPU (register instance first via /compute)
curl -X POST http://localhost:8000/jobs/1/start_training_remote \
  -H "Content-Type: application/json" \
  -d '{"compute_id": 1}'

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

## Remote Compute

Register any SSH-accessible machine as a compute backend:

```bash
# Register a GPU instance
curl -X POST http://localhost:8000/compute \
  -H "Content-Type: application/json" \
  -d '{"name": "A100 box", "host": "192.168.1.50", "username": "ubuntu", "key_path": "~/.ssh/id_rsa"}'

# Test the connection
curl -X POST http://localhost:8000/compute/1/test
# → {"success": true, "message": "Python 3.11.4\npip 23.0"}
```

**How it works:**
1. LLM Studio SSHes into the instance using your key
2. Uploads training data and a self-contained training script via SFTP
3. Installs PyTorch + Transformers on the remote (cached after first run)
4. Runs training; logs stream back in real time
5. Downloads model artifacts back via SFTP when complete

Requirements on the remote: Python 3.8+, pip, internet access for the initial `pip install`.

---

## Base Models

You never upload a base model. When training starts, HuggingFace's `from_pretrained` checks the local cache first, then downloads from huggingface.co if needed. The downloaded model is cached on disk so subsequent runs are instant.

| Model | HuggingFace ID | Context | Auth required |
|---|---|---|---|
| `gpt2` | `gpt2` | 1024 tokens | None |
| `gpt2-medium` | `gpt2-medium` | 1024 tokens | None |
| `t5-small` | `t5-small` | 512 tokens | None |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B` | 8192 tokens | HF token + Meta approval |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | 4096 tokens | HF token + Mistral approval |

For gated models set `HUGGING_FACE_HUB_TOKEN` in your environment before starting training.

### Adding a new base model

Edit **one file** — `llm_studio/config.py`:

```python
BASE_MODELS = {
    ...
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "max_tokens": 4096,
        "family": "causal-lm",
    },
}
```

The `GET /base-models` endpoint reads directly from this dict, and the frontend dropdown fetches it on load — nothing else to change.

---

## API Reference

Full reference: [API_DOCS.md](API_DOCS.md)

| Category | Endpoint | Method | Description |
|---|---|---|---|
| Models | `/base-models` | GET | List available base models |
| Jobs | `/jobs` | POST | Create job |
| Jobs | `/jobs/{id}/start_training_remote` | POST | Start training (local or remote) |
| Jobs | `/jobs/{id}/status` | GET | Status + loss curves |
| Jobs | `/jobs/{id}/logs` | GET | Training logs |
| Jobs | `/jobs/{id}` | DELETE | Delete job and all its data |
| Samples | `/sample-jobs` | POST | Seed two sample jobs with training data |
| Data | `/jobs/{id}/upload_data` | POST | Upload CSV or JSON |
| Data | `/jobs/{id}/data_preview` | GET | First N rows |
| Data | `/jobs/{id}/data_stats` | GET | Row count, vocab size, split sizes |
| Evaluation | `/jobs/{id}/metrics` | GET | Metrics for all versions |
| Evaluation | `/jobs/{id}/model_comparison` | GET | Rank versions, recommendation |
| Evaluation | `/models/{version_id}/confusion_matrix` | GET | Classification matrix |
| Inference | `/jobs/{id}/predict` | POST | Single prediction |
| Inference | `/jobs/{id}/predict_batch` | POST | Batch predictions |
| Inference | `/jobs/{id}/models` | GET | List versions + cache status |
| Download | `/jobs/{id}/models/{version}/download` | GET | Download model version as zip |
| Compute | `/compute` | POST | Register SSH instance |
| Compute | `/compute` | GET | List instances |
| Compute | `/compute/{id}/test` | POST | Test SSH connection |
| Compute | `/compute/{id}` | DELETE | Remove instance |
| Experiments | `/experiments` | GET | List MLflow experiments |
| Experiments | `/experiments/{id}` | GET | Runs in an experiment |
| Experiments | `/experiments/compare` | POST | Diff runs side-by-side |
| Monitoring | `/jobs/{id}/monitoring` | GET | Latency stats + drift alerts |
| Monitoring | `/metrics` | GET | Prometheus scrape endpoint |

---

## Running Tests

```bash
venv/bin/pytest tests/ -q
# 141 passed

venv/bin/pytest tests/test_data_loading.py   # Data pipeline (27)
venv/bin/pytest tests/test_training.py        # Training loop (13)
venv/bin/pytest tests/test_evaluation.py      # Metrics & comparison (27)
venv/bin/pytest tests/test_inference.py       # Inference & caching (18)
venv/bin/pytest tests/test_mlflow.py          # Experiment tracking (16)
venv/bin/pytest tests/test_edge_cases.py      # Edge cases (31)
venv/bin/pytest tests/test_integration.py     # End-to-end pipeline (9)
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

## Using a Downloaded Model

After training, open the job's Overview tab and click **Download** next to any model version. You get a zip containing a standard HuggingFace checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./llm-studio-job1-v1/")
model = AutoModelForCausalLM.from_pretrained("./llm-studio-job1-v1/")

inputs = tokenizer("Translate to French: Hello", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Or push directly to the HuggingFace Hub:

```python
model.push_to_hub("your-org/your-fine-tuned-model")
tokenizer.push_to_hub("your-org/your-fine-tuned-model")
```

The zip contains: model weights, tokenizer files, architecture config, and a `training_args.json` snapshot of the hyperparameters and best validation loss.

---

## AWS Deployment

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
./deploy.sh        # CPU
./deploy.sh --gpu  # GPU (uses Dockerfile.gpu)
```

The script runs the full test suite, builds and pushes to ECR, registers a new ECS task definition, and waits for the service to stabilise.

---

## Documentation

| File | Contents |
|---|---|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Data prep checklist, hyperparameter tuning, loss curve patterns, early stopping |
| [API_DOCS.md](API_DOCS.md) | All endpoints with full request/response JSON and error codes |

# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

All request bodies are JSON. All responses are JSON unless noted.

---

## Error Codes

| Code | Meaning |
|---|---|
| `400` | Bad request — invalid file type or unknown model name |
| `404` | Resource not found — job, version, or experiment does not exist |
| `409` | Conflict — job already training or already completed |
| `422` | Validation error — missing fields, blank data, too few samples |
| `503` | MLflow unavailable |

---

## Jobs

### `POST /jobs`
Create a new fine-tuning job.

**Request:**
```json
{ "user_id": 1, "model_name": "gpt2" }
```

**Response `201`:**
```json
{ "job_id": 1, "status": "queued", "model_name": "gpt2" }
```

**Errors:** `400` if `model_name` not in supported list. `404` if user not found.

---

### `POST /jobs/{job_id}/start_training`
Start training in the background. Returns immediately.

**Response `200`:**
```json
{ "job_id": 1, "message": "Training started" }
```

**Errors:** `409` if already training or completed. `422` if no training data uploaded.

---

### `GET /jobs/{job_id}/status`
Current status and loss curves.

**Response:**
```json
{
  "job_id": 1,
  "model_name": "gpt2",
  "status": "completed",
  "created_at": "2025-01-15T10:00:00+00:00",
  "updated_at": "2025-01-15T10:45:00+00:00",
  "model_versions": [
    { "version": 1, "loss": 1.82, "accuracy": null, "path": "./storage/models/job_1/v1" }
  ],
  "loss_curves": {
    "epochs": [1, 2, 3],
    "train_loss": [3.1, 2.4, 1.9],
    "val_loss": [3.3, 2.6, 2.1]
  }
}
```

---

### `GET /jobs/{job_id}/logs`
Training log entries.

**Response:**
```json
{
  "job_id": 1,
  "logs": [
    { "timestamp": "2025-01-15T10:01:00+00:00", "level": "info", "message": "Training started on device: mps" },
    { "timestamp": "2025-01-15T10:10:00+00:00", "level": "info", "message": "Epoch 1 complete — train_loss=3.10, val_loss=3.30" }
  ]
}
```

---

## Data

### `POST /jobs/{job_id}/upload_data`
Upload training data as a CSV or JSON file.

**Request:** `multipart/form-data` with a `file` field.

**CSV format:** columns `input_text` and `expected_output`.

**JSON format:** array of `{ "input": "...", "output": "..." }` objects.

**Response:**
```json
{ "job_id": 1, "rows_added": 500 }
```

**Errors:** `400` for unsupported file type. `422` for format or validation errors.

---

### `GET /jobs/{job_id}/data_preview?rows=5`
First N rows of training data.

**Response:**
```json
{
  "job_id": 1,
  "total_rows": 500,
  "preview": [
    { "input": "Translate: Hello", "output": "Bonjour" }
  ]
}
```

---

### `GET /jobs/{job_id}/data_stats`
Dataset statistics.

**Response:**
```json
{
  "job_id": 1,
  "num_rows": 500,
  "avg_input_length": 42.3,
  "avg_output_length": 18.7,
  "vocab_size": 3812,
  "split": { "train": 400, "val": 50, "test": 50 }
}
```

---

## Evaluation

### `GET /jobs/{job_id}/metrics`
Evaluation metrics for all model versions.

**Response:**
```json
{
  "job_id": 1,
  "versions": [
    {
      "version_num": 1,
      "training_loss": 1.82,
      "evaluation": {
        "task_type": "generation",
        "metrics": { "bleu": 28.4, "perplexity": 42.1, "avg_output_length": 14.2 },
        "interpretation": {
          "bleu": "28.4/100 — Good fluency",
          "perplexity": "42.1 — Good"
        }
      }
    }
  ]
}
```

---

### `GET /jobs/{job_id}/model_comparison`
Compare all evaluated versions. Requires at least 2 evaluated versions.

**Response:**
```json
{
  "job_id": 1,
  "best_version": 2,
  "primary_metric": "bleu",
  "recommendation": "Deploy v2 — meaningfully better on bleu (31.0 vs 22.5).",
  "ranking": [
    { "rank": 1, "version_num": 2, "metrics": { "bleu": 31.0, "perplexity": 30.0 } },
    { "rank": 2, "version_num": 1, "metrics": { "bleu": 22.5, "perplexity": 45.0 } }
  ],
  "significance": [
    { "versions": "v1 vs v2", "metric": "bleu", "significant": true, "better_version": 2, "p_value": null }
  ]
}
```

---

### `GET /models/{version_id}/confusion_matrix`
Confusion matrix for classification tasks only.

**Response:**
```json
{
  "version_id": 1,
  "job_id": 1,
  "version_num": 1,
  "labels": ["negative", "neutral", "positive"],
  "matrix": [[45, 3, 2], [4, 38, 8], [1, 5, 44]]
}
```

---

## Inference

### `POST /jobs/{job_id}/predict`
Single prediction.

**Request:**
```json
{
  "input": "Translate to French: The sky is blue",
  "version": null,
  "max_new_tokens": 128
}
```
`version`: integer version number, or `null` to use the best available model.

**Response:**
```json
{
  "job_id": 1,
  "version_num": 2,
  "input": "Translate to French: The sky is blue",
  "output": "Le ciel est bleu",
  "confidence": 0.8732,
  "input_token_count": 9,
  "output_token_count": 5,
  "latency_ms": 143.2
}
```

---

### `POST /jobs/{job_id}/predict_batch`
Multiple predictions in one request (faster than calling predict N times).

**Request:**
```json
{
  "inputs": ["Hello", "Good morning", "How are you?"],
  "version": null,
  "max_new_tokens": 64
}
```

**Response:**
```json
{
  "job_id": 1,
  "version_num": 2,
  "total_latency_ms": 380.4,
  "avg_latency_ms": 126.8,
  "predictions": [
    { "input": "Hello", "output": "Bonjour", "confidence": 0.91, "input_token_count": 1, "output_token_count": 1 },
    { "input": "Good morning", "output": "Bonjour matin", "confidence": 0.85, "input_token_count": 2, "output_token_count": 2 }
  ]
}
```

---

### `GET /jobs/{job_id}/models`
List all trained model versions for a job.

**Response:**
```json
{
  "job_id": 1,
  "versions": [
    {
      "version_num": 2,
      "model_path": "./storage/models/job_1/v2",
      "loss": 1.72,
      "accuracy": null,
      "created_at": "2025-01-15T11:00:00+00:00",
      "cached": true
    }
  ]
}
```

---

## Experiments (MLflow)

### `GET /experiments`
List all MLflow experiments.

**Response:**
```json
{
  "experiments": [
    { "experiment_id": "1", "name": "llm-studio", "lifecycle_stage": "active" }
  ]
}
```

---

### `GET /experiments/{experiment_id}`
All runs within an experiment.

**Response:**
```json
{
  "experiment_id": "1",
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "job_1",
      "job_id": "1",
      "status": "FINISHED",
      "params": { "learning_rate": "2e-05", "epochs": "3" },
      "metrics": { "val_loss": 1.82, "final.bleu": 28.4 }
    }
  ]
}
```

---

### `POST /experiments/compare`
Compare hyperparameters and metrics across runs.

**Request:**
```json
{ "run_ids": ["abc123", "def456"] }
```

**Response:**
```json
{
  "runs": [
    { "run_id": "abc123", "params": { "learning_rate": "2e-05" }, "metrics": { "val_loss": 1.82 } },
    { "run_id": "def456", "params": { "learning_rate": "5e-05" }, "metrics": { "val_loss": 1.61 } }
  ]
}
```

---

## Monitoring

### `GET /jobs/{job_id}/monitoring`
Latency statistics and drift alerts.

**Response:**
```json
{
  "job_id": 1,
  "latency_stats": {
    "count": 1024,
    "mean_ms": 148.3,
    "p50_ms": 132.0,
    "p95_ms": 310.5,
    "p99_ms": 480.2,
    "max_ms": 1204.0
  },
  "drift_alerts": [],
  "baseline_ready": true
}
```

---

### `GET /metrics`
Prometheus metrics in text exposition format. Scrape target for Prometheus.

**Response:** `text/plain; version=0.0.4`
```
llm_inference_requests_total{job_id="1",status="success"} 1024
llm_inference_latency_ms_bucket{job_id="1",le="100"} 612
llm_prediction_confidence_bucket{job_id="1",le="0.8"} 890
...
```

"""
REST API for LLM Studio: job management, data upload, training control, evaluation, inference.
"""
from __future__ import annotations

import io
import json
import os
from typing import Annotated, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from llm_studio.comparator import compare_versions
from llm_studio.config import BASE_MODELS, DATABASE_URL, DEFAULT_STORAGE_CONFIG, MLFLOW_TRACKING_URI
from llm_studio.data_loader import (
    clean_data,
    load_from_csv,
    load_from_json,
    load_training_data,
    split_data,
    validate_training_data,
)
from llm_studio.evaluator import load_eval_result
from llm_studio.inference import predict, predict_batch
from llm_studio.metrics import confusion_matrix_data
from llm_studio.mlflow_integration import compare_runs, get_experiment_runs, list_experiments
from llm_studio.model_loader import list_versions, load_model
from llm_studio.monitoring import JobMonitor, prometheus_metrics
from llm_studio.models import FineTuningJob, JobStatus, ModelVersion, TrainingData, User, create_all, get_engine
from llm_studio.preprocessor import normalize, tokenize

app = FastAPI(title="LLM Studio", version="0.3.0")

engine = get_engine(DATABASE_URL)
create_all(engine)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_session():
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    user_id: int
    model_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_job_or_404(job_id: int, session: Session) -> FineTuningJob:
    job = session.get(FineTuningJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def _persist_pairs(job_id: int, pairs: list[tuple[str, str]], session: Session) -> int:
    rows = [
        TrainingData(job_id=job_id, input=inp, expected_output=out)
        for inp, out in pairs
    ]
    session.add_all(rows)
    session.commit()
    return len(rows)


def _run_training(job_id: int) -> None:
    """Background task: creates its own DB session and runs the trainer."""
    from sqlalchemy.orm import sessionmaker
    from llm_studio.trainer import train_model

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        train_model(job_id, session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

@app.post("/jobs", summary="Create a new fine-tuning job", status_code=201)
def create_job(body: CreateJobRequest, session: Session = Depends(get_session)):
    if body.model_name not in BASE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model_name}'. Available: {list(BASE_MODELS)}",
        )
    user = session.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found")

    job = FineTuningJob(user_id=body.user_id, model_name=body.model_name)
    session.add(job)
    session.commit()
    session.refresh(job)
    return {"job_id": job.id, "status": job.status.value, "model_name": job.model_name}


@app.post("/jobs/{job_id}/start_training", summary="Start training for a job")
def start_training(
    job_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    job = _get_job_or_404(job_id, session)

    if job.status == JobStatus.training:
        raise HTTPException(status_code=409, detail="Job is already training")
    if job.status == JobStatus.completed:
        raise HTTPException(status_code=409, detail="Job already completed. Create a new job to retrain.")

    if not session.query(TrainingData).filter_by(job_id=job_id).first():
        raise HTTPException(status_code=422, detail="No training data uploaded for this job")

    background_tasks.add_task(_run_training, job_id)
    return {"job_id": job_id, "message": "Training started"}


@app.get("/jobs/{job_id}/status", summary="Training status and loss curves")
def job_status(job_id: int, session: Session = Depends(get_session)):
    job = _get_job_or_404(job_id, session)

    log_path = os.path.join(
        DEFAULT_STORAGE_CONFIG.base_dir, "logs", f"job_{job_id}", "training.json"
    )
    loss_curves = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            data = json.load(f)
            loss_curves = data.get("metrics", {})

    versions = [
        {"version": v.version_num, "loss": v.loss, "accuracy": v.accuracy, "path": v.model_path}
        for v in job.model_versions
    ]

    return {
        "job_id": job_id,
        "model_name": job.model_name,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "model_versions": versions,
        "loss_curves": loss_curves,
    }


@app.get("/jobs/{job_id}/logs", summary="Training logs")
def job_logs(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)

    log_path = os.path.join(
        DEFAULT_STORAGE_CONFIG.base_dir, "logs", f"job_{job_id}", "training.json"
    )
    if not os.path.exists(log_path):
        return {"job_id": job_id, "logs": []}

    with open(log_path) as f:
        data = json.load(f)
    return {"job_id": job_id, "logs": data.get("logs", [])}


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload_data", summary="Upload training data (CSV or JSON)")
async def upload_data(
    job_id: int,
    file: Annotated[UploadFile, File(description="CSV or JSON file")],
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)

    content = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            pairs = load_from_csv(io.StringIO(content.decode("utf-8")))
        elif filename.endswith(".json"):
            pairs = load_from_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .json files are supported")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    pairs = clean_data(pairs)
    errors = validate_training_data(pairs)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    count = _persist_pairs(job_id, pairs, session)
    return {"job_id": job_id, "rows_added": count}


@app.get("/jobs/{job_id}/data_preview", summary="Preview first rows of training data")
def data_preview(
    job_id: int,
    rows: int = 5,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    pairs = load_training_data(job_id, session)
    preview = [{"input": inp, "output": out} for inp, out in pairs[:rows]]
    return {"job_id": job_id, "total_rows": len(pairs), "preview": preview}


@app.get("/jobs/{job_id}/data_stats", summary="Statistics for a job's training data")
def data_stats(
    job_id: int,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    pairs = load_training_data(job_id, session)

    if not pairs:
        return {"job_id": job_id, "num_rows": 0}

    inputs = [inp for inp, _ in pairs]
    outputs = [out for _, out in pairs]
    all_tokens = [tok for text in inputs + outputs for tok in tokenize(normalize(text))]
    split = split_data(pairs)

    return {
        "job_id": job_id,
        "num_rows": len(pairs),
        "avg_input_length": round(sum(len(t) for t in inputs) / len(inputs), 1),
        "avg_output_length": round(sum(len(t) for t in outputs) / len(outputs), 1),
        "vocab_size": len(set(all_tokens)),
        "split": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/metrics", summary="Evaluation metrics for all versions of a job")
def job_metrics(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)

    versions = session.query(ModelVersion).filter_by(job_id=job_id).all()
    if not versions:
        raise HTTPException(status_code=404, detail="No model versions found. Run training first.")

    results = []
    for v in versions:
        eval_data = load_eval_result(job_id, v.version_num)
        results.append({
            "version_num": v.version_num,
            "model_path": v.model_path,
            "training_loss": v.loss,
            "training_accuracy": v.accuracy,
            "evaluation": eval_data or "not evaluated yet — POST /jobs/{id}/evaluate to run",
        })

    return {"job_id": job_id, "versions": results}


@app.get("/jobs/{job_id}/model_comparison", summary="Compare all evaluated versions for a job")
def model_comparison(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)

    versions = session.query(ModelVersion).filter_by(job_id=job_id).all()
    eval_results = [
        load_eval_result(job_id, v.version_num)
        for v in versions
        if load_eval_result(job_id, v.version_num) is not None
    ]

    if len(eval_results) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least 2 evaluated versions required for comparison.",
        )

    comparison = compare_versions(job_id, eval_results)
    return {
        "job_id": job_id,
        "best_version": comparison.best_version,
        "primary_metric": comparison.primary_metric,
        "recommendation": comparison.recommendation,
        "ranking": comparison.ranking,
        "significance": [
            {
                "versions": f"v{s.version_a} vs v{s.version_b}",
                "metric": s.metric,
                "significant": s.significant,
                "better_version": s.better_version,
                "p_value": s.p_value,
            }
            for s in comparison.significance
        ],
    }


@app.get("/models/{version_id}/confusion_matrix", summary="Confusion matrix data for a model version")
def get_confusion_matrix(version_id: int, session: Session = Depends(get_session)):
    version = session.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"ModelVersion {version_id} not found")

    eval_data = load_eval_result(version.job_id, version.version_num)
    if not eval_data:
        raise HTTPException(status_code=404, detail="No evaluation results found for this version.")

    if eval_data.get("task_type") != "classification":
        raise HTTPException(status_code=422, detail="Confusion matrix is only available for classification tasks.")

    per_class = eval_data["metrics"].get("per_class", {})
    labels = list(per_class.keys())

    # Reconstruct predictions from sample data if full matrix not stored
    sample_preds = eval_data.get("sample_predictions", [])
    if sample_preds:
        y_true = [p["expected"] for p in sample_preds]
        y_pred = [p["predicted"] for p in sample_preds]
        cm_data = confusion_matrix_data(y_true, y_pred, labels)
    else:
        cm_data = {"labels": labels, "matrix": []}

    return {
        "version_id": version_id,
        "job_id": version.job_id,
        "version_num": version.version_num,
        **cm_data,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    input: str
    version: Optional[int] = None   # None → best available version
    max_new_tokens: int = 128


class PredictBatchRequest(BaseModel):
    inputs: list[str]
    version: Optional[int] = None
    max_new_tokens: int = 128


@app.post("/jobs/{job_id}/predict", summary="Single prediction from a trained model")
def predict_single(
    job_id: int,
    body: PredictRequest,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    try:
        entry = load_model(job_id, session, version_num=body.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result = predict(entry.model, entry.tokenizer, body.input, max_new_tokens=body.max_new_tokens)
    return {
        "job_id": job_id,
        "version_num": entry.version_num,
        "input": body.input,
        "output": result.output,
        "confidence": result.confidence,
        "input_token_count": result.input_token_count,
        "output_token_count": result.output_token_count,
        "latency_ms": result.latency_ms,
    }


@app.post("/jobs/{job_id}/predict_batch", summary="Batch predictions from a trained model")
def predict_batch_endpoint(
    job_id: int,
    body: PredictBatchRequest,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    if not body.inputs:
        raise HTTPException(status_code=422, detail="inputs list cannot be empty")

    try:
        entry = load_model(job_id, session, version_num=body.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    batch_result = predict_batch(
        entry.model, entry.tokenizer, body.inputs, max_new_tokens=body.max_new_tokens
    )
    return {
        "job_id": job_id,
        "version_num": entry.version_num,
        "total_latency_ms": batch_result.total_latency_ms,
        "avg_latency_ms": batch_result.avg_latency_ms,
        "predictions": [
            {
                "input": inp,
                "output": r.output,
                "confidence": r.confidence,
                "input_token_count": r.input_token_count,
                "output_token_count": r.output_token_count,
            }
            for inp, r in zip(body.inputs, batch_result.results)
        ],
    }


@app.get("/jobs/{job_id}/models", summary="List available model versions for a job")
def get_models(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)
    versions = list_versions(job_id, session)
    if not versions:
        raise HTTPException(status_code=404, detail="No trained model versions found for this job.")
    return {"job_id": job_id, "versions": versions}


# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------

@app.get("/experiments", summary="List all MLflow experiments")
def get_experiments():
    try:
        experiments = list_experiments(MLFLOW_TRACKING_URI)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")
    return {"experiments": experiments}


@app.get("/experiments/{experiment_id}", summary="Runs within an experiment")
def get_experiment(experiment_id: str):
    try:
        runs = get_experiment_runs(experiment_id, MLFLOW_TRACKING_URI)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"experiment_id": experiment_id, "runs": runs}


class CompareRunsRequest(BaseModel):
    run_ids: list[str]


@app.post("/experiments/compare", summary="Compare hyperparameters and metrics across runs")
def compare_experiment_runs(body: CompareRunsRequest):
    if len(body.run_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 run_ids to compare.")
    try:
        result = compare_runs(body.run_ids, MLFLOW_TRACKING_URI)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

_monitors: dict[int, JobMonitor] = {}


def _get_monitor(job_id: int) -> JobMonitor:
    if job_id not in _monitors:
        _monitors[job_id] = JobMonitor(job_id)
    return _monitors[job_id]


@app.get("/metrics", summary="Prometheus metrics endpoint", include_in_schema=False)
def metrics():
    data, content_type = prometheus_metrics()
    return Response(content=data, media_type=content_type)


@app.get("/jobs/{job_id}/monitoring", summary="Latency stats and drift alerts for a job")
def job_monitoring(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)
    return _get_monitor(job_id).summary()

"""
REST API for LLM Studio: job management, data upload, training control, evaluation, inference.
"""
from __future__ import annotations

import io
import json
import os
from typing import Annotated, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
from llm_studio.models import ComputeInstance, ComputeStatus, FineTuningJob, JobStatus, ModelVersion, Prediction, TrainingData, User, create_all, get_engine
from llm_studio.remote_runner import test_connection as ssh_test
from llm_studio.preprocessor import normalize, tokenize

app = FastAPI(title="LLM Studio", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Base models
# ---------------------------------------------------------------------------

@app.get("/base-models", summary="List available base models for fine-tuning")
def list_base_models():
    return {
        "models": [
            {
                "id": model_id,
                "hf_id": meta["hf_id"],
                "max_tokens": meta["max_tokens"],
                "family": meta["family"],
            }
            for model_id, meta in BASE_MODELS.items()
        ]
    }


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


@app.delete("/jobs/{job_id}", summary="Delete a job and all its data", status_code=204)
def delete_job(job_id: int, session: Session = Depends(get_session)):
    _get_job_or_404(job_id, session)
    session.query(TrainingData).filter_by(job_id=job_id).delete()
    session.query(Prediction).filter_by(job_id=job_id).delete()
    session.query(ModelVersion).filter_by(job_id=job_id).delete()
    session.query(FineTuningJob).filter_by(id=job_id).delete()
    session.commit()


_SAMPLE_DATA = {
    "gpt2": {
        "description": "English to French translation",
        "pairs": [
            ("Translate to French: Hello", "Bonjour"),
            ("Translate to French: Goodbye", "Au revoir"),
            ("Translate to French: Thank you", "Merci"),
            ("Translate to French: Please", "S'il vous plaît"),
            ("Translate to French: Yes", "Oui"),
            ("Translate to French: No", "Non"),
            ("Translate to French: Good morning", "Bonjour"),
            ("Translate to French: Good night", "Bonne nuit"),
            ("Translate to French: How are you?", "Comment allez-vous?"),
            ("Translate to French: My name is", "Je m'appelle"),
            ("Translate to French: I love you", "Je t'aime"),
            ("Translate to French: Where is the station?", "Où est la gare?"),
        ],
    },
    "t5-small": {
        "description": "Text summarisation",
        "pairs": [
            ("Summarize: The quick brown fox jumps over the lazy dog near the river bank.", "Fox jumps over dog."),
            ("Summarize: Artificial intelligence is transforming industries across the globe at an unprecedented pace.", "AI is rapidly transforming industries."),
            ("Summarize: The stock market experienced significant volatility today due to inflation concerns.", "Markets volatile on inflation fears."),
            ("Summarize: Scientists have discovered a new species of deep-sea fish in the Pacific Ocean.", "New deep-sea fish species found."),
            ("Summarize: The government announced a new policy to reduce carbon emissions by 50% by 2030.", "Government targets 50% emission cut by 2030."),
            ("Summarize: Researchers at MIT developed a new battery that charges in under five minutes.", "MIT creates fast-charging battery."),
            ("Summarize: The annual rainfall this year has been the highest recorded in the past century.", "Record rainfall this year."),
            ("Summarize: A local startup raised 10 million dollars in its latest funding round.", "Startup raises $10M in funding."),
            ("Summarize: The new smartphone model features a camera with 200 megapixels resolution.", "New phone has 200MP camera."),
            ("Summarize: Health experts recommend at least 30 minutes of exercise every day for adults.", "Adults should exercise 30 min daily."),
            ("Summarize: The ancient ruins were discovered by a team of archaeologists in southern Italy.", "Ruins found in southern Italy."),
            ("Summarize: Electric vehicles now account for 15 percent of all new car sales globally.", "EVs are 15% of global new car sales."),
        ],
    },
}


@app.post("/sample-jobs", summary="Create two sample fine-tuning jobs with pre-loaded training data", status_code=201)
def create_sample_jobs(session: Session = Depends(get_session)):
    # Ensure a sample user exists
    user = session.query(User).filter_by(email="sample@llm-studio.dev").first()
    if not user:
        user = User(email="sample@llm-studio.dev")
        session.add(user)
        session.flush()

    created = []
    for model_id, sample in _SAMPLE_DATA.items():
        job = FineTuningJob(user_id=user.id, model_name=model_id)
        session.add(job)
        session.flush()
        for inp, out in sample["pairs"]:
            session.add(TrainingData(job_id=job.id, input=inp, expected_output=out))
        session.commit()
        created.append({
            "job_id": job.id,
            "model_name": model_id,
            "description": sample["description"],
            "rows": len(sample["pairs"]),
        })

    return {"jobs": created}


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


# ---------------------------------------------------------------------------
# Compute instances
# ---------------------------------------------------------------------------

class AddComputeRequest(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    key_path: Optional[str] = None


def _instance_to_dict(inst: ComputeInstance) -> dict:
    return {
        "id": inst.id,
        "name": inst.name,
        "host": inst.host,
        "port": inst.port,
        "username": inst.username,
        "key_path": inst.key_path,
        "last_status": inst.last_status.value,
        "last_checked": inst.last_checked.isoformat() if inst.last_checked else None,
        "created_at": inst.created_at.isoformat(),
    }


@app.post("/compute", summary="Register a compute instance", status_code=201)
def add_compute(body: AddComputeRequest, session: Session = Depends(get_session)):
    inst = ComputeInstance(
        name=body.name,
        host=body.host,
        port=body.port,
        username=body.username,
        key_path=body.key_path,
    )
    session.add(inst)
    session.commit()
    session.refresh(inst)
    return _instance_to_dict(inst)


@app.get("/compute", summary="List compute instances")
def list_compute(session: Session = Depends(get_session)):
    instances = session.query(ComputeInstance).order_by(ComputeInstance.created_at).all()
    return {"instances": [_instance_to_dict(i) for i in instances]}


@app.get("/compute/{instance_id}", summary="Get a compute instance")
def get_compute(instance_id: int, session: Session = Depends(get_session)):
    inst = session.get(ComputeInstance, instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Compute instance {instance_id} not found")
    return _instance_to_dict(inst)


@app.post("/compute/{instance_id}/test", summary="Test SSH connection to a compute instance")
def test_compute(instance_id: int, session: Session = Depends(get_session)):
    from datetime import datetime, timezone
    inst = session.get(ComputeInstance, instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Compute instance {instance_id} not found")

    success, message = ssh_test(inst.host, inst.port, inst.username, inst.key_path)
    inst.last_status = ComputeStatus.connected if success else ComputeStatus.error
    inst.last_checked = datetime.now(timezone.utc)
    session.commit()

    return {
        "instance_id": instance_id,
        "success": success,
        "message": message,
        "status": inst.last_status.value,
    }


@app.delete("/compute/{instance_id}", summary="Remove a compute instance", status_code=204)
def delete_compute(instance_id: int, session: Session = Depends(get_session)):
    inst = session.get(ComputeInstance, instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Compute instance {instance_id} not found")
    session.delete(inst)
    session.commit()


# ---------------------------------------------------------------------------
# Remote training trigger
# ---------------------------------------------------------------------------

class StartTrainingRequest(BaseModel):
    compute_id: Optional[int] = None   # None = run locally


@app.post("/jobs/{job_id}/start_training_remote", summary="Start training on a specific compute instance")
def start_training_remote(
    job_id: int,
    body: StartTrainingRequest,
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

    if body.compute_id is None:
        # Fall back to local training
        background_tasks.add_task(_run_training, job_id)
        return {"job_id": job_id, "compute": "local", "message": "Training started locally"}

    inst = session.get(ComputeInstance, body.compute_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Compute instance {body.compute_id} not found")

    background_tasks.add_task(_run_remote_training, job_id, body.compute_id)
    return {
        "job_id": job_id,
        "compute": "remote",
        "instance": inst.name,
        "host": inst.host,
        "message": f"Training started on {inst.name} ({inst.host})",
    }


def _run_remote_training(job_id: int, compute_id: int) -> None:
    """Background task: trains on a remote instance via SSH."""
    from datetime import datetime, timezone
    from sqlalchemy.orm import sessionmaker
    from llm_studio.config import DEFAULT_TRAINING_CONFIG, DEFAULT_STORAGE_CONFIG
    from llm_studio.data_loader import load_training_data
    from llm_studio.models import ModelVersion
    from llm_studio.remote_runner import RemoteRunner

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        job = session.get(FineTuningJob, job_id)
        inst = session.get(ComputeInstance, compute_id)
        job.status = JobStatus.training
        session.commit()

        data = load_training_data(job_id, session)
        training_data = [{"input": inp, "output": out} for inp, out in data]

        config = DEFAULT_TRAINING_CONFIG
        config_dict = {**config.__dict__, "model_name": job.model_name}

        local_output = DEFAULT_STORAGE_CONFIG.model_path(job_id, 1)
        log_entries = []

        def log_cb(msg: str):
            log_entries.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                                 "level": "info", "message": msg})

        runner = RemoteRunner(
            host=inst.host,
            port=inst.port,
            username=inst.username,
            key_path=inst.key_path,
            log_callback=log_cb,
        )
        runner.run(job_id, training_data, config_dict, local_output)

        # Record model version
        session.add(ModelVersion(
            job_id=job_id, version_num=1,
            model_path=local_output, loss=None,
        ))
        job.status = JobStatus.completed
        session.commit()

        # Persist logs
        import json as _json, os as _os
        log_dir = _os.path.join(DEFAULT_STORAGE_CONFIG.base_dir, "logs", f"job_{job_id}")
        _os.makedirs(log_dir, exist_ok=True)
        with open(_os.path.join(log_dir, "training.json"), "w") as f:
            _json.dump({"job_id": job_id, "logs": log_entries, "metrics": {}}, f, indent=2)

    except Exception as exc:
        job.status = JobStatus.failed
        session.commit()
        raise
    finally:
        session.close()

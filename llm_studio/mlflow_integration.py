"""
MLflow experiment tracking: logging params, metrics, artifacts, and metadata.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from llm_studio.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, TrainingConfig


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Thin wrapper around MLflow that ties a training run to a job.
    Supports use as a context manager for automatic run cleanup.

    Usage:
        with ExperimentTracker(job_id=5) as tracker:
            tracker.log_params(config)
            for epoch in ...:
                tracker.log_epoch(epoch, train_loss, val_loss, lr)
            tracker.log_final_metrics({"bleu": 28.4})
            tracker.log_model_artifact("/storage/models/job_5/v1")
    """

    def __init__(
        self,
        job_id: int,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ) -> None:
        self.job_id = job_id
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_id: Optional[str] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, *_) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, tags: Optional[dict] = None) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        run_tags = {"job_id": str(self.job_id), **(tags or {})}
        run = mlflow.start_run(run_name=f"job_{self.job_id}", tags=run_tags)
        self._run_id = run.info.run_id
        self._start_time = time.time()
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        if self._run_id and self._start_time:
            elapsed = round(time.time() - self._start_time, 2)
            mlflow.log_metric("training_time_seconds", elapsed)
        mlflow.end_run(status=status)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, config: TrainingConfig) -> None:
        """Log all training hyperparameters before the run starts."""
        params = {k: str(v) for k, v in asdict(config).items()}
        mlflow.log_params(params)

    def log_dataset_info(self, train_size: int, val_size: int, test_size: int) -> None:
        mlflow.log_params({
            "dataset.train_size": train_size,
            "dataset.val_size": val_size,
            "dataset.test_size": test_size,
            "dataset.total_size": train_size + val_size + test_size,
        })

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_perplexity": _safe_exp(train_loss),
                "val_perplexity": _safe_exp(val_loss),
                "learning_rate": lr,
            },
            step=epoch,
        )

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics at the end of training."""
        flat = {f"final.{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if flat:
            mlflow.log_metrics(flat)

    def log_model_artifact(self, model_path: str, artifact_subdir: str = "model") -> None:
        """Log the trained model directory as an MLflow artifact."""
        if os.path.isdir(model_path):
            mlflow.log_artifacts(model_path, artifact_path=artifact_subdir)
        else:
            mlflow.log_artifact(model_path, artifact_path=artifact_subdir)

    def log_metadata(self, **kwargs: Any) -> None:
        """Log arbitrary key-value metadata as params."""
        mlflow.log_params({f"meta.{k}": str(v) for k, v in kwargs.items()})

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id


# ---------------------------------------------------------------------------
# Query helpers (used by API endpoints)
# ---------------------------------------------------------------------------

def get_client(tracking_uri: str = MLFLOW_TRACKING_URI) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def list_experiments(tracking_uri: str = MLFLOW_TRACKING_URI) -> list[dict]:
    client = get_client(tracking_uri)
    experiments = client.search_experiments()
    return [
        {
            "experiment_id": e.experiment_id,
            "name": e.name,
            "artifact_location": e.artifact_location,
            "lifecycle_stage": e.lifecycle_stage,
            "creation_time": e.creation_time,
        }
        for e in experiments
    ]


def get_experiment_runs(
    experiment_id: str,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> list[dict]:
    client = get_client(tracking_uri)
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
    return [_run_to_dict(r) for r in runs]


def get_run(run_id: str, tracking_uri: str = MLFLOW_TRACKING_URI) -> dict:
    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    return _run_to_dict(run)


def compare_runs(run_ids: list[str], tracking_uri: str = MLFLOW_TRACKING_URI) -> dict:
    """Return params and final metrics side-by-side for a list of run IDs."""
    client = get_client(tracking_uri)
    rows = []
    for rid in run_ids:
        run = client.get_run(rid)
        rows.append({
            "run_id": rid,
            "run_name": run.data.tags.get("mlflow.runName", rid[:8]),
            "job_id": run.data.tags.get("job_id"),
            "status": run.info.status,
            "params": run.data.params,
            "metrics": run.data.metrics,
        })
    return {"runs": rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_exp(loss: float, max_loss: float = 20.0) -> float:
    import math
    return round(math.exp(min(loss, max_loss)), 4)


def _run_to_dict(run) -> dict:
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        "job_id": run.data.tags.get("job_id"),
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "artifact_uri": run.info.artifact_uri,
    }

"""
Tests for MLflow experiment tracking: logging params, metrics, and artifacts.
All tests use a temporary local tracking URI — no MLflow server required.
"""
from __future__ import annotations

import os
import tempfile

import mlflow
import pytest

from llm_studio.config import TrainingConfig
from llm_studio.mlflow_integration import (
    ExperimentTracker,
    compare_runs,
    get_client,
    get_experiment_runs,
    list_experiments,
)


# ---------------------------------------------------------------------------
# Fixture: isolated tracking directory per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("sqlite:///tmp_mlflow_default.db")


@pytest.fixture()
def tracker(tracking_uri):
    return ExperimentTracker(job_id=1, experiment_name="test-exp", tracking_uri=tracking_uri)


# ---------------------------------------------------------------------------
# test_experiment_logging
# ---------------------------------------------------------------------------

def test_start_run_returns_run_id(tracker):
    run_id = tracker.start_run()
    assert run_id is not None and len(run_id) > 0
    mlflow.end_run()


def test_run_is_tagged_with_job_id(tracker, tracking_uri):
    with tracker:
        pass
    client = get_client(tracking_uri)
    runs = client.search_runs(experiment_ids=["1"])
    if not runs:
        runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name("test-exp").experiment_id]
        )
    assert any(r.data.tags.get("job_id") == "1" for r in runs)


def test_context_manager_ends_run(tracker, tracking_uri):
    with tracker:
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"


def test_failed_run_marked_failed(tracking_uri):
    tracker = ExperimentTracker(job_id=99, experiment_name="test-exp", tracking_uri=tracking_uri)
    run_id = None
    try:
        with tracker:
            run_id = tracker.run_id
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FAILED"


def test_list_experiments_returns_list(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("list-test-exp")
    with mlflow.start_run():
        pass
    experiments = list_experiments(tracking_uri)
    assert isinstance(experiments, list)
    names = [e["name"] for e in experiments]
    assert "list-test-exp" in names


# ---------------------------------------------------------------------------
# test_metric_logging
# ---------------------------------------------------------------------------

def test_log_params_records_hyperparameters(tracker, tracking_uri):
    config = TrainingConfig(learning_rate=3e-5, epochs=5, batch_size=16)
    with tracker:
        tracker.log_params(config)
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["learning_rate"] == "3e-05"
    assert run.data.params["epochs"] == "5"
    assert run.data.params["batch_size"] == "16"


def test_log_epoch_records_train_and_val_loss(tracker, tracking_uri):
    with tracker:
        tracker.log_epoch(epoch=1, train_loss=2.5, val_loss=2.8, lr=1e-4)
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert "train_loss" in run.data.metrics
    assert "val_loss" in run.data.metrics
    assert run.data.metrics["train_loss"] == pytest.approx(2.5)
    assert run.data.metrics["val_loss"] == pytest.approx(2.8)


def test_log_epoch_records_perplexity(tracker, tracking_uri):
    with tracker:
        tracker.log_epoch(epoch=1, train_loss=1.0, val_loss=1.2, lr=1e-4)
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert "train_perplexity" in run.data.metrics
    assert run.data.metrics["train_perplexity"] == pytest.approx(2.7183, rel=1e-2)


def test_log_final_metrics(tracker, tracking_uri):
    with tracker:
        tracker.log_final_metrics({"bleu": 28.4, "perplexity": 42.1})
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.metrics["final.bleu"] == pytest.approx(28.4)
    assert run.data.metrics["final.perplexity"] == pytest.approx(42.1)


def test_log_dataset_info(tracker, tracking_uri):
    with tracker:
        tracker.log_dataset_info(train_size=800, val_size=100, test_size=100)
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["dataset.train_size"] == "800"
    assert run.data.params["dataset.total_size"] == "1000"


def test_log_metadata(tracker, tracking_uri):
    with tracker:
        tracker.log_metadata(model_name="gpt2", device="cpu")
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["meta.model_name"] == "gpt2"
    assert run.data.params["meta.device"] == "cpu"


def test_training_time_logged(tracker, tracking_uri):
    with tracker:
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    run = client.get_run(run_id)
    assert "training_time_seconds" in run.data.metrics
    assert run.data.metrics["training_time_seconds"] >= 0


# ---------------------------------------------------------------------------
# test_model_artifact_logging
# ---------------------------------------------------------------------------

def test_log_model_artifact_dir(tracker, tracking_uri, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model": "test"}')
    (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 16)

    with tracker:
        tracker.log_model_artifact(str(model_dir))
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    artifacts = client.list_artifacts(run_id, "model")
    artifact_names = [a.path for a in artifacts]
    assert any("config.json" in name for name in artifact_names)


def test_log_model_artifact_single_file(tracker, tracking_uri, tmp_path):
    artifact_file = tmp_path / "training_args.json"
    artifact_file.write_text('{"lr": 0.0001}')

    with tracker:
        tracker.log_model_artifact(str(artifact_file))
        run_id = tracker.run_id

    client = get_client(tracking_uri)
    artifacts = client.list_artifacts(run_id, "model")
    assert len(artifacts) >= 1


# ---------------------------------------------------------------------------
# test_compare_runs
# ---------------------------------------------------------------------------

def test_compare_runs_returns_both(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("compare-exp")

    run_ids = []
    for bleu in [22.0, 31.0]:
        with mlflow.start_run():
            mlflow.log_metric("final.bleu", bleu)
            run_ids.append(mlflow.active_run().info.run_id)

    result = compare_runs(run_ids, tracking_uri)
    assert len(result["runs"]) == 2
    returned_ids = {r["run_id"] for r in result["runs"]}
    assert returned_ids == set(run_ids)


def test_get_experiment_runs(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment("runs-exp")
    with mlflow.start_run():
        mlflow.log_metric("val_loss", 1.5)

    runs = get_experiment_runs(exp.experiment_id, tracking_uri)
    assert len(runs) >= 1
    assert "val_loss" in runs[0]["metrics"]

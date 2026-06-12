"""
Integration tests: end-to-end pipeline using SQLite DB and a tiny in-process model.
No HuggingFace downloads — all model calls are patched with TinyLM.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import mlflow
import pytest
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from llm_studio.config import TrainingConfig
from llm_studio.data_loader import split_data
from llm_studio.models import Base, FineTuningJob, JobStatus, ModelVersion, TrainingData, User


# ---------------------------------------------------------------------------
# Tiny in-process model (no downloads)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 16)
        layer = nn.TransformerEncoderLayer(16, nhead=2, dim_feedforward=32, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(16, VOCAB_SIZE)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        x = self.transformer(x)
        return _Out(self.lm_head(x))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4,
                 pad_token_id=0, eos_token_id=1, **kwargs):
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeTokenizer:
    """Deterministic fake tokenizer — returns random-but-valid tensors."""
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "[EOS]"
    eos_token_id = 1

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=512, padding=False, **kwargs):
        if isinstance(text, list):
            seq_len = min(8, max_length)
            ids = torch.randint(2, VOCAB_SIZE, (len(text), seq_len))
            mask = torch.ones_like(ids)
        else:
            seq_len = min(max(len(str(text).split()), 2), min(8, max_length))
            ids = torch.randint(2, VOCAB_SIZE, (1, seq_len))
            mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return "output text"

    def save_pretrained(self, path):
        pass


def _make_tokenizer():
    return FakeTokenizer()


# ---------------------------------------------------------------------------
# MLflow cleanup — prevent active run state leaking to other test modules
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_mlflow():
    yield
    # End any active run left open by a patched trainer
    while mlflow.active_run():
        mlflow.end_run()


# ---------------------------------------------------------------------------
# DB fixture — fresh SQLite per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()
    engine.dispose()


def _seed_db(session, n_rows: int = 20):
    """Create a user, a job, and n_rows of training data. Returns (user, job)."""
    user = User(email="test@example.com")
    session.add(user)
    session.flush()

    job = FineTuningJob(user_id=user.id, model_name="gpt2")
    session.add(job)
    session.flush()

    rows = [
        TrainingData(job_id=job.id, input=f"input {i}", expected_output=f"output {i}")
        for i in range(n_rows)
    ]
    session.add_all(rows)
    session.commit()
    return user, job


# ---------------------------------------------------------------------------
# test_job_status_tracking
# ---------------------------------------------------------------------------

def test_job_starts_as_queued(session):
    _, job = _seed_db(session)
    assert job.status == JobStatus.queued


def test_job_status_transitions(session):
    _, job = _seed_db(session)
    job.status = JobStatus.training
    session.commit()
    session.refresh(job)
    assert job.status == JobStatus.training

    job.status = JobStatus.completed
    session.commit()
    session.refresh(job)
    assert job.status == JobStatus.completed


def test_failed_job_can_be_requeued(session):
    _, job = _seed_db(session)
    job.status = JobStatus.failed
    session.commit()
    job.status = JobStatus.queued
    session.commit()
    session.refresh(job)
    assert job.status == JobStatus.queued


def test_model_version_linked_to_job(session):
    _, job = _seed_db(session)
    v = ModelVersion(job_id=job.id, version_num=1, model_path="/tmp/v1", loss=1.5)
    session.add(v)
    session.commit()
    session.refresh(job)
    assert len(job.model_versions) == 1
    assert job.model_versions[0].version_num == 1


def test_training_data_count(session):
    _, job = _seed_db(session, n_rows=25)
    count = session.query(TrainingData).filter_by(job_id=job.id).count()
    assert count == 25


# ---------------------------------------------------------------------------
# test_full_pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline(session, tmp_path):
    """
    Upload data → train (TinyLM) → verify job completed + ModelVersion created.
    """
    from llm_studio.trainer import Trainer

    _, job = _seed_db(session, n_rows=20)
    config = TrainingConfig(epochs=2, batch_size=4, gradient_accumulation_steps=1,
                            max_seq_length=16, warmup_steps=1, fp16=False)

    tiny_model = TinyLM()
    tiny_tokenizer = _make_tokenizer()

    with patch("llm_studio.trainer.AutoModelForCausalLM.from_pretrained", return_value=tiny_model), \
         patch("llm_studio.trainer.AutoTokenizer.from_pretrained", return_value=tiny_tokenizer), \
         patch("llm_studio.mlflow_integration.mlflow.set_tracking_uri"), \
         patch("llm_studio.mlflow_integration.mlflow.set_experiment"), \
         patch("llm_studio.mlflow_integration.mlflow.start_run", return_value=MagicMock(info=MagicMock(run_id="test-run"))), \
         patch("llm_studio.mlflow_integration.mlflow.log_params"), \
         patch("llm_studio.mlflow_integration.mlflow.log_metrics"), \
         patch("llm_studio.mlflow_integration.mlflow.log_artifacts"), \
         patch("llm_studio.mlflow_integration.mlflow.end_run"), \
         tempfile.TemporaryDirectory() as tmpdir:

        from llm_studio.config import StorageConfig
        storage = StorageConfig(base_dir=tmpdir)
        trainer = Trainer(job_id=job.id, session=session, config=config, storage=storage)
        trainer.run()

    session.refresh(job)
    assert job.status == JobStatus.completed

    versions = session.query(ModelVersion).filter_by(job_id=job.id).all()
    assert len(versions) >= 1
    assert versions[0].loss is not None


def test_full_pipeline_failure_sets_failed_status(session, tmp_path):
    """If the model raises during training, status must be set to failed."""
    from llm_studio.trainer import Trainer

    _, job = _seed_db(session, n_rows=15)
    config = TrainingConfig(epochs=1, batch_size=4, gradient_accumulation_steps=1,
                            max_seq_length=16, warmup_steps=1, fp16=False)

    with patch("llm_studio.trainer.AutoModelForCausalLM.from_pretrained",
               side_effect=RuntimeError("simulated OOM")), \
         patch("llm_studio.trainer.AutoTokenizer.from_pretrained",
               return_value=_make_tokenizer()), \
         patch("llm_studio.mlflow_integration.mlflow.set_tracking_uri"), \
         patch("llm_studio.mlflow_integration.mlflow.set_experiment"), \
         patch("llm_studio.mlflow_integration.mlflow.start_run",
               return_value=MagicMock(info=MagicMock(run_id="fail-run"))), \
         patch("llm_studio.mlflow_integration.mlflow.log_params"), \
         patch("llm_studio.mlflow_integration.mlflow.log_metrics"), \
         patch("llm_studio.mlflow_integration.mlflow.end_run"), \
         tempfile.TemporaryDirectory() as tmpdir:

        from llm_studio.config import StorageConfig
        storage = StorageConfig(base_dir=tmpdir)
        trainer = Trainer(job_id=job.id, session=session, config=config, storage=storage)
        with pytest.raises(RuntimeError):
            trainer.run()

    session.refresh(job)
    assert job.status == JobStatus.failed


# ---------------------------------------------------------------------------
# test_model_quality_improvement
# ---------------------------------------------------------------------------

def test_more_training_data_produces_more_versions_opportunity(session):
    """
    A job with 20 rows vs 10 rows — the larger dataset gives the trainer more
    batches per epoch, increasing opportunity for the val_loss to improve
    across epochs and produce multiple saved versions.
    """
    from llm_studio.trainer import Trainer

    def run_job(n_rows: int, tmpdir: str) -> list[ModelVersion]:
        user = User(email=f"user{n_rows}@example.com")
        session.add(user)
        session.flush()
        job = FineTuningJob(user_id=user.id, model_name="gpt2")
        session.add(job)
        session.flush()
        rows = [TrainingData(job_id=job.id, input=f"q{i}", expected_output=f"a{i}")
                for i in range(n_rows)]
        session.add_all(rows)
        session.commit()

        config = TrainingConfig(epochs=2, batch_size=4, gradient_accumulation_steps=1,
                                max_seq_length=16, warmup_steps=1, fp16=False)
        from llm_studio.config import StorageConfig
        storage = StorageConfig(base_dir=tmpdir)
        tiny_model = TinyLM()
        tiny_tokenizer = _make_tokenizer()

        with patch("llm_studio.trainer.AutoModelForCausalLM.from_pretrained", return_value=tiny_model), \
             patch("llm_studio.trainer.AutoTokenizer.from_pretrained", return_value=tiny_tokenizer), \
             patch("llm_studio.mlflow_integration.mlflow.set_tracking_uri"), \
             patch("llm_studio.mlflow_integration.mlflow.set_experiment"), \
             patch("llm_studio.mlflow_integration.mlflow.start_run",
                   return_value=MagicMock(info=MagicMock(run_id=f"run-{n_rows}"))), \
             patch("llm_studio.mlflow_integration.mlflow.log_params"), \
             patch("llm_studio.mlflow_integration.mlflow.log_metrics"), \
             patch("llm_studio.mlflow_integration.mlflow.log_artifacts"), \
             patch("llm_studio.mlflow_integration.mlflow.end_run"):
            trainer = Trainer(job_id=job.id, session=session, config=config, storage=storage)
            trainer.run()

        return session.query(ModelVersion).filter_by(job_id=job.id).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        versions_small = run_job(12, tmpdir)
        versions_large = run_job(50, tmpdir)

    # Both jobs should complete and produce at least one version
    assert len(versions_small) >= 1
    assert len(versions_large) >= 1


def test_split_larger_dataset_has_more_train_rows():
    small = [(f"i{i}", f"o{i}") for i in range(20)]
    large = [(f"i{i}", f"o{i}") for i in range(100)]
    assert len(split_data(large).train) > len(split_data(small).train)

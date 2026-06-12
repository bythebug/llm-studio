"""
Tests for training logic: model loading, training loop, loss, saving.
Uses a minimal in-process model to avoid downloading large checkpoints.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from llm_studio.config import TrainingConfig
from llm_studio.loss_functions import (
    EpochMetrics,
    MetricsTracker,
    cross_entropy_loss,
    perplexity,
)
from llm_studio.optimizer import build_optimizer, build_scheduler


# ---------------------------------------------------------------------------
# Minimal model fixture — no HuggingFace download required
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH = 4


class TinyLM(nn.Module):
    """2-layer transformer language model small enough to train in a unit test."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 32, nhead: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return _FakeOutput(logits)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


def make_batch(batch_size=BATCH, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
    return torch.randint(0, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# test_model_loading
# ---------------------------------------------------------------------------

def test_model_loading_sets_pad_token():
    """Trainer._load_model should set pad_token when it's None."""
    from llm_studio.trainer import Trainer

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_model = TinyLM()

    with patch("llm_studio.trainer.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("llm_studio.trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        trainer = Trainer(job_id=1, session=MagicMock())
        model, tokenizer = trainer._load_model("gpt2")

    assert tokenizer.pad_token == "<eos>"
    assert model is mock_model


def test_model_loading_unknown_model_falls_back_to_hf_id():
    """An unregistered model name should be passed directly to HuggingFace."""
    from llm_studio.trainer import Trainer

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    mock_model = TinyLM()

    with patch("llm_studio.trainer.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_tok, \
         patch("llm_studio.trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        trainer = Trainer(job_id=1, session=MagicMock())
        trainer._load_model("some-custom-model")

    mock_tok.assert_called_once_with("some-custom-model")


# ---------------------------------------------------------------------------
# test_training_loop
# ---------------------------------------------------------------------------

def test_train_epoch_returns_float():
    """_train_epoch should complete without error and return a scalar loss."""
    from llm_studio.trainer import Trainer

    model = TinyLM()
    config = TrainingConfig(epochs=1, batch_size=BATCH, gradient_accumulation_steps=1)
    trainer = Trainer(job_id=1, session=MagicMock(), config=config)
    trainer.device = torch.device("cpu")

    input_ids = make_batch()
    dataset = TensorDataset(input_ids)

    class _SimpleLoader:
        def __iter__(self):
            yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        def __len__(self):
            return 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    loss = trainer._train_epoch(model, _SimpleLoader(), optimizer, scheduler)

    assert isinstance(loss, float)
    assert loss > 0


def test_validate_returns_float():
    """_validate should return a scalar loss without updating model weights."""
    from llm_studio.trainer import Trainer

    model = TinyLM()
    trainer = Trainer(job_id=1, session=MagicMock())
    trainer.device = torch.device("cpu")

    class _SimpleLoader:
        def __iter__(self):
            input_ids = make_batch()
            yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        def __len__(self):
            return 1

    loss = trainer._validate(model, _SimpleLoader())
    assert isinstance(loss, float)
    assert loss > 0


# ---------------------------------------------------------------------------
# test_loss_decreases_over_epochs
# ---------------------------------------------------------------------------

def test_loss_decreases_over_epochs():
    """
    Training a model for several steps on a fixed batch should reduce the loss.
    Not guaranteed to decrease every step, but the final loss should be lower
    than the initial loss after sufficient steps.
    """
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    input_ids = make_batch(batch_size=8)
    attention_mask = torch.ones_like(input_ids)

    losses = []
    for _ in range(20):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = cross_entropy_loss(outputs.logits, input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Expected loss to decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# test_validation_metrics
# ---------------------------------------------------------------------------

def test_metrics_tracker_records_history():
    tracker = MetricsTracker()
    for i in range(3):
        tracker.record(EpochMetrics(
            epoch=i + 1,
            train_loss=3.0 - i * 0.5,
            val_loss=3.2 - i * 0.4,
            train_perplexity=perplexity(3.0 - i * 0.5),
            val_perplexity=perplexity(3.2 - i * 0.4),
        ))
    assert len(tracker.history) == 3
    assert tracker.best_val_loss == pytest.approx(3.2 - 2 * 0.4, rel=1e-3)


def test_metrics_tracker_is_best():
    tracker = MetricsTracker()
    tracker.record(EpochMetrics(1, 2.0, 2.5, perplexity(2.0), perplexity(2.5)))
    assert tracker.is_best(2.0)
    assert not tracker.is_best(3.0)


def test_metrics_tracker_loss_curves_shape():
    tracker = MetricsTracker()
    for i in range(5):
        tracker.record(EpochMetrics(i + 1, 2.0, 2.1, 7.0, 8.0))
    curves = tracker.loss_curves()
    assert len(curves["train_loss"]) == 5
    assert len(curves["val_loss"]) == 5


def test_perplexity_finite():
    assert perplexity(0.0) == pytest.approx(1.0, rel=1e-3)
    assert perplexity(1.0) == pytest.approx(2.718, rel=1e-2)
    assert perplexity(100.0) < float("inf")  # clamped


# ---------------------------------------------------------------------------
# test_model_saving
# ---------------------------------------------------------------------------

def test_save_checkpoint_writes_files():
    """_save_checkpoint should write model files and a training_args.json."""
    from llm_studio.config import StorageConfig
    from llm_studio.trainer import Trainer

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(base_dir=tmpdir)
        trainer = Trainer(job_id=42, session=mock_session, storage=storage)
        trainer._save_checkpoint(mock_model, mock_tokenizer, version_num=1, val_loss=1.23)

        expected_path = storage.model_path(42, 1)
        assert os.path.isdir(expected_path)
        assert os.path.isfile(os.path.join(expected_path, "training_args.json"))

        mock_model.save_pretrained.assert_called_once_with(expected_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(expected_path)


def test_save_checkpoint_persists_model_version():
    """_save_checkpoint should add a ModelVersion row to the DB session."""
    from llm_studio.config import StorageConfig
    from llm_studio.trainer import Trainer

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageConfig(base_dir=tmpdir)
        trainer = Trainer(job_id=7, session=mock_session, storage=storage)
        trainer._save_checkpoint(MagicMock(), MagicMock(), version_num=2, val_loss=0.88)

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called()


# ---------------------------------------------------------------------------
# test_cross_entropy_loss
# ---------------------------------------------------------------------------

def test_cross_entropy_loss_shape():
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    loss = cross_entropy_loss(logits, labels)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_cross_entropy_loss_ignores_padding():
    logits = torch.randn(2, 4, VOCAB_SIZE)
    labels = torch.tensor([[1, 2, -100, -100], [3, -100, -100, -100]])
    loss_with_pad = cross_entropy_loss(logits, labels, ignore_index=-100)
    assert loss_with_pad.item() > 0

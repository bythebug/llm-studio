"""
Training orchestration: load model, run training loop, validate, save best checkpoint.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from sqlalchemy.orm import Session

from llm_studio.config import DEFAULT_STORAGE_CONFIG, DEFAULT_TRAINING_CONFIG, TrainingConfig
from llm_studio.data_loader import load_training_data, split_data
from llm_studio.loss_functions import EpochMetrics, MetricsTracker, cross_entropy_loss, perplexity
from llm_studio.models import FineTuningJob, JobStatus, ModelVersion
from llm_studio.optimizer import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FineTuningDataset(Dataset):
    """Tokenizes (input, output) pairs into model-ready tensors."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.encodings = []
        for inp, out in pairs:
            text = inp + tokenizer.eos_token + out + tokenizer.eos_token
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        job_id: int,
        session: Session,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        storage=DEFAULT_STORAGE_CONFIG,
    ) -> None:
        self.job_id = job_id
        self.session = session
        self.config = config
        self.storage = storage
        self.tracker = MetricsTracker()
        self.logs: list[dict] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        job = self._get_job()
        self._update_status(job, JobStatus.training)
        self._log(f"Training started on device: {self.device}")

        try:
            data = load_training_data(self.job_id, self.session)
            if not data:
                raise ValueError("No training data found for this job")

            split = split_data(data)
            self._log(f"Data split — train: {len(split.train)}, val: {len(split.val)}, test: {len(split.test)}")

            model, tokenizer = self._load_model(job.model_name)
            model.to(self.device)

            train_loader = self._make_loader(split.train, tokenizer, shuffle=True)
            val_loader = self._make_loader(split.val, tokenizer, shuffle=False)

            optimizer = build_optimizer(model, self.config)
            scheduler = build_scheduler(optimizer, self.config, len(train_loader))

            best_val_loss = float("inf")
            best_version = 0

            for epoch in range(1, self.config.epochs + 1):
                self._log(f"Epoch {epoch}/{self.config.epochs} — training")
                train_loss = self._train_epoch(model, train_loader, optimizer, scheduler)

                self._log(f"Epoch {epoch} — validating")
                val_loss = self._validate(model, val_loader)

                current_lr = scheduler.get_last_lr()[0]
                metrics = EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_perplexity=perplexity(train_loss),
                    val_perplexity=perplexity(val_loss),
                    lr=current_lr,
                )
                self.tracker.record(metrics)
                self._log(
                    f"Epoch {epoch} complete — "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_version += 1
                    self._save_checkpoint(model, tokenizer, best_version, val_loss)
                    self._log(f"New best model saved as v{best_version} (val_loss={val_loss:.4f})")

            self._update_status(job, JobStatus.completed)
            self._log("Training complete")
            self._persist_logs()

        except Exception as exc:
            self._update_status(job, JobStatus.failed)
            self._log(f"Training failed: {exc}", level="error")
            self._persist_logs()
            raise

    # ------------------------------------------------------------------
    # Epoch-level methods
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: PreTrainedModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> float:
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy_loss(outputs.logits, input_ids)

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate(self, model: PreTrainedModel, loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy_loss(outputs.logits, input_ids)
            total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        from llm_studio.config import BASE_MODELS
        hf_id = BASE_MODELS.get(model_name, {}).get("hf_id", model_name)
        self._log(f"Loading model: {hf_id}")

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16 if self.config.fp16 and self.device.type == "cuda" else torch.float32,
        )
        return model, tokenizer

    def _save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        version_num: int,
        val_loss: float,
    ) -> None:
        path = self.storage.model_path(self.job_id, version_num)
        os.makedirs(path, exist_ok=True)

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        with open(os.path.join(path, "training_args.json"), "w") as f:
            json.dump(
                {
                    "job_id": self.job_id,
                    "version_num": version_num,
                    "val_loss": val_loss,
                    "config": self.config.__dict__,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        existing = (
            self.session.query(ModelVersion)
            .filter_by(job_id=self.job_id, version_num=version_num)
            .first()
        )
        if not existing:
            self.session.add(
                ModelVersion(
                    job_id=self.job_id,
                    version_num=version_num,
                    model_path=path,
                    loss=val_loss,
                )
            )
            self.session.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        shuffle: bool,
    ) -> DataLoader:
        dataset = FineTuningDataset(pairs, tokenizer, self.config.max_seq_length)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _get_job(self) -> FineTuningJob:
        job = self.session.get(FineTuningJob, self.job_id)
        if not job:
            raise ValueError(f"Job {self.job_id} not found")
        return job

    def _update_status(self, job: FineTuningJob, status: JobStatus) -> None:
        job.status = status
        self.session.commit()

    def _log(self, message: str, level: str = "info") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        getattr(logger, level)(message)

    def _persist_logs(self) -> None:
        log_dir = os.path.join(self.storage.base_dir, "logs", f"job_{self.job_id}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "training.json"), "w") as f:
            json.dump(
                {"job_id": self.job_id, "logs": self.logs, "metrics": self.tracker.loss_curves()},
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_model(
    job_id: int,
    session: Session,
    config: Optional[TrainingConfig] = None,
) -> MetricsTracker:
    """Start training for a job. Returns the metrics tracker on completion."""
    trainer = Trainer(job_id, session, config=config or DEFAULT_TRAINING_CONFIG)
    trainer.run()
    return trainer.tracker

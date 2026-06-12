"""
Loss calculations for language model fine-tuning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_perplexity: float
    val_perplexity: float
    lr: float = 0.0


@dataclass
class MetricsTracker:
    history: list[EpochMetrics] = field(default_factory=list)

    def record(self, metrics: EpochMetrics) -> None:
        self.history.append(metrics)

    @property
    def best_val_loss(self) -> float:
        if not self.history:
            return float("inf")
        return min(m.val_loss for m in self.history)

    def is_best(self, val_loss: float) -> bool:
        return val_loss < self.best_val_loss

    def loss_curves(self) -> dict:
        return {
            "epochs": [m.epoch for m in self.history],
            "train_loss": [m.train_loss for m in self.history],
            "val_loss": [m.val_loss for m in self.history],
            "train_perplexity": [m.train_perplexity for m in self.history],
            "val_perplexity": [m.val_perplexity for m in self.history],
        }


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Causal LM loss: shift logits and labels by one position so each token
    predicts the next, then compute cross-entropy.

    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def perplexity(loss: float) -> float:
    """Exponentiate loss to get perplexity. Clamp to avoid overflow."""
    return math.exp(min(loss, 20.0))

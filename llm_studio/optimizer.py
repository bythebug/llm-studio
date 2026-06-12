"""
Optimizer and learning rate scheduler construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from llm_studio.config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    AdamW is the standard choice for transformer fine-tuning: Adam momentum
    with decoupled weight decay (doesn't apply decay to bias/LayerNorm params).
    Falls back to plain Adam or SGD via config.
    """
    decay_params, no_decay_params = _split_params(model)
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = getattr(config, "optimizer_name", "adamw").lower()
    if name == "adamw":
        return AdamW(param_groups, lr=config.learning_rate)
    if name == "adam":
        return Adam(param_groups, lr=config.learning_rate)
    if name == "sgd":
        return SGD(param_groups, lr=config.learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}. Choose adamw, adam, or sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> _LRScheduler:
    """
    Linear warmup followed by cosine annealing — standard for fine-tuning.
    Warmup avoids large gradient updates at the start when weights are
    still far from the fine-tuned optimum.
    """
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = min(config.warmup_steps, total_steps // 10)

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=config.learning_rate * 0.1,
    )
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_params(model: nn.Module) -> tuple[list, list]:
    """
    Separate parameters that should receive weight decay from those that
    should not (biases and LayerNorm weights).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay

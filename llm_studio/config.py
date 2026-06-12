import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Available base models
# ---------------------------------------------------------------------------

BASE_MODELS: dict[str, dict[str, Any]] = {
    "gpt2": {
        "hf_id": "gpt2",
        "max_tokens": 1024,
        "family": "causal-lm",
    },
    "gpt2-medium": {
        "hf_id": "gpt2-medium",
        "max_tokens": 1024,
        "family": "causal-lm",
    },
    "llama-3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "max_tokens": 8192,
        "family": "causal-lm",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "max_tokens": 4096,
        "family": "causal-lm",
    },
    "t5-small": {
        "hf_id": "t5-small",
        "max_tokens": 512,
        "family": "seq2seq",
    },
}


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    eval_steps: int = 500
    save_steps: int = 500
    fp16: bool = True


DEFAULT_TRAINING_CONFIG = TrainingConfig()


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    base_dir: str = field(default_factory=lambda: os.getenv("STORAGE_BASE", "./storage"))

    @property
    def models_dir(self) -> str:
        return os.path.join(self.base_dir, "models")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.base_dir, "checkpoints")

    @property
    def datasets_dir(self) -> str:
        return os.path.join(self.base_dir, "datasets")

    def model_path(self, job_id: int, version_num: int) -> str:
        return os.path.join(self.models_dir, f"job_{job_id}", f"v{version_num}")

    def checkpoint_path(self, job_id: int) -> str:
        return os.path.join(self.checkpoints_dir, f"job_{job_id}")


DEFAULT_STORAGE_CONFIG = StorageConfig()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost/llm_studio",
)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "llm-studio")

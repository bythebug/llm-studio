"""
Model loading with LRU in-memory cache.
Avoids re-loading weights from disk on every request.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from sqlalchemy.orm import Session

from llm_studio.models import ModelVersion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class CachedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    job_id: int
    version_num: int
    model_path: str


class ModelCache:
    """Thread-unsafe LRU cache — suitable for single-process FastAPI (no workers)."""

    def __init__(self, max_size: int = 3) -> None:
        self._store: OrderedDict[tuple, CachedModel] = OrderedDict()
        self.max_size = max_size

    def get(self, job_id: int, version_num: int) -> Optional[CachedModel]:
        key = (job_id, version_num)
        if key not in self._store:
            return None
        self._store.move_to_end(key)     # mark as recently used
        return self._store[key]

    def put(self, entry: CachedModel) -> None:
        key = (entry.job_id, entry.version_num)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        if len(self._store) > self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Cache evicted model %s", evicted_key)

    def evict(self, job_id: int, version_num: int) -> None:
        self._store.pop((job_id, version_num), None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def keys(self) -> list[tuple]:
        return list(self._store.keys())


# Module-level singleton — shared across all requests in a process.
_cache = ModelCache(max_size=3)


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------

def resolve_version(
    job_id: int,
    version_num: Optional[int],
    session: Session,
) -> ModelVersion:
    """
    If version_num is None, return the version with the lowest loss (best model).
    Raises ValueError if no versions exist.
    """
    query = session.query(ModelVersion).filter(ModelVersion.job_id == job_id)

    if version_num is not None:
        version = query.filter(ModelVersion.version_num == version_num).first()
        if not version:
            raise ValueError(f"Version {version_num} not found for job {job_id}")
        return version

    # Best = lowest loss; fall back to latest version_num if loss is NULL
    best = (
        query.filter(ModelVersion.loss.isnot(None))
        .order_by(ModelVersion.loss.asc())
        .first()
    )
    if best:
        return best

    latest = query.order_by(ModelVersion.version_num.desc()).first()
    if not latest:
        raise ValueError(f"No trained model versions found for job {job_id}")
    return latest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(
    job_id: int,
    session: Session,
    version_num: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> CachedModel:
    """
    Load a model for inference. Returns immediately from cache if already loaded.
    version_num=None → picks the best available version (lowest val loss).
    """
    version = resolve_version(job_id, version_num, session)

    cached = _cache.get(job_id, version.version_num)
    if cached:
        logger.debug("Cache hit: job=%d version=%d", job_id, version.version_num)
        return cached

    logger.info("Loading model from %s", version.model_path)
    device = device or _default_device()

    tokenizer = AutoTokenizer.from_pretrained(version.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        version.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    entry = CachedModel(
        model=model,
        tokenizer=tokenizer,
        job_id=job_id,
        version_num=version.version_num,
        model_path=version.model_path,
    )
    _cache.put(entry)
    return entry


def list_versions(job_id: int, session: Session) -> list[dict]:
    """Return all available model versions for a job, best first."""
    versions = (
        session.query(ModelVersion)
        .filter(ModelVersion.job_id == job_id)
        .order_by(ModelVersion.loss.asc().nullslast(), ModelVersion.version_num.desc())
        .all()
    )
    cached_keys = set(_cache.keys)
    return [
        {
            "version_num": v.version_num,
            "model_path": v.model_path,
            "loss": v.loss,
            "accuracy": v.accuracy,
            "created_at": v.created_at.isoformat(),
            "cached": (job_id, v.version_num) in cached_keys,
        }
        for v in versions
    ]


def get_cache() -> ModelCache:
    """Expose the module-level cache (primarily for testing)."""
    return _cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

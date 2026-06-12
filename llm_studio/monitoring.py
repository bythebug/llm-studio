"""
Production monitoring: latency tracking, prediction logging, drift detection, Prometheus metrics.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric definitions (module-level singletons)
# ---------------------------------------------------------------------------

INFERENCE_REQUESTS = Counter(
    "llm_inference_requests_total",
    "Total inference requests",
    ["job_id", "status"],          # status: success | error
)

INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_ms",
    "Inference latency in milliseconds",
    ["job_id"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

PREDICTION_CONFIDENCE = Histogram(
    "llm_prediction_confidence",
    "Per-request prediction confidence (0–1)",
    ["job_id"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ACTIVE_TRAINING_JOBS = Gauge(
    "llm_active_training_jobs",
    "Number of jobs currently in training state",
)

DRIFT_SCORE = Gauge(
    "llm_model_drift_score",
    "Rolling drift score — deviation from baseline confidence (higher = more drift)",
    ["job_id"],
)

PREDICTION_ERRORS = Counter(
    "llm_prediction_errors_total",
    "Total prediction errors by type",
    ["job_id", "error_type"],
)


# ---------------------------------------------------------------------------
# Prediction logger
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    job_id: int
    version_num: int
    input_text: str
    output_text: str
    confidence: float
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ground_truth: Optional[str] = None
    correct: Optional[bool] = None


class PredictionLogger:
    """Append-only audit log of every prediction, written as JSONL."""

    def __init__(self, log_dir: str = "./storage/prediction_logs") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, record: PredictionRecord) -> None:
        path = os.path.join(self.log_dir, f"job_{record.job_id}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def load(self, job_id: int) -> list[PredictionRecord]:
        path = os.path.join(self.log_dir, f"job_{job_id}.jsonl")
        if not os.path.exists(path):
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(PredictionRecord(**json.loads(line)))
        return records


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Detects confidence drift using a rolling window.

    Baseline = mean confidence over the first `baseline_size` predictions.
    Drift score = how many std deviations the current window mean is below the baseline.
    Alert fires when drift_score > threshold.
    """

    def __init__(
        self,
        job_id: int,
        baseline_size: int = 50,
        window_size: int = 20,
        alert_threshold: float = 2.0,
    ) -> None:
        self.job_id = job_id
        self.baseline_size = baseline_size
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self._baseline_samples: list[float] = []
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._window: deque[float] = deque(maxlen=window_size)
        self._alerts: list[dict] = []

    def record(self, confidence: float) -> Optional[str]:
        """
        Record a confidence score. Returns an alert message if drift detected,
        otherwise None.
        """
        if self._baseline_mean is None:
            self._baseline_samples.append(confidence)
            if len(self._baseline_samples) >= self.baseline_size:
                self._fit_baseline()
            return None

        self._window.append(confidence)
        if len(self._window) < self.window_size:
            return None

        score = self._drift_score()
        DRIFT_SCORE.labels(job_id=str(self.job_id)).set(score)

        if score > self.alert_threshold:
            msg = (
                f"[DRIFT ALERT] job={self.job_id} — drift_score={score:.2f} "
                f"(threshold={self.alert_threshold}). "
                f"Current window mean={sum(self._window)/len(self._window):.3f}, "
                f"baseline mean={self._baseline_mean:.3f}"
            )
            self._alerts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drift_score": score,
                "message": msg,
            })
            logger.warning(msg)
            return msg
        return None

    def _fit_baseline(self) -> None:
        import statistics
        self._baseline_mean = statistics.mean(self._baseline_samples)
        self._baseline_std = statistics.stdev(self._baseline_samples) or 1e-6

    def _drift_score(self) -> float:
        window_mean = sum(self._window) / len(self._window)
        return (self._baseline_mean - window_mean) / self._baseline_std

    @property
    def alerts(self) -> list[dict]:
        return list(self._alerts)

    @property
    def is_baseline_ready(self) -> bool:
        return self._baseline_mean is not None


# ---------------------------------------------------------------------------
# Latency tracker (in-memory percentile stats)
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Rolling window of latency samples with percentile reporting."""

    def __init__(self, job_id: int, maxlen: int = 1000) -> None:
        self.job_id = job_id
        self._samples: deque[float] = deque(maxlen=maxlen)

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        INFERENCE_LATENCY.labels(job_id=str(self.job_id)).observe(latency_ms)

    def stats(self) -> dict:
        if not self._samples:
            return {}
        import statistics
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(sorted_samples), 2),
            "p50_ms": round(sorted_samples[int(n * 0.50)], 2),
            "p95_ms": round(sorted_samples[int(n * 0.95)], 2),
            "p99_ms": round(sorted_samples[min(int(n * 0.99), n - 1)], 2),
            "max_ms": round(sorted_samples[-1], 2),
        }


# ---------------------------------------------------------------------------
# Monitoring facade — one instance per job
# ---------------------------------------------------------------------------

class JobMonitor:
    """
    Aggregate monitor for a single job: records every prediction,
    tracks latency, and runs drift detection.
    """

    def __init__(
        self,
        job_id: int,
        log_dir: str = "./storage/prediction_logs",
        baseline_size: int = 50,
    ) -> None:
        self.job_id = job_id
        self.logger = PredictionLogger(log_dir)
        self.latency = LatencyTracker(job_id)
        self.drift = DriftDetector(job_id, baseline_size=baseline_size)

    def record_prediction(
        self,
        version_num: int,
        input_text: str,
        output_text: str,
        confidence: float,
        latency_ms: float,
        ground_truth: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log the prediction, update metrics, and check for drift.
        Returns a drift alert message if triggered, otherwise None.
        """
        correct = None
        if ground_truth is not None:
            correct = output_text.strip() == ground_truth.strip()

        record = PredictionRecord(
            job_id=self.job_id,
            version_num=version_num,
            input_text=input_text,
            output_text=output_text,
            confidence=confidence,
            latency_ms=latency_ms,
            ground_truth=ground_truth,
            correct=correct,
        )
        self.logger.log(record)
        self.latency.record(latency_ms)

        INFERENCE_REQUESTS.labels(job_id=str(self.job_id), status="success").inc()
        PREDICTION_CONFIDENCE.labels(job_id=str(self.job_id)).observe(confidence)

        return self.drift.record(confidence)

    def record_error(self, error_type: str) -> None:
        INFERENCE_REQUESTS.labels(job_id=str(self.job_id), status="error").inc()
        PREDICTION_ERRORS.labels(job_id=str(self.job_id), error_type=error_type).inc()

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "latency_stats": self.latency.stats(),
            "drift_alerts": self.drift.alerts,
            "baseline_ready": self.drift.is_baseline_ready,
        }


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------

def prometheus_metrics() -> tuple[bytes, str]:
    """Return (metrics_bytes, content_type) for a /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST

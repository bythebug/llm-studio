"""
Compare evaluation results across model versions with statistical significance testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class VersionSnapshot:
    version_num: int
    metrics: dict
    task_type: str


@dataclass
class SignificanceResult:
    metric: str
    version_a: int
    version_b: int
    p_value: float
    significant: bool      # p < 0.05
    better_version: Optional[int]  # None if not significant


@dataclass
class ComparisonResult:
    job_id: int
    versions: list[VersionSnapshot]
    best_version: int
    primary_metric: str    # metric used to pick the winner
    significance: list[SignificanceResult]
    recommendation: str
    ranking: list[dict]    # versions sorted best → worst


# ---------------------------------------------------------------------------
# Main comparator
# ---------------------------------------------------------------------------

def compare_versions(
    job_id: int,
    eval_results: list[dict],
) -> ComparisonResult:
    """
    Compare evaluation results across versions.
    `eval_results` is a list of dicts loaded from persisted JSON files.
    """
    if not eval_results:
        raise ValueError("No evaluation results to compare")

    snapshots = [
        VersionSnapshot(
            version_num=r["version_num"],
            metrics=r["metrics"],
            task_type=r["task_type"],
        )
        for r in eval_results
    ]

    task_type = snapshots[0].task_type
    primary_metric = _primary_metric(task_type)

    ranking = _rank(snapshots, primary_metric)
    best = ranking[0]["version_num"]

    significance = []
    for i in range(len(snapshots)):
        for j in range(i + 1, len(snapshots)):
            sig = _pairwise_significance(snapshots[i], snapshots[j], primary_metric)
            if sig:
                significance.append(sig)

    return ComparisonResult(
        job_id=job_id,
        versions=snapshots,
        best_version=best,
        primary_metric=primary_metric,
        significance=significance,
        recommendation=_recommendation(ranking, primary_metric, significance),
        ranking=ranking,
    )


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def significance_test(
    scores_a: list[float],
    scores_b: list[float],
    metric: str = "score",
) -> SignificanceResult:
    """
    Welch's t-test (unequal variance) on two lists of per-sample scores.
    Returns whether the difference is statistically significant (p < 0.05).
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return SignificanceResult(metric=metric, version_a=0, version_b=1,
                                  p_value=1.0, significant=False, better_version=None)

    _, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)

    higher_is_better = metric not in ("perplexity", "loss")
    if higher_is_better:
        better = 0 if mean_a > mean_b else 1
    else:
        better = 0 if mean_a < mean_b else 1

    return SignificanceResult(
        metric=metric,
        version_a=0,
        version_b=1,
        p_value=round(float(p_value), 4),
        significant=bool(p_value < 0.05),
        better_version=better if p_value < 0.05 else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_metric(task_type: str) -> str:
    return "f1" if task_type == "classification" else "bleu"


def _rank(snapshots: list[VersionSnapshot], metric: str) -> list[dict]:
    higher_is_better = metric not in ("perplexity", "loss")
    ranked = sorted(
        snapshots,
        key=lambda s: s.metrics.get(metric, 0),
        reverse=higher_is_better,
    )
    return [
        {
            "rank": i + 1,
            "version_num": s.version_num,
            "metrics": s.metrics,
        }
        for i, s in enumerate(ranked)
    ]


def _pairwise_significance(
    a: VersionSnapshot,
    b: VersionSnapshot,
    metric: str,
) -> Optional[SignificanceResult]:
    score_a = a.metrics.get(metric)
    score_b = b.metrics.get(metric)
    if score_a is None or score_b is None:
        return None

    higher_is_better = metric not in ("perplexity", "loss")
    better = a.version_num if (
        (higher_is_better and score_a > score_b) or
        (not higher_is_better and score_a < score_b)
    ) else b.version_num

    # Single-point comparison — no p-value possible without per-sample scores.
    # Mark as significant only if the difference is meaningful (>5% relative).
    diff = abs(score_a - score_b) / max(abs(score_a), 1e-9)
    significant = diff > 0.05

    return SignificanceResult(
        metric=metric,
        version_a=a.version_num,
        version_b=b.version_num,
        p_value=None,  # requires per-sample scores; use significance_test() for that
        significant=significant,
        better_version=better if significant else None,
    )


def _recommendation(
    ranking: list[dict],
    metric: str,
    significance: list[SignificanceResult],
) -> str:
    if len(ranking) == 1:
        return f"Only one version evaluated. Deploy v{ranking[0]['version_num']}."

    best = ranking[0]
    second = ranking[1]
    sig = next(
        (s for s in significance
         if {s.version_a, s.version_b} == {best['version_num'], second['version_num']}),
        None,
    )

    best_score = best["metrics"].get(metric, "N/A")
    second_score = second["metrics"].get(metric, "N/A")

    if sig and sig.significant:
        return (
            f"Deploy v{best['version_num']} — meaningfully better on {metric} "
            f"({best_score} vs {second_score})."
        )
    return (
        f"v{best['version_num']} leads on {metric} ({best_score} vs {second_score}), "
        f"but the gap is small. Either version is acceptable; prefer v{best['version_num']} "
        f"for the marginal edge."
    )

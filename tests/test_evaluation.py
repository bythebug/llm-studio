"""
Tests for evaluation metrics, model comparison, and confusion matrix.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from llm_studio.comparator import (
    ComparisonResult,
    VersionSnapshot,
    compare_versions,
    significance_test,
)
from llm_studio.metrics import (
    ClassificationMetrics,
    GenerationMetrics,
    accuracy,
    bleu_score,
    classification_metrics,
    confusion_matrix_data,
    generation_metrics,
    interpret_classification,
    interpret_generation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Y_TRUE = ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "dog", "cat", "dog"]
Y_PRED = ["cat", "dog", "dog", "bird", "dog", "cat", "cat",  "dog", "cat", "cat"]
LABELS = ["bird", "cat", "dog"]

HYPOTHESES = [
    "the cat sat on the mat",
    "a dog runs in the park",
    "the bird sings a song",
]
REFERENCES = [
    "the cat sat on the mat",
    "a dog is running in the park",
    "the bird sang a song",
]


# ---------------------------------------------------------------------------
# test_accuracy_calculation
# ---------------------------------------------------------------------------

def test_accuracy_perfect():
    assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_zero():
    assert accuracy(["a", "b"], ["b", "a"]) == 0.0


def test_accuracy_partial():
    result = accuracy(["a", "b", "c", "d"], ["a", "b", "x", "x"])
    assert result == pytest.approx(0.5)


def test_classification_metrics_returns_correct_type():
    result = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    assert isinstance(result, ClassificationMetrics)


def test_classification_metrics_accuracy_in_range():
    result = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    assert 0.0 <= result.accuracy <= 1.0


def test_classification_metrics_per_class_keys():
    result = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    assert set(result.per_class.keys()) == set(LABELS)


def test_classification_metrics_per_class_fields():
    result = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    for v in result.per_class.values():
        assert "precision" in v and "recall" in v and "f1" in v and "support" in v


# ---------------------------------------------------------------------------
# test_metric_consistency
# ---------------------------------------------------------------------------

def test_f1_between_precision_and_recall():
    """F1 is always between precision and recall."""
    result = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    assert min(result.precision, result.recall) <= result.f1 <= max(result.precision, result.recall)


def test_perfect_predictions_give_perfect_scores():
    result = classification_metrics(Y_TRUE, Y_TRUE, LABELS)
    assert result.accuracy == 1.0
    assert result.f1 == pytest.approx(1.0, abs=1e-3)
    assert result.precision == pytest.approx(1.0, abs=1e-3)
    assert result.recall == pytest.approx(1.0, abs=1e-3)


def test_bleu_perfect_match():
    texts = ["the cat sat on the mat"]
    score = bleu_score(texts, texts)
    assert score == pytest.approx(100.0, abs=1.0)


def test_bleu_empty_hypothesis_is_low():
    score = bleu_score([""], ["the cat sat on the mat"])
    assert score == 0.0


def test_bleu_partial_overlap_is_between_0_and_100():
    score = bleu_score(HYPOTHESES, REFERENCES)
    assert 0.0 <= score <= 100.0


def test_generation_metrics_returns_correct_type():
    result = generation_metrics(HYPOTHESES, REFERENCES, loss=2.0)
    assert isinstance(result, GenerationMetrics)
    assert result.bleu >= 0
    assert result.perplexity > 0


# ---------------------------------------------------------------------------
# test_confusion_matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_shape():
    cm = confusion_matrix_data(Y_TRUE, Y_PRED, LABELS)
    assert len(cm["matrix"]) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in cm["matrix"])


def test_confusion_matrix_diagonal_are_correct_predictions():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "a"]
    cm = confusion_matrix_data(y_true, y_pred, ["a", "b"])
    # diagonal: a→a=1, b→b=1
    assert cm["matrix"][0][0] == 1
    assert cm["matrix"][1][1] == 1


def test_confusion_matrix_labels_match():
    cm = confusion_matrix_data(Y_TRUE, Y_PRED, LABELS)
    assert cm["labels"] == [str(l) for l in LABELS]


# ---------------------------------------------------------------------------
# test_model_comparison
# ---------------------------------------------------------------------------

EVAL_V1 = {
    "job_id": 1, "version_num": 1, "task_type": "generation",
    "metrics": {"bleu": 22.5, "perplexity": 45.0, "avg_output_length": 12.0},
    "interpretation": {}, "sample_predictions": [],
}
EVAL_V2 = {
    "job_id": 1, "version_num": 2, "task_type": "generation",
    "metrics": {"bleu": 31.0, "perplexity": 30.0, "avg_output_length": 11.5},
    "interpretation": {}, "sample_predictions": [],
}
EVAL_V3 = {
    "job_id": 1, "version_num": 3, "task_type": "generation",
    "metrics": {"bleu": 31.1, "perplexity": 29.5, "avg_output_length": 11.0},
    "interpretation": {}, "sample_predictions": [],
}


def test_compare_versions_picks_best_bleu():
    result = compare_versions(1, [EVAL_V1, EVAL_V2])
    assert result.best_version == 2


def test_compare_versions_ranking_order():
    result = compare_versions(1, [EVAL_V1, EVAL_V2, EVAL_V3])
    ranks = [r["version_num"] for r in result.ranking]
    assert ranks[0] in (2, 3)  # v2 and v3 are close; v1 is last
    assert ranks[-1] == 1


def test_compare_versions_recommendation_is_string():
    result = compare_versions(1, [EVAL_V1, EVAL_V2])
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_compare_versions_requires_at_least_one():
    with pytest.raises(ValueError):
        compare_versions(1, [])


def test_compare_single_version_names_best():
    result = compare_versions(1, [EVAL_V1])
    assert result.best_version == 1


def test_significance_test_detects_clear_difference():
    scores_a = [0.9, 0.91, 0.88, 0.92, 0.89]
    scores_b = [0.4, 0.42, 0.38, 0.41, 0.40]
    result = significance_test(scores_a, scores_b, metric="accuracy")
    assert result.significant
    assert result.p_value < 0.05


def test_significance_test_no_difference():
    scores = [0.85, 0.86, 0.84, 0.85, 0.86]
    result = significance_test(scores, scores, metric="f1")
    assert not result.significant


def test_significance_test_too_few_samples():
    result = significance_test([0.9], [0.8], metric="bleu")
    assert not result.significant


# ---------------------------------------------------------------------------
# test interpretation guides
# ---------------------------------------------------------------------------

def test_interpret_classification_returns_dict():
    clf = classification_metrics(Y_TRUE, Y_PRED, LABELS)
    interp = interpret_classification(clf)
    assert "accuracy" in interp and "f1" in interp


def test_interpret_generation_labels():
    interp = interpret_generation(bleu=35.0, ppl=18.0)
    assert "bleu" in interp and "perplexity" in interp
    assert "Near human quality" in interp["bleu"] or "Good" in interp["bleu"]
    assert "Excellent" in interp["perplexity"]


def test_interpret_generation_poor_scores():
    interp = interpret_generation(bleu=5.0, ppl=200.0)
    assert "Poor" in interp["bleu"]
    assert "High" in interp["perplexity"]

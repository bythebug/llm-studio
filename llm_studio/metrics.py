"""
Metric calculations for both classification and generation tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sacrebleu
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)

from llm_studio.loss_functions import perplexity


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class: dict[str, dict]  # label → {precision, recall, f1, support}
    confusion_matrix: list[list[int]]
    labels: list[str]


@dataclass
class GenerationMetrics:
    bleu: float          # 0–100
    perplexity: float    # lower is better
    avg_output_length: float
    interpretation: dict[str, str]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: list, y_pred: list) -> float:
    return float(accuracy_score(y_true, y_pred))


def classification_metrics(
    y_true: list,
    y_pred: list,
    labels: Optional[list[str]] = None,
) -> ClassificationMetrics:
    labels = labels or sorted(set(y_true) | set(y_pred))
    label_set = list(labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_set, average=None, zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class = {
        str(lbl): {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, lbl in enumerate(label_set)
    }

    cm = sk_confusion_matrix(y_true, y_pred, labels=label_set).tolist()

    return ClassificationMetrics(
        accuracy=round(accuracy(y_true, y_pred), 4),
        precision=round(float(wp), 4),
        recall=round(float(wr), 4),
        f1=round(float(wf1), 4),
        per_class=per_class,
        confusion_matrix=cm,
        labels=[str(l) for l in label_set],
    )


def confusion_matrix_data(
    y_true: list, y_pred: list, labels: Optional[list[str]] = None
) -> dict:
    """Return confusion matrix in a format ready for frontend rendering."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    labels = [str(l) for l in labels]
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"labels": labels, "matrix": cm}


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def bleu_score(hypotheses: list[str], references: list[str]) -> float:
    """
    Corpus-level BLEU using sacreBLEU (tokenization-agnostic, reproducible).
    Returns score in [0, 100].
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(result.score, 2)


def generation_metrics(
    hypotheses: list[str],
    references: list[str],
    loss: float,
) -> GenerationMetrics:
    bleu = bleu_score(hypotheses, references)
    ppl = perplexity(loss)
    avg_len = sum(len(h.split()) for h in hypotheses) / max(len(hypotheses), 1)

    return GenerationMetrics(
        bleu=bleu,
        perplexity=round(ppl, 2),
        avg_output_length=round(avg_len, 1),
        interpretation=interpret_generation(bleu, ppl),
    )


# ---------------------------------------------------------------------------
# Interpretation guides
# ---------------------------------------------------------------------------

def interpret_classification(metrics: ClassificationMetrics) -> dict[str, str]:
    acc = metrics.accuracy
    f1 = metrics.f1

    acc_note = (
        "Excellent" if acc >= 0.95
        else "Good" if acc >= 0.85
        else "Moderate" if acc >= 0.70
        else "Poor — consider more data or a larger model"
    )
    f1_note = (
        "Excellent" if f1 >= 0.90
        else "Good" if f1 >= 0.80
        else "Moderate" if f1 >= 0.65
        else "Poor — check class imbalance and training data quality"
    )

    return {
        "accuracy": f"{acc:.1%} — {acc_note}",
        "f1": f"{f1:.4f} — {f1_note}",
        "tip": (
            "Low recall with high precision → model is too conservative (misses many positives). "
            "Low precision with high recall → model is too aggressive (many false positives)."
        ),
    }


def interpret_generation(bleu: float, ppl: float) -> dict[str, str]:
    bleu_note = (
        "Near human quality" if bleu >= 40
        else "Good fluency" if bleu >= 25
        else "Understandable" if bleu >= 15
        else "Poor — outputs diverge significantly from references"
    )
    ppl_note = (
        "Excellent" if ppl <= 20
        else "Good" if ppl <= 50
        else "Moderate" if ppl <= 100
        else "High — model is uncertain; more training or data may help"
    )

    return {
        "bleu": f"{bleu:.1f}/100 — {bleu_note}",
        "perplexity": f"{ppl:.1f} — {ppl_note}",
        "tip": (
            "BLEU measures n-gram overlap with reference outputs. "
            "Perplexity measures how confident the model is per token. "
            "Both should decrease as training improves."
        ),
    }

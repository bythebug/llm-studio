"""
Model evaluation: run a fine-tuned model on test data and compute metrics.
"""
from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from sqlalchemy.orm import Session

from llm_studio.config import DEFAULT_STORAGE_CONFIG, DEFAULT_TRAINING_CONFIG, StorageConfig, TrainingConfig
from llm_studio.data_loader import load_training_data, split_data
from llm_studio.loss_functions import cross_entropy_loss, perplexity
from llm_studio.metrics import (
    ClassificationMetrics,
    GenerationMetrics,
    classification_metrics,
    generation_metrics,
    interpret_classification,
)
from llm_studio.models import ModelVersion


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    job_id: int
    version_num: int
    task_type: TaskType
    metrics: dict
    sample_predictions: list[dict]   # first N predictions for inspection
    interpretation: dict[str, str]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        job_id: int,
        version_num: int,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        storage: StorageConfig = DEFAULT_STORAGE_CONFIG,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.job_id = job_id
        self.version_num = version_num
        self.config = config
        self.storage = storage
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def evaluate(
        self,
        test_data: list[tuple[str, str]],
        task_type: TaskType = TaskType.GENERATION,
        max_samples: int = 10,
    ) -> EvaluationResult:
        if task_type == TaskType.GENERATION:
            return self._evaluate_generation(test_data, max_samples)
        return self._evaluate_classification(test_data, max_samples)

    # ------------------------------------------------------------------

    def _evaluate_generation(
        self, test_data: list[tuple[str, str]], max_samples: int
    ) -> EvaluationResult:
        self.model.eval()
        hypotheses, references = [], []
        total_loss = 0.0
        sample_preds = []

        with torch.no_grad():
            for inp, ref in test_data:
                hypothesis = self._generate(inp)
                hypotheses.append(hypothesis)
                references.append(ref)
                total_loss += self._compute_loss(inp, ref)

                if len(sample_preds) < max_samples:
                    sample_preds.append({"input": inp, "expected": ref, "predicted": hypothesis})

        avg_loss = total_loss / max(len(test_data), 1)
        gen_metrics = generation_metrics(hypotheses, references, avg_loss)

        metrics_dict = {
            "bleu": gen_metrics.bleu,
            "perplexity": gen_metrics.perplexity,
            "avg_output_length": gen_metrics.avg_output_length,
        }

        result = EvaluationResult(
            job_id=self.job_id,
            version_num=self.version_num,
            task_type=TaskType.GENERATION,
            metrics=metrics_dict,
            sample_predictions=sample_preds,
            interpretation=gen_metrics.interpretation,
        )
        self._persist(result)
        return result

    def _evaluate_classification(
        self, test_data: list[tuple[str, str]], max_samples: int
    ) -> EvaluationResult:
        self.model.eval()
        y_true, y_pred = [], []
        sample_preds = []

        with torch.no_grad():
            for inp, label in test_data:
                predicted = self._generate(inp)
                y_true.append(label.strip())
                y_pred.append(predicted.strip())

                if len(sample_preds) < max_samples:
                    sample_preds.append({"input": inp, "expected": label, "predicted": predicted})

        clf_metrics = classification_metrics(y_true, y_pred)
        metrics_dict = {
            "accuracy": clf_metrics.accuracy,
            "precision": clf_metrics.precision,
            "recall": clf_metrics.recall,
            "f1": clf_metrics.f1,
            "per_class": clf_metrics.per_class,
        }

        result = EvaluationResult(
            job_id=self.job_id,
            version_num=self.version_num,
            task_type=TaskType.CLASSIFICATION,
            metrics=metrics_dict,
            sample_predictions=sample_preds,
            interpretation=interpret_classification(clf_metrics),
        )
        self._persist(result)
        return result

    # ------------------------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length // 2,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _compute_loss(self, inp: str, out: str) -> float:
        text = inp + self.tokenizer.eos_token + out
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)
            loss = cross_entropy_loss(outputs.logits, enc["input_ids"])
        return loss.item()

    def _persist(self, result: EvaluationResult) -> None:
        eval_dir = os.path.join(
            self.storage.base_dir, "evals", f"job_{self.job_id}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        path = os.path.join(eval_dir, f"v{self.version_num}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "job_id": result.job_id,
                    "version_num": result.version_num,
                    "task_type": result.task_type.value,
                    "metrics": result.metrics,
                    "interpretation": result.interpretation,
                    "sample_predictions": result.sample_predictions,
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: list[tuple[str, str]],
    job_id: int,
    version_num: int,
    task_type: TaskType = TaskType.GENERATION,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> EvaluationResult:
    evaluator = Evaluator(model, tokenizer, job_id, version_num, config)
    return evaluator.evaluate(test_data, task_type)


def load_eval_result(job_id: int, version_num: int, storage: StorageConfig = DEFAULT_STORAGE_CONFIG) -> Optional[dict]:
    """Load a previously persisted evaluation result from disk."""
    path = os.path.join(storage.base_dir, "evals", f"job_{job_id}", f"v{version_num}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

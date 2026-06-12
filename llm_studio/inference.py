"""
Inference: single and batch predictions with latency tracking and confidence scoring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_studio.config import DEFAULT_TRAINING_CONFIG, TrainingConfig


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    output: str
    confidence: float           # mean per-token probability of the generated text (0–1)
    input_token_count: int
    output_token_count: int
    latency_ms: float


@dataclass
class BatchInferenceResult:
    results: list[InferenceResult]
    total_latency_ms: float
    avg_latency_ms: float = field(init=False)

    def __post_init__(self):
        n = len(self.results)
        self.avg_latency_ms = round(self.total_latency_ms / n, 2) if n else 0.0


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    max_new_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> InferenceResult:
    device = device or next(model.parameters()).device
    t0 = time.perf_counter()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length // 2,
        padding=False,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,          # greedy — deterministic, fast
            temperature=1.0,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    generated_ids = output_ids[0][input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    output_len = generated_ids.shape[0]

    confidence = _confidence(model, output_ids[0], input_len, device)

    return InferenceResult(
        output=output_text,
        confidence=round(confidence, 4),
        input_token_count=input_len,
        output_token_count=output_len,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    max_new_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> BatchInferenceResult:
    """
    Tokenize all inputs together and run a single batched forward pass.
    Faster than calling predict() in a loop because GPU utilization is higher.
    """
    if not inputs:
        return BatchInferenceResult(results=[], total_latency_ms=0.0)

    device = device or next(model.parameters()).device
    t0 = time.perf_counter()

    encodings = tokenizer(
        inputs,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length // 2,
        padding=True,
    ).to(device)

    input_lengths = encodings["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        output_ids = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    total_latency_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = round(total_latency_ms / len(inputs), 2)

    results = []
    for i, (ids, inp_len) in enumerate(zip(output_ids, input_lengths)):
        inp_len = int(inp_len)
        generated_ids = ids[inp_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        confidence = _confidence(model, ids, inp_len, device)

        results.append(InferenceResult(
            output=output_text,
            confidence=round(confidence, 4),
            input_token_count=inp_len,
            output_token_count=int((generated_ids != tokenizer.pad_token_id).sum()),
            latency_ms=per_item_ms,
        ))

    return BatchInferenceResult(results=results, total_latency_ms=round(total_latency_ms, 2))


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _confidence(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    input_len: int,
    device: torch.device,
) -> float:
    """
    Average probability the model assigned to each generated token.
    Re-runs a forward pass on the full sequence (input + output).
    Returns a value in [0, 1]; higher = more confident.
    """
    if input_len >= len(token_ids):
        return 0.0

    with torch.no_grad():
        logits = model(token_ids.unsqueeze(0).to(device)).logits[0]  # (seq_len, vocab)

    probs = F.softmax(logits, dim=-1)

    # Probability of each generated token (positions input_len-1 to end-1 predict input_len to end)
    generated_token_ids = token_ids[input_len:]
    pred_probs = probs[input_len - 1: input_len - 1 + len(generated_token_ids)]

    if pred_probs.shape[0] == 0:
        return 0.0

    token_probs = pred_probs[
        torch.arange(len(generated_token_ids), device=device),
        generated_token_ids.to(device),
    ]
    return float(token_probs.mean().cpu())

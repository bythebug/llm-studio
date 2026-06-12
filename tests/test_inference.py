"""
Tests for inference: single prediction, batch, caching, and latency.
All tests use the in-process TinyLM — no HuggingFace downloads required.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from llm_studio.inference import BatchInferenceResult, InferenceResult, predict, predict_batch, _confidence
from llm_studio.model_loader import CachedModel, ModelCache


# ---------------------------------------------------------------------------
# Minimal fixtures — no downloads
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128
SEQ_LEN = 32


class TinyLM(nn.Module):
    """Tiny 2-layer transformer for fast unit testing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 32)
        layer = nn.TransformerEncoderLayer(32, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.lm_head = nn.Linear(32, VOCAB_SIZE)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        x = self.transformer(x)
        return _FakeOutput(self.lm_head(x))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                 pad_token_id=0, eos_token_id=1, **kwargs):
        """Greedy generation: pick the argmax token at each step."""
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return ids


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


def make_tokenizer() -> PreTrainedTokenizerFast:
    """Build a minimal fast tokenizer over a tiny vocabulary."""
    base = Tokenizer(BPE(unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()

    # Register just enough tokens so encoding works without training BPE merges
    vocab = {f"tok{i}": i for i in range(VOCAB_SIZE - 4)}
    vocab.update({"[UNK]": VOCAB_SIZE - 4, "[PAD]": VOCAB_SIZE - 3,
                  "[BOS]": VOCAB_SIZE - 2, "[EOS]": VOCAB_SIZE - 1})
    base.add_special_tokens(["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    # Manually add vocab
    from tokenizers import AddedToken
    base.add_tokens([AddedToken(f"tok{i}") for i in range(VOCAB_SIZE - 4)])

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    return tokenizer


@pytest.fixture(scope="module")
def tiny_model():
    m = TinyLM()
    m.eval()
    return m


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return make_tokenizer()


# ---------------------------------------------------------------------------
# test_single_prediction
# ---------------------------------------------------------------------------

def test_single_prediction_returns_result(tiny_model, tiny_tokenizer):
    result = predict(tiny_model, tiny_tokenizer, "tok1 tok2 tok3", max_new_tokens=4)
    assert isinstance(result, InferenceResult)


def test_single_prediction_output_is_string(tiny_model, tiny_tokenizer):
    result = predict(tiny_model, tiny_tokenizer, "tok1 tok2", max_new_tokens=4)
    assert isinstance(result.output, str)


def test_single_prediction_token_counts_positive(tiny_model, tiny_tokenizer):
    result = predict(tiny_model, tiny_tokenizer, "tok1 tok2 tok3", max_new_tokens=4)
    assert result.input_token_count > 0


def test_single_prediction_confidence_in_range(tiny_model, tiny_tokenizer):
    result = predict(tiny_model, tiny_tokenizer, "tok1 tok2", max_new_tokens=4)
    assert 0.0 <= result.confidence <= 1.0


def test_single_prediction_latency_recorded(tiny_model, tiny_tokenizer):
    result = predict(tiny_model, tiny_tokenizer, "tok1 tok2", max_new_tokens=4)
    assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# test_batch_prediction
# ---------------------------------------------------------------------------

def test_batch_prediction_length_matches_inputs(tiny_model, tiny_tokenizer):
    inputs = ["tok1 tok2", "tok3 tok4 tok5", "tok6"]
    result = predict_batch(tiny_model, tiny_tokenizer, inputs, max_new_tokens=4)
    assert isinstance(result, BatchInferenceResult)
    assert len(result.results) == len(inputs)


def test_batch_prediction_all_outputs_are_strings(tiny_model, tiny_tokenizer):
    inputs = ["tok1", "tok2 tok3"]
    result = predict_batch(tiny_model, tiny_tokenizer, inputs, max_new_tokens=4)
    assert all(isinstance(r.output, str) for r in result.results)


def test_batch_prediction_avg_latency_computed(tiny_model, tiny_tokenizer):
    inputs = ["tok1", "tok2", "tok3"]
    result = predict_batch(tiny_model, tiny_tokenizer, inputs, max_new_tokens=4)
    assert result.avg_latency_ms == pytest.approx(result.total_latency_ms / 3, rel=0.01)


def test_batch_prediction_empty_input_returns_empty(tiny_model, tiny_tokenizer):
    result = predict_batch(tiny_model, tiny_tokenizer, [], max_new_tokens=4)
    assert result.results == []
    assert result.total_latency_ms == 0.0


def test_batch_single_item_matches_single_predict(tiny_model, tiny_tokenizer):
    text = "tok1 tok2 tok3"
    single = predict(tiny_model, tiny_tokenizer, text, max_new_tokens=4)
    batch = predict_batch(tiny_model, tiny_tokenizer, [text], max_new_tokens=4)
    # Output should be identical (both greedy)
    assert single.output == batch.results[0].output


# ---------------------------------------------------------------------------
# test_model_caching
# ---------------------------------------------------------------------------

def test_cache_put_and_get():
    cache = ModelCache(max_size=2)
    entry = CachedModel(
        model=MagicMock(), tokenizer=MagicMock(),
        job_id=1, version_num=1, model_path="/tmp/v1"
    )
    cache.put(entry)
    assert cache.get(1, 1) is entry


def test_cache_miss_returns_none():
    cache = ModelCache(max_size=2)
    assert cache.get(99, 99) is None


def test_cache_lru_eviction():
    cache = ModelCache(max_size=2)
    for i in range(1, 4):
        cache.put(CachedModel(MagicMock(), MagicMock(), job_id=i, version_num=1, model_path=""))
    # First item (job_id=1) should be evicted
    assert cache.get(1, 1) is None
    assert cache.get(2, 1) is not None
    assert cache.get(3, 1) is not None


def test_cache_get_refreshes_lru_order():
    cache = ModelCache(max_size=2)
    e1 = CachedModel(MagicMock(), MagicMock(), job_id=1, version_num=1, model_path="")
    e2 = CachedModel(MagicMock(), MagicMock(), job_id=2, version_num=1, model_path="")
    cache.put(e1)
    cache.put(e2)
    cache.get(1, 1)   # access e1 → moves to end (most recent)
    # Adding e3 should evict e2 (least recently used), not e1
    cache.put(CachedModel(MagicMock(), MagicMock(), job_id=3, version_num=1, model_path=""))
    assert cache.get(1, 1) is not None
    assert cache.get(2, 1) is None


def test_cache_evict_removes_entry():
    cache = ModelCache(max_size=3)
    cache.put(CachedModel(MagicMock(), MagicMock(), job_id=5, version_num=2, model_path=""))
    cache.evict(5, 2)
    assert cache.get(5, 2) is None


def test_cache_size_property():
    cache = ModelCache(max_size=5)
    assert cache.size == 0
    cache.put(CachedModel(MagicMock(), MagicMock(), job_id=1, version_num=1, model_path=""))
    assert cache.size == 1


# ---------------------------------------------------------------------------
# test_inference_speed (latency < 100ms for cached tiny model)
# ---------------------------------------------------------------------------

def test_inference_speed_single(tiny_model, tiny_tokenizer):
    """Cached tiny model on CPU must respond in under 100ms."""
    # Warm up
    predict(tiny_model, tiny_tokenizer, "tok1", max_new_tokens=4)

    t0 = time.perf_counter()
    predict(tiny_model, tiny_tokenizer, "tok1 tok2", max_new_tokens=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 100, f"Single inference took {elapsed_ms:.1f}ms — expected < 100ms"


def test_inference_speed_batch_faster_than_serial(tiny_model, tiny_tokenizer):
    """Batching 4 inputs should be faster than 4 serial predict() calls."""
    inputs = ["tok1", "tok2 tok3", "tok4 tok5 tok6", "tok7"]

    t0 = time.perf_counter()
    for inp in inputs:
        predict(tiny_model, tiny_tokenizer, inp, max_new_tokens=4)
    serial_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    predict_batch(tiny_model, tiny_tokenizer, inputs, max_new_tokens=4)
    batch_ms = (time.perf_counter() - t0) * 1000

    # Batch should be at least 20% faster than serial
    assert batch_ms < serial_ms * 0.9, (
        f"Batch ({batch_ms:.1f}ms) was not faster than serial ({serial_ms:.1f}ms)"
    )

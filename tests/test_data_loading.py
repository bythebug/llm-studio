"""
Tests for data loading, validation, splitting, and preprocessing.
"""
from __future__ import annotations

import io
import json

import pytest

from llm_studio.data_loader import (
    DataSplit,
    clean_data,
    load_from_csv,
    load_from_json,
    split_data,
    validate_training_data,
)
from llm_studio.preprocessor import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    PreprocessingPipeline,
    normalize,
    pad_sequences,
    tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PAIRS = [(f"input {i}", f"output {i}") for i in range(20)]

CSV_CONTENT = "input_text,expected_output\nhello world,hi there\nfoo bar,baz qux\n"
JSON_CONTENT = json.dumps([
    {"input": "hello world", "output": "hi there"},
    {"input": "foo bar", "output": "baz qux"},
])


# ---------------------------------------------------------------------------
# test_load_csv
# ---------------------------------------------------------------------------

def test_load_csv_returns_pairs():
    pairs = load_from_csv(io.StringIO(CSV_CONTENT))
    assert pairs == [("hello world", "hi there"), ("foo bar", "baz qux")]


def test_load_csv_missing_column_raises():
    bad_csv = "text,label\nhello,world\n"
    with pytest.raises(ValueError, match="missing required columns"):
        load_from_csv(io.StringIO(bad_csv))


def test_load_csv_drops_blank_rows():
    csv = "input_text,expected_output\nhello,world\n,,\ngoodbye,moon\n"
    pairs = load_from_csv(io.StringIO(csv))
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# test_load_json
# ---------------------------------------------------------------------------

def test_load_json_returns_pairs():
    pairs = load_from_json(io.StringIO(JSON_CONTENT))
    assert pairs == [("hello world", "hi there"), ("foo bar", "baz qux")]


def test_load_json_missing_key_raises():
    bad = json.dumps([{"input": "hello"}])
    with pytest.raises(ValueError, match="missing 'input' or 'output'"):
        load_from_json(io.StringIO(bad))


def test_load_json_non_list_raises():
    with pytest.raises(ValueError, match="top-level array"):
        load_from_json(io.StringIO('{"input": "a", "output": "b"}'))


# ---------------------------------------------------------------------------
# test_validation
# ---------------------------------------------------------------------------

def test_validate_passes_for_valid_data():
    errors = validate_training_data(VALID_PAIRS)
    assert errors == []


def test_validate_empty_dataset():
    errors = validate_training_data([])
    assert any("empty" in e.lower() for e in errors)


def test_validate_too_few_samples():
    errors = validate_training_data([("a", "b")] * 5)
    assert any("few" in e.lower() or "minimum" in e.lower() for e in errors)


def test_validate_blank_input():
    pairs = VALID_PAIRS[:] + [("  ", "some output")]
    errors = validate_training_data(pairs)
    assert any("blank" in e for e in errors)


def test_validate_wrong_type():
    errors = validate_training_data(["not a tuple"])  # type: ignore[list-item]
    assert errors


# ---------------------------------------------------------------------------
# test_train_val_test_split
# ---------------------------------------------------------------------------

def test_split_sizes():
    split = split_data(VALID_PAIRS)
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(VALID_PAIRS)


def test_split_ratios_approx():
    data = [(str(i), str(i)) for i in range(100)]
    split = split_data(data, train_ratio=0.8, val_ratio=0.1)
    assert len(split.train) == 80
    assert len(split.val) == 10
    assert len(split.test) == 10


def test_split_is_reproducible():
    s1 = split_data(VALID_PAIRS, seed=42)
    s2 = split_data(VALID_PAIRS, seed=42)
    assert s1.train == s2.train


def test_split_no_overlap():
    split = split_data(VALID_PAIRS)
    all_sets = split.train + split.val + split.test
    assert len(all_sets) == len(set(all_sets))


# ---------------------------------------------------------------------------
# test_tokenization
# ---------------------------------------------------------------------------

def test_tokenize_returns_list_of_strings():
    tokens = tokenize("hello world")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)


def test_tokenize_splits_words():
    tokens = tokenize("the quick brown fox")
    assert "quick" in tokens


def test_normalize_lowercase():
    assert normalize("Hello WORLD", lowercase=True) == "hello world"


def test_normalize_removes_special_chars():
    result = normalize("hello, world!", remove_special_chars=True)
    assert "," not in result and "!" not in result


def test_normalize_preserves_when_disabled():
    result = normalize("Hello, World!", lowercase=False, remove_special_chars=False)
    assert "," in result and "!" in result


# ---------------------------------------------------------------------------
# test_preprocessing_pipeline
# ---------------------------------------------------------------------------

def test_pipeline_fit_transform_shape():
    texts = ["hello world", "foo bar baz"]
    pipeline = PreprocessingPipeline()
    encoded = pipeline.fit_transform(texts)
    assert len(encoded) == 2
    assert all(len(seq) == pipeline.config.max_length for seq in encoded)


def test_pipeline_adds_bos_eos():
    pipeline = PreprocessingPipeline()
    pipeline.fit(["hello world"])
    encoded = pipeline.transform(["hello world"])
    assert encoded[0][0] == BOS_ID
    last_non_pad = next(x for x in reversed(encoded[0]) if x != PAD_ID)
    assert last_non_pad == EOS_ID


def test_pipeline_pads_to_max_length():
    from llm_studio.preprocessor import PreprocessorConfig
    pipeline = PreprocessingPipeline(config=PreprocessorConfig(max_length=10))
    pipeline.fit(["hi"])
    encoded = pipeline.transform(["hi"])
    assert len(encoded[0]) == 10


def test_pipeline_truncates_long_sequences():
    from llm_studio.preprocessor import PreprocessorConfig
    pipeline = PreprocessingPipeline(config=PreprocessorConfig(max_length=5))
    long_text = "one two three four five six seven eight nine ten"
    pipeline.fit([long_text])
    encoded = pipeline.transform([long_text])
    assert len(encoded[0]) == 5


def test_pipeline_raises_without_fit():
    pipeline = PreprocessingPipeline()
    with pytest.raises(RuntimeError, match="fit"):
        pipeline.transform(["hello"])


def test_pad_sequences_pads_short():
    seqs = [[1, 2], [3, 4, 5, 6]]
    padded = pad_sequences(seqs, max_length=6)
    assert padded[0] == [1, 2, 0, 0, 0, 0]
    assert padded[1] == [3, 4, 5, 6, 0, 0]


def test_pad_sequences_truncates_long():
    seqs = [[1, 2, 3, 4, 5]]
    padded = pad_sequences(seqs, max_length=3)
    assert padded[0] == [1, 2, 3]

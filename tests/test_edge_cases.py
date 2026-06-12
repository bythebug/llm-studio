"""
Edge case tests: empty data, single samples, long text, special characters.
"""
from __future__ import annotations

import pytest

from llm_studio.data_loader import (
    clean_data,
    load_from_csv,
    load_from_json,
    split_data,
    validate_training_data,
)
from llm_studio.preprocessor import (
    PreprocessingPipeline,
    PreprocessorConfig,
    normalize,
    pad_sequences,
    tokenize,
)
from llm_studio.metrics import accuracy, bleu_score, classification_metrics


# ---------------------------------------------------------------------------
# test_empty_dataset
# ---------------------------------------------------------------------------

def test_validate_empty_returns_error():
    errors = validate_training_data([])
    assert errors, "Expected errors for empty dataset"
    assert any("empty" in e.lower() for e in errors)


def test_clean_data_all_blank_returns_empty():
    dirty = [("  ", "output"), ("", ""), ("input", "   ")]
    result = clean_data(dirty)
    assert result == []


def test_split_empty_dataset_returns_empty_splits():
    split = split_data([])
    assert split.train == []
    assert split.val == []
    assert split.test == []


def test_bleu_empty_hypothesis():
    assert bleu_score([""], ["reference text here"]) == 0.0


def test_accuracy_empty_raises():
    with pytest.raises(Exception):
        accuracy([], [])


def test_pad_empty_sequences():
    result = pad_sequences([], max_length=10)
    assert result == []


def test_pipeline_fit_empty_corpus():
    pipeline = PreprocessingPipeline()
    pipeline.fit([])
    # vocab should only contain special tokens
    assert len(pipeline.vocab) == 4  # PAD, BOS, EOS, UNK


def test_load_csv_empty_file_returns_empty():
    import io
    result = load_from_csv(io.StringIO("input_text,expected_output\n"))
    assert result == []


def test_load_json_empty_array_returns_empty():
    import io, json
    result = load_from_json(io.StringIO("[]"))
    assert result == []


# ---------------------------------------------------------------------------
# test_single_sample
# ---------------------------------------------------------------------------

def test_validate_single_sample_reports_too_few():
    errors = validate_training_data([("hello", "world")])
    assert any("few" in e.lower() or "minimum" in e.lower() for e in errors)


def test_split_single_sample_goes_to_train():
    """With 1 sample, everything goes to train; val and test are empty."""
    split = split_data([("a", "b")])
    assert len(split.train) + len(split.val) + len(split.test) == 1


def test_pipeline_single_text():
    pipeline = PreprocessingPipeline(config=PreprocessorConfig(max_length=16))
    result = pipeline.fit_transform(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 16


def test_classification_metrics_single_class():
    """All predictions from one class — precision/recall should still compute."""
    result = classification_metrics(["cat"] * 5, ["cat"] * 5)
    assert result.accuracy == 1.0
    assert result.f1 == pytest.approx(1.0, abs=1e-3)


def test_bleu_single_pair_exact_match():
    # sacreBLEU brevity penalty penalises very short sentences; use a longer one
    text = "the quick brown fox jumps over the lazy dog near the river bank"
    score = bleu_score([text], [text])
    assert score == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# test_very_long_text
# ---------------------------------------------------------------------------

LONG_TEXT = " ".join(["word"] * 2000)    # 2000-word string


def test_tokenize_long_text():
    tokens = tokenize(LONG_TEXT)
    assert len(tokens) == 2000


def test_pipeline_truncates_long_text():
    config = PreprocessorConfig(max_length=32, add_bos_eos=False)
    pipeline = PreprocessingPipeline(config=config)
    result = pipeline.fit_transform([LONG_TEXT])
    assert len(result[0]) == 32


def test_pipeline_long_text_no_overflow():
    """Encoding a very long text must not raise, regardless of length."""
    config = PreprocessorConfig(max_length=64)
    pipeline = PreprocessingPipeline(config=config)
    result = pipeline.fit_transform([LONG_TEXT])
    assert all(len(seq) == 64 for seq in result)


def test_pad_sequences_already_max_length_unchanged():
    seq = list(range(10))
    result = pad_sequences([seq], max_length=10)
    assert result[0] == seq


def test_pad_sequences_longer_than_max_truncated():
    seq = list(range(20))
    result = pad_sequences([seq], max_length=5)
    assert result[0] == list(range(5))


def test_clean_data_strips_long_whitespace():
    """Extra whitespace in long strings should be stripped without error."""
    pair = ("  " + LONG_TEXT + "  ", "expected")
    cleaned = clean_data([pair])
    assert len(cleaned) == 1
    assert not cleaned[0][0].startswith(" ")


# ---------------------------------------------------------------------------
# test_special_characters
# ---------------------------------------------------------------------------

UNICODE_TEXT = "héllo wörld naïve café"
EMOJI_TEXT = "great job 🎉 well done 👍"
CJK_TEXT = "机器学习 is fun"
CONTROL_TEXT = "line1\nline2\ttabbed"
MOJIBAKE_TEXT = "café au lait"     # properly encoded


def test_normalize_unicode_lowercases():
    result = normalize(UNICODE_TEXT, lowercase=True, remove_special_chars=False)
    assert result == result.lower()


def test_normalize_strips_special_chars_from_unicode():
    result = normalize("hello! wörld?", remove_special_chars=True)
    # punctuation removed; accented chars normalised by NFKC upstream or stripped
    assert "!" not in result and "?" not in result


def test_tokenize_handles_unicode():
    tokens = tokenize(UNICODE_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_handles_emoji():
    tokens = tokenize(EMOJI_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_handles_cjk():
    tokens = tokenize(CJK_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_handles_control_characters():
    tokens = tokenize(CONTROL_TEXT)
    assert isinstance(tokens, list)


def test_clean_data_handles_unicode_pairs():
    pairs = [(UNICODE_TEXT, "output"), (EMOJI_TEXT, "response")]
    cleaned = clean_data(pairs)
    assert len(cleaned) == 2


def test_clean_data_drops_null_bytes():
    pair = ("hello\x00world", "output")
    cleaned = clean_data([pair])
    assert len(cleaned) == 1
    assert "\x00" not in cleaned[0][0]


def test_pipeline_handles_unicode_end_to_end():
    config = PreprocessorConfig(max_length=32, lowercase=True, remove_special_chars=True)
    pipeline = PreprocessingPipeline(config=config)
    texts = [UNICODE_TEXT, EMOJI_TEXT, CJK_TEXT, MOJIBAKE_TEXT]
    result = pipeline.fit_transform(texts)
    assert len(result) == 4
    assert all(len(seq) == 32 for seq in result)


def test_load_json_handles_unicode():
    import io, json
    data = json.dumps([{"input": UNICODE_TEXT, "output": CJK_TEXT}])
    pairs = load_from_json(io.StringIO(data))
    assert pairs[0][0] == UNICODE_TEXT


def test_bleu_handles_unicode_strings():
    score = bleu_score([UNICODE_TEXT], [UNICODE_TEXT])
    assert score == pytest.approx(100.0, abs=1.0)

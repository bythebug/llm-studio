"""
Text preprocessing: normalization, tokenization, padding, special tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk

# Ensure punkt tokenizer data is available.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    lowercase: bool = True
    remove_special_chars: bool = True
    add_bos_eos: bool = True
    max_length: int = 512


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def normalize(text: str, lowercase: bool = True, remove_special_chars: bool = True) -> str:
    """Lowercase and/or strip non-alphanumeric characters (configurable)."""
    if lowercase:
        text = text.lower()
    if remove_special_chars:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into word tokens using NLTK's punkt tokenizer."""
    return nltk.word_tokenize(text)


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Map tokens to vocabulary IDs; unknown tokens map to UNK_ID."""
    return [vocab.get(tok, UNK_ID) for tok in tokens]


def pad_sequences(
    sequences: list[list[int]],
    max_length: int,
    pad_id: int = PAD_ID,
) -> list[list[int]]:
    """Truncate to max_length and right-pad shorter sequences with pad_id."""
    padded = []
    for seq in sequences:
        seq = seq[:max_length]
        seq = seq + [pad_id] * (max_length - len(seq))
        padded.append(seq)
    return padded


def build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    """Build a vocabulary dict from a list of token sequences."""
    vocab: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    next_id = len(SPECIAL_TOKENS)
    for tokens in token_lists:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
    return vocab


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingPipeline:
    """End-to-end text preprocessing: normalize → tokenize → encode → pad."""

    config: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    vocab: dict[str, int] = field(default_factory=dict)

    def fit(self, texts: list[str]) -> "PreprocessingPipeline":
        """Build vocabulary from a corpus of texts."""
        token_lists = [self._tokenize(t) for t in texts]
        self.vocab = build_vocab(token_lists)
        return self

    def transform(self, texts: list[str]) -> list[list[int]]:
        """Normalize, tokenize, add special tokens, encode, and pad."""
        if not self.vocab:
            raise RuntimeError("Call fit() before transform()")
        sequences = [self._encode(t) for t in texts]
        return pad_sequences(sequences, max_length=self.config.max_length)

    def fit_transform(self, texts: list[str]) -> list[list[int]]:
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        normalized = normalize(
            text,
            lowercase=self.config.lowercase,
            remove_special_chars=self.config.remove_special_chars,
        )
        return tokenize(normalized)

    def _encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        ids = tokens_to_ids(tokens, self.vocab)
        if self.config.add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

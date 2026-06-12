"""
Training data ingestion: load from CSV/JSON, validate, clean, and split.
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import IO

import pandas as pd
from sqlalchemy.orm import Session

from llm_studio.models import TrainingData


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    train: list[tuple[str, str]]
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_from_csv(source: str | IO) -> list[tuple[str, str]]:
    """Parse a CSV with columns ``input_text`` and ``expected_output``."""
    df = pd.read_csv(source, skip_blank_lines=True, on_bad_lines="skip")
    _assert_columns(df)
    return _df_to_pairs(df)


def load_from_json(source: str | IO) -> list[tuple[str, str]]:
    """Parse a JSON array of ``{input, output}`` objects."""
    if hasattr(source, "read"):
        records = json.load(source)
    else:
        with open(source) as f:
            records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("JSON must be a top-level array")

    pairs = []
    for i, rec in enumerate(records):
        if "input" not in rec or "output" not in rec:
            raise ValueError(f"Record {i} missing 'input' or 'output' key")
        pairs.append((rec["input"], rec["output"]))
    return pairs


def load_training_data(job_id: int, session: Session) -> list[tuple[str, str]]:
    """Load all training pairs for a job from the database."""
    rows = (
        session.query(TrainingData)
        .filter(TrainingData.job_id == job_id)
        .order_by(TrainingData.id)
        .all()
    )
    return [(row.input, row.expected_output) for row in rows]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_training_data(data: list[tuple[str, str]]) -> list[str]:
    """
    Check format and types. Returns a list of error messages;
    empty list means the data is valid.
    """
    errors: list[str] = []

    if not data:
        errors.append("Dataset is empty")
        return errors

    if len(data) < 10:
        errors.append(f"Too few samples ({len(data)}); minimum is 10")

    for i, pair in enumerate(data):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            errors.append(f"Row {i}: expected (input, output) pair, got {type(pair)}")
            continue
        inp, out = pair
        if not isinstance(inp, str) or not isinstance(out, str):
            errors.append(f"Row {i}: input and output must be strings")
        elif not inp.strip():
            errors.append(f"Row {i}: input is blank")
        elif not out.strip():
            errors.append(f"Row {i}: output is blank")

    return errors


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(data: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove blank rows and fix encoding issues."""
    cleaned = []
    for inp, out in data:
        inp = _fix_encoding(inp).strip()
        out = _fix_encoding(out).strip()
        if inp and out:
            cleaned.append((inp, out))
    return cleaned


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(
    data: list[tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DataSplit:
    """Shuffle and split into train / val / test sets."""
    if abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) > 1e-9:
        pass  # always sums to 1 by construction

    df = pd.DataFrame(data, columns=["input", "output"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    def to_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
        return list(zip(frame["input"], frame["output"]))

    return DataSplit(
        train=to_pairs(df.iloc[:train_end]),
        val=to_pairs(df.iloc[train_end:val_end]),
        test=to_pairs(df.iloc[val_end:]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_columns(df: pd.DataFrame) -> None:
    required = {"input_text", "expected_output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _df_to_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    df = df.dropna(subset=["input_text", "expected_output"])
    df["input_text"] = df["input_text"].astype(str)
    df["expected_output"] = df["expected_output"].astype(str)
    return list(zip(df["input_text"], df["expected_output"]))


def _fix_encoding(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = normalized.encode("utf-8", errors="ignore").decode("utf-8")
    return cleaned.replace("\x00", "")   # strip null bytes — invalid in most text contexts

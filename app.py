"""
REST API for training data management.
"""
from __future__ import annotations

import io
import json
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from llm_studio.config import DATABASE_URL
from llm_studio.data_loader import (
    clean_data,
    load_from_csv,
    load_from_json,
    load_training_data,
    split_data,
    validate_training_data,
)
from llm_studio.models import FineTuningJob, TrainingData, create_all, get_engine
from llm_studio.preprocessor import PreprocessingPipeline, tokenize, normalize

app = FastAPI(title="LLM Studio", version="0.1.0")

engine = get_engine(DATABASE_URL)
create_all(engine)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_session():
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_job_or_404(job_id: int, session: Session) -> FineTuningJob:
    job = session.get(FineTuningJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def _persist_pairs(job_id: int, pairs: list[tuple[str, str]], session: Session) -> int:
    rows = [
        TrainingData(job_id=job_id, input=inp, expected_output=out)
        for inp, out in pairs
    ]
    session.add_all(rows)
    session.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/upload_data", summary="Upload training data (CSV or JSON)")
async def upload_data(
    job_id: int,
    file: Annotated[UploadFile, File(description="CSV or JSON file")],
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)

    content = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            pairs = load_from_csv(io.StringIO(content.decode("utf-8")))
        elif filename.endswith(".json"):
            pairs = load_from_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .json files are supported")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    pairs = clean_data(pairs)
    errors = validate_training_data(pairs)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    count = _persist_pairs(job_id, pairs, session)
    return {"job_id": job_id, "rows_added": count}


@app.get("/jobs/{job_id}/data_preview", summary="Preview first rows of training data")
def data_preview(
    job_id: int,
    rows: int = 5,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    pairs = load_training_data(job_id, session)
    preview = [{"input": inp, "output": out} for inp, out in pairs[:rows]]
    return {"job_id": job_id, "total_rows": len(pairs), "preview": preview}


@app.get("/jobs/{job_id}/data_stats", summary="Statistics for a job's training data")
def data_stats(
    job_id: int,
    session: Session = Depends(get_session),
):
    _get_job_or_404(job_id, session)
    pairs = load_training_data(job_id, session)

    if not pairs:
        return {"job_id": job_id, "num_rows": 0}

    inputs = [inp for inp, _ in pairs]
    outputs = [out for _, out in pairs]

    all_tokens = [tok for text in inputs + outputs for tok in tokenize(normalize(text))]
    vocab_size = len(set(all_tokens))

    split = split_data(pairs)

    return {
        "job_id": job_id,
        "num_rows": len(pairs),
        "avg_input_length": round(sum(len(t) for t in inputs) / len(inputs), 1),
        "avg_output_length": round(sum(len(t) for t in outputs) / len(outputs), 1),
        "vocab_size": vocab_size,
        "split": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
    }

# LLM Studio

A backend system for managing LLM fine-tuning jobs — data ingestion, job lifecycle tracking, model versioning, and prediction logging.

## Project Structure

```
llm-studio/
├── llm_studio/
│   ├── __init__.py
│   ├── models.py       # SQLAlchemy ORM models
│   └── config.py       # Base models, hyperparameters, storage paths
├── db/
│   └── schema.sql      # PostgreSQL DDL (tables, indexes, triggers)
├── notes/
│   └── study_notes.txt # Design notes on job tracking, versioning, data org
├── requirements.txt
└── .gitignore
```

## Data Model

```
User
 └── FineTuningJob (status: queued → training → completed | failed)
      ├── TrainingData   (input / expected_output pairs)
      ├── ModelVersion   (versioned checkpoints with accuracy + loss)
      └── Prediction     (inference results for evaluation)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Apply the schema to a PostgreSQL database:

```bash
psql $DATABASE_URL < db/schema.sql
```

Set the database URL:

```bash
export DATABASE_URL=postgresql://localhost/llm_studio
```

## Configuration

`llm_studio/config.py` exposes:

- **`BASE_MODELS`** — registry of supported base models (GPT-2, LLaMA 3, Mistral, T5)
- **`TrainingConfig`** — dataclass for learning rate, epochs, batch size, etc.
- **`StorageConfig`** — path helpers for model weights and checkpoints

## Status

Foundation layer complete (models, schema, config). Training pipeline, REST API, and worker queue coming next.

-- Job status enum
CREATE TYPE job_status AS ENUM ('queued', 'training', 'completed', 'failed');

-- Users
CREATE TABLE users (
    id         SERIAL PRIMARY KEY,
    email      VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Fine-tuning jobs (top-level training task)
CREATE TABLE fine_tuning_jobs (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    status     job_status   NOT NULL DEFAULT 'queued',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_fine_tuning_jobs_user_id ON fine_tuning_jobs(user_id);
CREATE INDEX idx_fine_tuning_jobs_status  ON fine_tuning_jobs(status);

-- Training data (input-output pairs for a job)
CREATE TABLE training_data (
    id              SERIAL PRIMARY KEY,
    job_id          INTEGER     NOT NULL REFERENCES fine_tuning_jobs(id) ON DELETE CASCADE,
    input           TEXT        NOT NULL,
    expected_output TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_training_data_job_id ON training_data(job_id);

-- Model versions (one per completed training run)
CREATE TABLE model_versions (
    id          SERIAL PRIMARY KEY,
    job_id      INTEGER      NOT NULL REFERENCES fine_tuning_jobs(id) ON DELETE CASCADE,
    version_num INTEGER      NOT NULL,
    model_path  VARCHAR(500) NOT NULL,
    accuracy    FLOAT,
    loss        FLOAT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_job_version UNIQUE (job_id, version_num)
);

CREATE INDEX idx_model_versions_job_id ON model_versions(job_id);

-- Predictions (test runs against a versioned model)
CREATE TABLE predictions (
    id        SERIAL PRIMARY KEY,
    job_id    INTEGER     NOT NULL REFERENCES fine_tuning_jobs(id) ON DELETE CASCADE,
    input     TEXT        NOT NULL,
    output    TEXT        NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_predictions_job_id ON predictions(job_id);

-- Compute instances (remote SSH GPU machines)
CREATE TYPE compute_status AS ENUM ('unknown', 'connected', 'error');

CREATE TABLE compute_instances (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    host         VARCHAR(255) NOT NULL,
    port         INTEGER      NOT NULL DEFAULT 22,
    username     VARCHAR(100) NOT NULL,
    key_path     VARCHAR(500),
    last_status  compute_status NOT NULL DEFAULT 'unknown',
    last_checked TIMESTAMPTZ,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at on fine_tuning_jobs
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fine_tuning_jobs_updated_at
BEFORE UPDATE ON fine_tuning_jobs
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

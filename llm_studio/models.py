import enum
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Float, DateTime, ForeignKey, Enum
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(enum.Enum):
    queued = "queued"
    training = "training"
    completed = "completed"
    failed = "failed"


class ComputeStatus(enum.Enum):
    unknown = "unknown"
    connected = "connected"
    error = "error"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    jobs = relationship("FineTuningJob", back_populates="user")


class FineTuningJob(Base):
    __tablename__ = "fine_tuning_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.queued, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="jobs")
    training_data = relationship("TrainingData", back_populates="job")
    model_versions = relationship("ModelVersion", back_populates="job")
    predictions = relationship("Prediction", back_populates="job")


class TrainingData(Base):
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("fine_tuning_jobs.id"), nullable=False)
    input = Column(Text, nullable=False)
    expected_output = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    job = relationship("FineTuningJob", back_populates="training_data")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("fine_tuning_jobs.id"), nullable=False)
    version_num = Column(Integer, nullable=False)
    model_path = Column(String(500), nullable=False)
    accuracy = Column(Float)
    loss = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    job = relationship("FineTuningJob", back_populates="model_versions")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("fine_tuning_jobs.id"), nullable=False)
    input = Column(Text, nullable=False)
    output = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    job = relationship("FineTuningJob", back_populates="predictions")


class ComputeInstance(Base):
    __tablename__ = "compute_instances"

    id          = Column(Integer, primary_key=True)
    name        = Column(String(100), nullable=False)
    host        = Column(String(255), nullable=False)
    port        = Column(Integer, default=22, nullable=False)
    username    = Column(String(100), nullable=False)
    key_path    = Column(String(500), nullable=True)   # path to SSH private key on the server
    last_status = Column(Enum(ComputeStatus), default=ComputeStatus.unknown, nullable=False)
    last_checked = Column(DateTime, nullable=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


def get_engine(database_url: str):
    return create_engine(database_url, echo=False)


def create_all(engine):
    Base.metadata.create_all(engine)

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel, Relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Session(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    metadata_: dict | None = Field(default=None, sa_column=Column("metadata", JSON))
    tags: str | None = Field(default=None)

    messages: list["Message"] = Relationship(back_populates="session")


class Message(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    session_id: str = Field(foreign_key="session.id", index=True)
    created_at: datetime = Field(default_factory=_now)
    role: str
    content: str
    model: str | None = Field(default=None)
    prompt_tokens: int | None = Field(default=None)
    completion_tokens: int | None = Field(default=None)
    raw_request: dict | None = Field(default=None, sa_column=Column(JSON))
    raw_response: dict | None = Field(default=None, sa_column=Column(JSON))

    session: Session | None = Relationship(back_populates="messages")
    judgments: list["Judgment"] = Relationship(back_populates="message")


class Judgment(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    message_id: str = Field(foreign_key="message.id", index=True)
    created_at: datetime = Field(default_factory=_now)
    score: float | None = Field(default=None)
    labels: dict | None = Field(default=None, sa_column=Column(JSON))
    rewrite: str | None = Field(default=None)
    rationale: str | None = Field(default=None)
    judge_model: str | None = Field(default=None)

    message: Message | None = Relationship(back_populates="judgments")


class TrainingRun(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    method: str
    config: dict | None = Field(default=None, sa_column=Column(JSON))
    base_model: str | None = Field(default=None)
    adapter_path: str | None = Field(default=None)
    metrics: dict | None = Field(default=None, sa_column=Column(JSON))
    status: str = Field(default="pending")

    checkpoints: list["Checkpoint"] = Relationship(back_populates="training_run")


class Checkpoint(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    training_run_id: str = Field(foreign_key="trainingrun.id", index=True)
    created_at: datetime = Field(default_factory=_now)
    step: int
    adapter_path: str | None = Field(default=None)
    eval_metrics: dict | None = Field(default=None, sa_column=Column(JSON))

    training_run: TrainingRun | None = Relationship(back_populates="checkpoints")

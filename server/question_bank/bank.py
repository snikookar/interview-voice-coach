"""Question bank schema and YAML loader."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

BANK_PATH = Path(__file__).with_name("questions.yaml")

Role = Literal["ai-engineer", "backend"]
Level = Literal["junior", "mid", "senior"]
ROLES: tuple[str, ...] = ("ai-engineer", "backend")
LEVELS: tuple[str, ...] = ("junior", "mid", "senior")


class Question(BaseModel):
    id: str
    role: list[Role] = Field(min_length=1)
    level: list[Level] = Field(min_length=1)
    topic: str
    type: Literal["technical", "behavioral"] = "technical"
    question: str
    key_points: list[str] = Field(min_length=2, max_length=6)
    follow_ups: list[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5)
    source: Literal["bank", "generated"] = "bank"

    @field_validator("question")
    @classmethod
    def _is_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 15:
            raise ValueError("question text too short")
        return v

    def embedding_text(self) -> str:
        # Key points are included so a job posting that mentions "nDCG" or
        # "PgBouncer" matches the question that tests it.
        return f"{self.topic}: {self.question} Key points: {'; '.join(self.key_points)}"

    def content_hash(self, embedding_model: str) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True) + embedding_model
        return hashlib.sha256(payload.encode()).hexdigest()


def load_bank(path: Path = BANK_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [Question.model_validate(item) for item in raw]
    ids = [q.id for q in questions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids: {sorted(dupes)}")
    return questions


@lru_cache
def bank_by_id() -> dict[str, Question]:
    return {q.id: q for q in load_bank()}

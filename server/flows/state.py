"""In-memory state of one live interview, shared by the flow, its tools and the recorder."""

import time
from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["intro", "question", "follow_up", "wrap_up"]

# Leave this much time for the last answer and the goodbye.
WRAP_UP_RESERVE_SECS = 90


@dataclass
class AnswerNote:
    """What the live interviewer noticed. The post-session judge re-scores carefully."""

    phase: Phase
    covered_points: list[str]
    missing_points: list[str]
    follow_up_question: str = ""


@dataclass
class InterviewState:
    session_id: str
    role: str
    level: str
    plan: list[dict]
    max_minutes: int = 12
    mode: str = "technical"
    started_at: float = field(default_factory=time.monotonic)
    index: int = -1
    phase: Phase = "intro"
    follow_up_asked: bool = False
    ended: bool = False
    end_reason: str = ""
    notes: dict[str, list[AnswerNote]] = field(default_factory=dict)

    @property
    def current(self) -> dict | None:
        return self.plan[self.index] if 0 <= self.index < len(self.plan) else None

    @property
    def current_id(self) -> str | None:
        q = self.current
        return q["id"] if q else None

    @property
    def elapsed_secs(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def out_of_time(self) -> bool:
        return self.elapsed_secs > self.max_minutes * 60 - WRAP_UP_RESERVE_SECS

    def has_next(self) -> bool:
        return self.index + 1 < len(self.plan) and not self.out_of_time

    def advance(self) -> dict:
        self.index += 1
        self.phase = "question"
        self.follow_up_asked = False
        return self.plan[self.index]

    def add_note(self, note: AnswerNote) -> None:
        if self.current_id:
            self.notes.setdefault(self.current_id, []).append(note)

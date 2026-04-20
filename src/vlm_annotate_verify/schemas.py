"""Shared dataclasses for proposals, verified annotations, and review log."""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Literal


@dataclass(frozen=True)
class Boundary:
    t_s: float
    label: str
    confidence: float


@dataclass(frozen=True)
class Mistake:
    type: Literal["drop", "slip", "miss", "collision", "other"]
    t_s: float
    note: str
    confidence: float


@dataclass(frozen=True)
class Segment:
    idx: int
    start_s: float
    end_s: float
    quality: int
    quality_conf: float
    success: bool
    success_conf: float
    mistakes: list[Mistake] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Proposal:
    ep_id: str
    video_path: str
    duration_s: float
    frame_paths: list[str]
    task: str
    model: str
    boundaries: list[Boundary]
    segments: list[Segment]
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "Proposal":
        d = json.loads(line)
        d["boundaries"] = [Boundary(**b) for b in d["boundaries"]]
        d["segments"] = [
            Segment(**{**s, "mistakes": [Mistake(**m) for m in s.get("mistakes", [])]})
            for s in d["segments"]
        ]
        return cls(**d)


@dataclass
class Review:
    reviewer_id: str
    review_seconds: float
    actions: list[str]
    reprompt_used: bool


@dataclass
class Verified:
    ep_id: str
    task: str
    boundaries: list[Boundary]
    segments: list[Segment]
    review: Review
    verified_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "Verified":
        d = json.loads(line)
        d["boundaries"] = [Boundary(**b) for b in d["boundaries"]]
        d["segments"] = [
            Segment(**{**s, "mistakes": [Mistake(**m) for m in s.get("mistakes", [])]})
            for s in d["segments"]
        ]
        d["review"] = Review(**d["review"])
        return cls(**d)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FailureRow:
    ep_id: str
    error: str
    attempted_at: str
    status: str = "failed"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

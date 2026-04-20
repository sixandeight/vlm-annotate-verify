"""Confidence band to default cursor and visible-panel policy."""
from dataclasses import dataclass
from enum import Enum

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.50

BOUNDARY_AUTOACCEPT_THRESHOLD = 0.85


class Band(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CursorTarget(str, Enum):
    SPACE = "SPACE"
    EDIT = "EDIT"
    REPLAY = "REPLAY"


@dataclass(frozen=True)
class Defaults:
    band: Band
    cursor: CursorTarget
    show_reasoning: bool
    show_frames: bool
    auto_enter_scrub: bool


def classify(confidence: float) -> Band:
    if confidence > HIGH_THRESHOLD:
        return Band.HIGH
    if confidence >= LOW_THRESHOLD:
        return Band.MEDIUM
    return Band.LOW


def defaults_for(confidence: float) -> Defaults:
    band = classify(confidence)
    if band is Band.HIGH:
        return Defaults(band, CursorTarget.SPACE, False, False, False)
    if band is Band.MEDIUM:
        return Defaults(band, CursorTarget.EDIT, True, True, False)
    return Defaults(band, CursorTarget.REPLAY, True, True, True)


def boundary_visible(boundary_confidence: float) -> bool:
    """Return True when this boundary should be surfaced for review."""
    return boundary_confidence <= BOUNDARY_AUTOACCEPT_THRESHOLD

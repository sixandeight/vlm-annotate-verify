import json
import pytest
from dataclasses import FrozenInstanceError

from vlm_annotate_verify.schemas import (
    Boundary, Mistake, Segment, Proposal, Review, Verified, FailureRow,
    utc_now_iso,
)


def _sample_proposal_dict() -> dict:
    return {
        "ep_id": "ep_000",
        "video_path": "episodes/ep_000.mp4",
        "duration_s": 12.4,
        "frame_paths": ["frames/ep_000/01.jpg", "frames/ep_000/02.jpg"],
        "task": "place cube on shelf",
        "model": "gemini-2.5-flash",
        "boundaries": [
            {"t_s": 5.2, "label": "grasp", "confidence": 0.91},
        ],
        "segments": [
            {
                "idx": 0,
                "start_s": 0.0, "end_s": 5.2,
                "quality": 4, "quality_conf": 0.82,
                "success": True, "success_conf": 0.95,
                "mistakes": [
                    {"type": "slip", "t_s": 3.2, "note": "cube wobble", "confidence": 0.65},
                ],
                "reasoning": "smooth grasp",
            }
        ],
        "created_at": "2026-04-20T15:30:00Z",
    }


def test_boundary_frozen():
    b = Boundary(t_s=1.0, label="grasp", confidence=0.9)
    with pytest.raises(FrozenInstanceError):
        b.t_s = 2.0


def test_mistake_type_values():
    for t in ("drop", "slip", "miss", "collision", "other"):
        Mistake(type=t, t_s=1.0, note="x", confidence=0.5)


def test_proposal_round_trip():
    d = _sample_proposal_dict()
    line = json.dumps(d)
    prop = Proposal.from_json(line)
    assert prop.ep_id == "ep_000"
    assert prop.task == "place cube on shelf"
    assert prop.boundaries[0].label == "grasp"
    assert prop.segments[0].mistakes[0].type == "slip"
    # round-trip: to_json then from_json yields equivalent object
    redo = Proposal.from_json(prop.to_json())
    assert redo.ep_id == prop.ep_id
    assert redo.boundaries == prop.boundaries
    assert redo.segments == prop.segments


def test_proposal_missing_field_raises():
    d = _sample_proposal_dict()
    del d["task"]
    with pytest.raises((KeyError, TypeError)):
        Proposal.from_json(json.dumps(d))


def test_segment_default_mistakes_empty():
    s = Segment(idx=0, start_s=0.0, end_s=1.0, quality=3, quality_conf=0.5,
                success=True, success_conf=0.5)
    assert s.mistakes == []
    assert s.reasoning == ""


def test_verified_round_trip():
    v = Verified(
        ep_id="ep_000",
        task="place cube on shelf",
        boundaries=[Boundary(t_s=5.2, label="grasp", confidence=1.0)],
        segments=[Segment(
            idx=0, start_s=0.0, end_s=5.2,
            quality=5, quality_conf=1.0, success=True, success_conf=1.0,
            mistakes=[], reasoning="",
        )],
        review=Review(
            reviewer_id="nathan", review_seconds=3.4,
            actions=["q=5", "accept"], reprompt_used=False,
        ),
        verified_at="2026-04-20T16:30:00Z",
    )
    redo = Verified.from_json(v.to_json())
    assert redo.ep_id == "ep_000"
    assert redo.review.reviewer_id == "nathan"
    assert redo.review.actions == ["q=5", "accept"]
    assert redo.boundaries[0].confidence == 1.0


def test_failure_row_status_default():
    f = FailureRow(ep_id="ep_042", error="ffmpeg crash", attempted_at="2026-04-20T15:00:00Z")
    assert f.status == "failed"
    line = f.to_json()
    parsed = json.loads(line)
    assert parsed["status"] == "failed"
    assert parsed["ep_id"] == "ep_042"


def test_utc_now_iso_format():
    s = utc_now_iso()
    assert "T" in s
    assert s.endswith("+00:00") or s.endswith("Z")

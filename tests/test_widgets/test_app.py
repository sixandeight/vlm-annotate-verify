import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_annotate_verify.proposer.gemini import GeminiConfig
from vlm_annotate_verify.reviewer.app import (
    ReviewerApp, already_verified_ids, load_proposals,
)
from vlm_annotate_verify.schemas import Boundary, Mistake, Proposal, Segment, Verified


def _sample_proposal(ep_id: str = "ep_000") -> Proposal:
    return Proposal(
        ep_id=ep_id,
        video_path=f"episodes/{ep_id}.mp4",
        duration_s=10.0,
        frame_paths=[f"frames/{ep_id}/01.jpg", f"frames/{ep_id}/02.jpg"],
        task="place cube on shelf",
        model="gemini-2.5-flash",
        boundaries=[Boundary(t_s=5.0, label="grasp", confidence=0.9)],
        segments=[Segment(
            idx=0, start_s=0.0, end_s=5.0,
            quality=4, quality_conf=0.8,
            success=True, success_conf=0.9,
            mistakes=[],
            reasoning="smooth",
        )],
        created_at="2026-04-20T15:30:00Z",
    )


def _write_jsonl(path: Path, records: list[str]) -> None:
    path.write_text("\n".join(records) + "\n")


def test_load_proposals_returns_empty_when_file_missing(tmp_path):
    assert load_proposals(tmp_path / "nope.jsonl") == []


def test_load_proposals_parses_rows(tmp_path):
    path = tmp_path / "proposals.jsonl"
    _write_jsonl(path, [_sample_proposal("ep_000").to_json(),
                        _sample_proposal("ep_001").to_json()])
    props = load_proposals(path)
    assert [p.ep_id for p in props] == ["ep_000", "ep_001"]


def test_load_proposals_skips_failure_rows(tmp_path):
    path = tmp_path / "proposals.jsonl"
    failure = '{"ep_id": "ep_042", "status": "failed", "error": "x"}'
    _write_jsonl(path, [_sample_proposal("ep_000").to_json(), failure,
                        _sample_proposal("ep_001").to_json()])
    props = load_proposals(path)
    assert [p.ep_id for p in props] == ["ep_000", "ep_001"]


def test_already_verified_ids_missing_file(tmp_path):
    assert already_verified_ids(tmp_path / "nope.jsonl") == set()


def test_already_verified_ids_parses(tmp_path):
    path = tmp_path / "verified.jsonl"
    _write_jsonl(path, [
        '{"ep_id": "ep_000", "task": "x", "boundaries": [], "segments": [],'
        ' "review": {"reviewer_id": "n", "review_seconds": 1.0, "actions": [], "reprompt_used": false},'
        ' "verified_at": "2026-04-20T10:00:00Z"}',
        '{"ep_id": "ep_001", "task": "x", "boundaries": [], "segments": [],'
        ' "review": {"reviewer_id": "n", "review_seconds": 1.0, "actions": [], "reprompt_used": false},'
        ' "verified_at": "2026-04-20T10:00:00Z"}',
    ])
    assert already_verified_ids(path) == {"ep_000", "ep_001"}


def test_reviewer_app_initial_queue_filters_verified(tmp_path):
    _write_jsonl(tmp_path / "proposals.jsonl", [
        _sample_proposal("ep_000").to_json(),
        _sample_proposal("ep_001").to_json(),
    ])
    _write_jsonl(tmp_path / "verified.jsonl", [
        '{"ep_id": "ep_000", "task": "x", "boundaries": [], "segments": [],'
        ' "review": {"reviewer_id": "n", "review_seconds": 1.0, "actions": [], "reprompt_used": false},'
        ' "verified_at": "2026-04-20T10:00:00Z"}',
    ])
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )
    app._load_queue()
    assert [p.ep_id for p in app.queue] == ["ep_001"]
    assert app.queue_idx == 0


def test_reviewer_app_from_ep_jumps_to_index(tmp_path):
    _write_jsonl(tmp_path / "proposals.jsonl", [
        _sample_proposal("ep_000").to_json(),
        _sample_proposal("ep_001").to_json(),
        _sample_proposal("ep_002").to_json(),
    ])
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
        from_ep="ep_002",
    )
    app._load_queue()
    assert app.queue_idx == 2


def test_reviewer_app_min_ep_confidence_picks_lowest(tmp_path):
    prop = _sample_proposal("ep_000")
    # Rebuild with a mix of confidences
    prop.segments = [
        Segment(idx=0, start_s=0.0, end_s=2.0,
                quality=4, quality_conf=0.9,
                success=True, success_conf=0.95,
                mistakes=[], reasoning=""),
        Segment(idx=1, start_s=2.0, end_s=5.0,
                quality=3, quality_conf=0.40,  # lowest
                success=True, success_conf=0.90,
                mistakes=[], reasoning=""),
    ]
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="n",
    )
    assert app._min_ep_confidence(prop) == 0.40


def test_commit_writes_verified_line(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "proposals.jsonl", [
        _sample_proposal("ep_000").to_json(),
    ])
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )
    app._load_queue()
    # Simulate being in the middle of reviewing ep_000
    app.current_boundaries = app.queue[0].boundaries
    app.current_segments = list(app.queue[0].segments)
    app.actions_log = ["accept"]
    app.reprompt_used = False
    app.start_time = 0.0
    monkeypatch.setattr(
        "vlm_annotate_verify.reviewer.app.time.monotonic",
        lambda: 4.2,
    )

    # Stub has_unfixed_generic_notes and query_one for the panel
    class _FakePanel:
        task = "place cube on shelf"
        def has_unfixed_generic_notes(self):
            return False

    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    # Stub _render_current to prevent UI work after advance
    app._render_current = lambda: None  # type: ignore[assignment]
    app._commit_and_next()

    verified_lines = (tmp_path / "verified.jsonl").read_text().strip().splitlines()
    assert len(verified_lines) == 1
    parsed = json.loads(verified_lines[0])
    assert parsed["ep_id"] == "ep_000"
    assert parsed["task"] == "place cube on shelf"
    assert parsed["review"]["actions"] == ["accept"]
    assert parsed["review"]["review_seconds"] == pytest.approx(4.2)


def test_commit_blocks_on_generic_note(tmp_path):
    _write_jsonl(tmp_path / "proposals.jsonl", [
        _sample_proposal("ep_000").to_json(),
    ])
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )
    app._load_queue()
    app.current_boundaries = app.queue[0].boundaries
    app.current_segments = list(app.queue[0].segments)
    app.actions_log = []
    app.reprompt_used = False
    app.start_time = 0.0

    class _FakePanel:
        task = "x"
        def has_unfixed_generic_notes(self):
            return True

    bell_calls = {"n": 0}
    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    app.bell = lambda: bell_calls.__setitem__("n", bell_calls["n"] + 1)  # type: ignore[assignment]
    app._commit_and_next()

    assert not (tmp_path / "verified.jsonl").exists()
    assert bell_calls["n"] == 1

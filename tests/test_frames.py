import pytest
from pathlib import Path

from vlm_annotate_verify.proposer.frames import (
    DEFAULT_NUM_FRAMES,
    FrameExtractionError,
    extract_frames,
    get_video_duration,
)

FIXTURE_VIDEO = (
    Path(__file__).parent / "fixtures" / "mini_dataset" / "episodes" / "ep_000.mp4"
)


def test_get_video_duration_returns_about_five_seconds():
    duration = get_video_duration(FIXTURE_VIDEO)
    assert 4.5 < duration < 5.5


def test_get_video_duration_missing_video_raises(tmp_path):
    with pytest.raises(FrameExtractionError, match="not found"):
        get_video_duration(tmp_path / "nope.mp4")


def test_extract_frames_writes_expected_count(tmp_path):
    out = tmp_path / "ep_000"
    paths = extract_frames(FIXTURE_VIDEO, out)
    assert len(paths) == DEFAULT_NUM_FRAMES
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"
        assert p.stat().st_size > 0


def test_extract_frames_respects_num_frames(tmp_path):
    out = tmp_path / "ep_000"
    paths = extract_frames(FIXTURE_VIDEO, out, num_frames=4)
    assert len(paths) == 4


def test_extract_frames_is_idempotent(tmp_path):
    out = tmp_path / "ep_000"
    first = extract_frames(FIXTURE_VIDEO, out, num_frames=4)
    first_mtimes = [p.stat().st_mtime_ns for p in first]
    second = extract_frames(FIXTURE_VIDEO, out, num_frames=4)
    second_mtimes = [p.stat().st_mtime_ns for p in second]
    assert first == second
    assert first_mtimes == second_mtimes  # cache hit — no rewrite


def test_extract_frames_missing_video_raises(tmp_path):
    with pytest.raises(FrameExtractionError):
        extract_frames(tmp_path / "nope.mp4", tmp_path / "out")

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


def test_extract_frames_rejects_zero_duration(tmp_path, monkeypatch):
    from vlm_annotate_verify.proposer import frames as frames_mod
    v = tmp_path / "corrupt.mp4"
    v.write_bytes(b"\x00")
    monkeypatch.setattr(frames_mod, "get_video_duration", lambda _p: 0.0)
    with pytest.raises(FrameExtractionError, match="duration"):
        extract_frames(v, tmp_path / "out", num_frames=4)


def test_extract_frames_redoes_zero_byte_cache(tmp_path):
    # Pre-create a zero-byte frame to simulate a crashed previous run
    out = tmp_path / "ep_000"
    out.mkdir()
    for i in range(1, 5):
        (out / f"{i:02d}.jpg").write_bytes(b"")
    # Real extraction should overwrite the zero-byte files with non-empty ones
    paths = extract_frames(FIXTURE_VIDEO, out, num_frames=4)
    for p in paths:
        assert p.stat().st_size > 0

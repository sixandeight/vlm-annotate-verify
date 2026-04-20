from pathlib import Path

import pytest

from vlm_annotate_verify.reviewer.widgets.frame_view import (
    FrameMode, FrameView, REPLAY_FPS,
)
from vlm_annotate_verify.schemas import Boundary


def _dummy_frames(tmp_path: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        p = tmp_path / f"{i:02d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg magic
        out.append(p)
    return out


def test_starts_in_grid_mode(tmp_path):
    fv = FrameView(_dummy_frames(tmp_path, 8), [], duration_s=8.0)
    assert fv.mode is FrameMode.GRID
    assert fv.cursor == 0


def test_step_forward_and_back(tmp_path):
    fv = FrameView(_dummy_frames(tmp_path, 8), [], duration_s=8.0)
    fv.step(3)
    assert fv.cursor == 3
    fv.step(-1)
    assert fv.cursor == 2


def test_step_clamps_to_bounds(tmp_path):
    fv = FrameView(_dummy_frames(tmp_path, 4), [], duration_s=4.0)
    fv.step(100)
    assert fv.cursor == 3  # n - 1
    fv.step(-500)
    assert fv.cursor == 0


def test_step_with_no_frames_is_noop(tmp_path):
    fv = FrameView([], [], duration_s=0.0)
    fv.step(1)  # must not raise
    assert fv.cursor == 0


def test_jump_to_subtask_forward(tmp_path):
    # 10 frames across 10 seconds; boundary at 5s -> frame index ~5
    frames = _dummy_frames(tmp_path, 10)
    boundaries = [Boundary(t_s=5.0, label="grasp", confidence=0.9)]
    fv = FrameView(frames, boundaries, duration_s=10.0)
    fv.cursor = 1  # inside subtask 0
    fv.jump_to_subtask(1)
    assert fv.cursor == 5  # boundary timestamp * 10 / 10


def test_jump_to_subtask_backward(tmp_path):
    frames = _dummy_frames(tmp_path, 10)
    boundaries = [Boundary(t_s=5.0, label="grasp", confidence=0.9)]
    fv = FrameView(frames, boundaries, duration_s=10.0)
    fv.cursor = 7  # inside subtask 1
    fv.jump_to_subtask(-1)
    # prev subtask starts at 0 -> cursor 0
    assert fv.cursor == 0


def test_jump_to_subtask_no_boundaries_is_noop(tmp_path):
    fv = FrameView(_dummy_frames(tmp_path, 8), [], duration_s=8.0)
    fv.cursor = 3
    fv.jump_to_subtask(1)
    assert fv.cursor == 3


def test_mode_transitions(tmp_path):
    fv = FrameView(_dummy_frames(tmp_path, 4), [], duration_s=4.0)
    fv.enter_scrub()
    assert fv.mode is FrameMode.SCRUB
    fv.enter_replay()
    assert fv.mode is FrameMode.REPLAY
    fv.enter_grid()
    assert fv.mode is FrameMode.GRID


def test_replay_fps_is_positive():
    assert REPLAY_FPS > 0

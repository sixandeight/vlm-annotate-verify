import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from vlm_annotate_verify.reviewer.widgets.timeline import (
    Timeline, WIDTH, render_timeline,
)
from vlm_annotate_verify.schemas import Boundary, Mistake, Segment


def _make_seg(idx, start, end, mistakes=None):
    return Segment(
        idx=idx, start_s=start, end_s=end,
        quality=4, quality_conf=0.9, success=True, success_conf=0.9,
        mistakes=mistakes or [],
    )


def test_render_returns_three_rows():
    out = render_timeline(10.0, [], [_make_seg(0, 0.0, 10.0)])
    rows = out.split("\n")
    assert len(rows) == 3


def test_render_bar_spans_full_width():
    out = render_timeline(10.0, [], [_make_seg(0, 0.0, 10.0)])
    bar_row = out.split("\n")[0]
    # Bar is delimited by │ ... │
    assert bar_row.startswith("│")
    assert bar_row.endswith("│")
    assert len(bar_row) == WIDTH + 2


def test_render_zero_duration_returns_safe_placeholder():
    out = render_timeline(0.0, [], [])
    # Must not raise; return something width-ish
    assert len(out) > 0


def test_render_marks_mistakes_with_dot():
    seg = _make_seg(0, 0.0, 10.0,
                    mistakes=[Mistake(type="slip", t_s=5.0, note="x", confidence=0.8)])
    out = render_timeline(10.0, [], [seg])
    mistakes_row = out.split("\n")[2]
    assert "●" in mistakes_row


def test_render_hides_high_confidence_boundaries():
    boundaries = [Boundary(t_s=5.0, label="grasp", confidence=0.99)]
    segs = [_make_seg(0, 0.0, 5.0), _make_seg(1, 5.0, 10.0)]
    out = render_timeline(10.0, boundaries, segs)
    markers_row = out.split("\n")[1]
    assert "▼" not in markers_row  # high-conf boundary is hidden


def test_render_shows_low_confidence_boundaries():
    boundaries = [Boundary(t_s=5.0, label="grasp", confidence=0.60)]
    segs = [_make_seg(0, 0.0, 5.0), _make_seg(1, 5.0, 10.0)]
    out = render_timeline(10.0, boundaries, segs)
    markers_row = out.split("\n")[1]
    assert "▼" in markers_row


def test_render_two_segments_use_alternating_fill():
    segs = [_make_seg(0, 0.0, 5.0), _make_seg(1, 5.0, 10.0)]
    out = render_timeline(10.0, [Boundary(t_s=5.0, label="grasp", confidence=0.99)], segs)
    bar = out.split("\n")[0]
    # Must contain at least one of each fill char
    assert "■" in bar
    assert "░" in bar


class _HostApp(App):
    CSS = ""
    def __init__(self, duration, boundaries, segments):
        super().__init__()
        self._duration = duration
        self._boundaries = boundaries
        self._segments = segments
    def compose(self) -> ComposeResult:
        yield Timeline(self._duration, self._boundaries, self._segments)


@pytest.mark.asyncio
async def test_timeline_widget_mounts():
    segs = [_make_seg(0, 0.0, 10.0)]
    async with _HostApp(10.0, [], segs).run_test() as pilot:
        tl = pilot.app.query_one(Timeline)
        assert tl.duration_s == 10.0


@pytest.mark.asyncio
async def test_timeline_widget_updates_state():
    segs = [_make_seg(0, 0.0, 10.0)]
    async with _HostApp(10.0, [], segs).run_test() as pilot:
        tl = pilot.app.query_one(Timeline)
        new_segs = [_make_seg(0, 0.0, 10.0,
                              mistakes=[Mistake(type="slip", t_s=3.0, note="x",
                                                confidence=0.8)])]
        tl.update_state([], new_segs)
        await pilot.pause()
        static = pilot.app.query_one(Static)
        rendered = str(static.render())
        assert "●" in rendered

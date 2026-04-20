"""Subtask timeline with mistake markers, rendered as ASCII bands."""
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vlm_annotate_verify.reviewer.confidence import boundary_visible
from vlm_annotate_verify.schemas import Boundary, Segment


WIDTH = 60
FILL_BLOCKS = ["■", "░"]


def render_timeline(
    duration_s: float,
    boundaries: list[Boundary],
    segments: list[Segment],
) -> str:
    """Return a three-row ASCII representation of the timeline."""
    if duration_s <= 0:
        bar = "│" + ("─" * WIDTH) + "│"
        empty = " " * (WIDTH + 2)
        return f"{bar}\n{empty}\n{empty}"

    cells = [" "] * WIDTH
    seg_starts = [0.0] + [b.t_s for b in boundaries]
    seg_ends = [b.t_s for b in boundaries] + [duration_s]
    for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        ch = FILL_BLOCKS[i % len(FILL_BLOCKS)]
        c_start = int(WIDTH * s / duration_s)
        c_end = int(WIDTH * e / duration_s)
        for c in range(c_start, max(c_start + 1, c_end)):
            if c < WIDTH:
                cells[c] = ch
    bar = "│" + "".join(cells) + "│"

    markers = [" "] * (WIDTH + 2)
    for seg in segments:
        for m in seg.mistakes:
            c = 1 + int(WIDTH * m.t_s / duration_s)
            if 0 <= c < len(markers):
                markers[c] = "●"
    bdry_marks = [" "] * (WIDTH + 2)
    for b in boundaries:
        if boundary_visible(b.confidence):
            c = 1 + int(WIDTH * b.t_s / duration_s)
            if 0 <= c < len(bdry_marks):
                bdry_marks[c] = "▼"
    return f"{bar}\n{''.join(bdry_marks)}\n{''.join(markers)}"


class Timeline(Widget):
    duration_s: reactive[float] = reactive(0.0)

    DEFAULT_CSS = """
    Timeline { height: 4; padding: 0 1; }
    """

    def __init__(
        self,
        duration_s: float,
        boundaries: list[Boundary],
        segments: list[Segment],
    ) -> None:
        super().__init__()
        self._boundaries = boundaries
        self._segments = segments
        self.duration_s = duration_s

    def compose(self) -> ComposeResult:
        yield Static(
            render_timeline(self.duration_s, self._boundaries, self._segments)
        )

    def update_state(
        self,
        boundaries: list[Boundary],
        segments: list[Segment],
    ) -> None:
        self._boundaries = boundaries
        self._segments = segments
        self.query_one(Static).update(
            render_timeline(self.duration_s, boundaries, segments)
        )

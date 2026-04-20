"""Frame display with three modes: grid, scrub, replay.

Image rendering uses textual-imageview's ImageViewer, which picks Kitty
graphics on WezTerm and falls back to sixel or half-block as available.
"""
import asyncio
from enum import Enum
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.reactive import reactive
from textual.widget import Widget

try:
    from textual_imageview.viewer import ImageViewer
except ImportError:  # pragma: no cover - optional at test time
    ImageViewer = None  # type: ignore[assignment]

from vlm_annotate_verify.schemas import Boundary


REPLAY_FPS = 6


class FrameMode(str, Enum):
    GRID = "GRID"
    SCRUB = "SCRUB"
    REPLAY = "REPLAY"


class FrameView(Widget):
    mode: reactive[FrameMode] = reactive(FrameMode.GRID)
    cursor: reactive[int] = reactive(0)

    DEFAULT_CSS = """
    FrameView { height: 12; }
    Grid#frame-grid { grid-size: 8 2; }
    """

    BINDINGS = [
        Binding("left",        "step(-1)",     "back 1 frame",  show=False),
        Binding("right",       "step(1)",      "fwd 1 frame",   show=False),
        Binding("shift+left",  "step(-5)",     "back 5",        show=False),
        Binding("shift+right", "step(5)",      "fwd 5",         show=False),
        Binding("shift+up",    "subtask(-1)",  "prev subtask",  show=False),
        Binding("shift+down",  "subtask(1)",   "next subtask",  show=False),
    ]

    def __init__(
        self,
        frame_paths: list[Path],
        boundaries: list[Boundary],
        duration_s: float,
    ) -> None:
        super().__init__()
        self.frame_paths = frame_paths
        self.boundaries = boundaries
        self.duration_s = duration_s
        self._timer_handle = None

    def compose(self) -> ComposeResult:
        if ImageViewer is None:
            return
        grid = Grid(id="frame-grid")
        yield grid

    def on_mount(self) -> None:
        if ImageViewer is None:
            return
        grid = self.query_one("#frame-grid", Grid)
        for i, p in enumerate(self.frame_paths):
            viewer = ImageViewer(image=str(p), id=f"frame-{i}")
            grid.mount(viewer)
        self._highlight_cursor()

    def watch_cursor(self, _old: int, _new: int) -> None:
        self._highlight_cursor()

    def watch_mode(self, _old: FrameMode, new: FrameMode) -> None:
        if new is FrameMode.REPLAY:
            self._start_replay()
        else:
            self._stop_replay()

    def _highlight_cursor(self) -> None:
        if ImageViewer is None or not self.is_mounted:
            return
        for i in range(len(self.frame_paths)):
            try:
                w = self.query_one(f"#frame-{i}", ImageViewer)
            except Exception:
                continue
            w.styles.border = ("solid", "cyan") if i == self.cursor else None

    def _start_replay(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # no running event loop (e.g. unit test without a pilot)
            self._timer_handle = None
            return
        self._timer_handle = self.set_interval(1 / REPLAY_FPS, self._tick)

    def _stop_replay(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None

    def _tick(self) -> None:
        if not self.frame_paths:
            return
        self.cursor = (self.cursor + 1) % len(self.frame_paths)

    def step(self, delta: int) -> None:
        n = len(self.frame_paths)
        if n == 0:
            return
        self.cursor = max(0, min(n - 1, self.cursor + delta))

    def action_step(self, delta: int) -> None:
        self.step(delta)

    def jump_to_subtask(self, direction: int) -> None:
        """direction = +1 for next subtask, -1 for previous."""
        if not self.boundaries or self.duration_s <= 0:
            return
        n = len(self.frame_paths)
        if n == 0:
            return
        cursor_t = self.duration_s * (self.cursor + 0.5) / n
        starts = [0.0] + [b.t_s for b in self.boundaries]
        idx = max(i for i, s in enumerate(starts) if s <= cursor_t)
        new_idx = max(0, min(len(starts) - 1, idx + direction))
        target_t = starts[new_idx]
        self.cursor = max(0, min(n - 1, int(n * target_t / self.duration_s)))

    def action_subtask(self, direction: int) -> None:
        self.jump_to_subtask(direction)

    def enter_scrub(self) -> None:
        self.mode = FrameMode.SCRUB

    def enter_replay(self) -> None:
        self.mode = FrameMode.REPLAY

    def enter_grid(self) -> None:
        self.mode = FrameMode.GRID

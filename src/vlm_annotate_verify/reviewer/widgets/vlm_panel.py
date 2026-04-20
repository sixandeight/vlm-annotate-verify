"""VLM proposal panel: quality, success, mistakes, reasoning, with single-key edits."""
import re
from dataclasses import replace

from textual.reactive import reactive
from textual.widgets import Static

from vlm_annotate_verify.schemas import Mistake, Segment


GENERIC_NOTE_PATTERNS = [
    r"^minor issue$",
    r"^slight imperfection$",
    r"^small problem$",
    r"^.{0,4}$",  # fewer than 5 chars is too short to be useful
]


def is_generic_note(note: str) -> bool:
    n = note.strip().lower()
    return any(re.search(p, n) for p in GENERIC_NOTE_PATTERNS)


class VLMPanel(Static):
    seg_index: reactive[int] = reactive(0)

    DEFAULT_CSS = """
    VLMPanel { height: auto; padding: 1 1; }
    """

    def __init__(self, segments: list[Segment], task: str) -> None:
        super().__init__("")
        self.segments: list[Segment] = list(segments)
        self.ep_task = task

    def on_mount(self) -> None:
        self.refresh_text()

    def _render(self) -> str:
        lines = [f"Task: {self.ep_task}", ""]
        for i, seg in enumerate(self.segments):
            marker = ">" if i == self.seg_index else " "
            lines.append(
                f"{marker} [{i}] q={seg.quality} ({seg.quality_conf:.0%}) "
                f"success={'yes' if seg.success else 'no'} "
                f"({seg.success_conf:.0%})"
            )
            for mi, m in enumerate(seg.mistakes):
                flag = " GENERIC" if is_generic_note(m.note) else ""
                lines.append(
                    f"     - [{mi}] {m.type} @ {m.t_s:.1f}s "
                    f'"{m.note}"{flag}'
                )
            if seg.reasoning:
                lines.append(f"     reasoning: {seg.reasoning}")
            lines.append("")
        return "\n".join(lines)

    def refresh_text(self) -> None:
        try:
            self.update(self._render())
        except Exception:
            pass

    def set_quality(self, q: int) -> None:
        q = max(1, min(5, q))
        seg = self.segments[self.seg_index]
        self.segments[self.seg_index] = replace(seg, quality=q, quality_conf=1.0)
        self.refresh_text()

    def adjust_quality(self, delta: int) -> None:
        seg = self.segments[self.seg_index]
        new_q = max(1, min(5, seg.quality + delta))
        self.segments[self.seg_index] = replace(seg, quality=new_q, quality_conf=1.0)
        self.refresh_text()

    def toggle_success(self) -> None:
        seg = self.segments[self.seg_index]
        self.segments[self.seg_index] = replace(
            seg, success=not seg.success, success_conf=1.0,
        )
        self.refresh_text()

    def delete_mistake(self, mistake_idx: int) -> None:
        seg = self.segments[self.seg_index]
        if 0 <= mistake_idx < len(seg.mistakes):
            new_mistakes = seg.mistakes[:mistake_idx] + seg.mistakes[mistake_idx + 1:]
            self.segments[self.seg_index] = replace(seg, mistakes=new_mistakes)
            self.refresh_text()

    def add_mistake(self, mistake: Mistake) -> None:
        seg = self.segments[self.seg_index]
        self.segments[self.seg_index] = replace(
            seg, mistakes=seg.mistakes + [mistake],
        )
        self.refresh_text()

    def update_task(self, new_task: str) -> None:
        self.ep_task = new_task
        self.refresh_text()

    def has_unfixed_generic_notes(self) -> bool:
        return any(
            is_generic_note(m.note)
            for seg in self.segments for m in seg.mistakes
        )

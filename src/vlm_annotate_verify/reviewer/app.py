"""Textual App that loads proposals, renders per-episode UI, writes verified records."""
import asyncio
import json
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Input, Select, Static

from vlm_annotate_verify.proposer.gemini import (
    GeminiConfig, MODEL_PRO, GeminiError,
)
from vlm_annotate_verify.proposer.passes import (
    PassError, pass1_boundaries, pass2_labels,
)
from vlm_annotate_verify.reviewer.confidence import (
    CursorTarget, classify, defaults_for,
)
from vlm_annotate_verify.reviewer.keymap import Action, dispatch
from vlm_annotate_verify.reviewer.widgets.footer import KeybindFooter
from vlm_annotate_verify.reviewer.widgets.frame_view import FrameMode, FrameView
from vlm_annotate_verify.reviewer.widgets.timeline import Timeline
from vlm_annotate_verify.reviewer.widgets.vlm_panel import VLMPanel
from vlm_annotate_verify.schemas import (
    Boundary, Mistake, Proposal, Review, Segment, Verified, utc_now_iso,
)


def load_proposals(path: Path) -> list[Proposal]:
    if not path.exists():
        return []
    out: list[Proposal] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "failed":
            continue
        out.append(Proposal.from_json(line))
    return out


def already_verified_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.add(json.loads(line)["ep_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


MISTAKE_TYPES = ["drop", "slip", "miss", "collision", "other"]


class TextInputModal(ModalScreen[str]):
    def __init__(self, prompt: str, initial: str = "") -> None:
        super().__init__()
        self._prompt = prompt
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._prompt),
            Input(value=self._initial, id="text-input"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class MistakeModal(ModalScreen[Mistake | None]):
    def __init__(self, default_t_s: float) -> None:
        super().__init__()
        self._default_t_s = default_t_s

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Add mistake - pick type, then type a note and press enter"),
            Select([(t, t) for t in MISTAKE_TYPES], id="mistake-type", value="slip"),
            Input(placeholder="note (e.g. cube wobble)", id="mistake-note"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        type_select = self.query_one("#mistake-type", Select)
        self.dismiss(Mistake(
            type=str(type_select.value),
            t_s=self._default_t_s,
            note=event.value,
            confidence=1.0,
        ))


class ReviewerApp(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; overflow-y: auto; }
    """

    BINDINGS = [Binding("ctrl+c", "quit", "Quit", priority=True)]

    def __init__(
        self,
        dataset_root: Path,
        gemini_config: GeminiConfig,
        reviewer_id: str,
        from_ep: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.gemini_config = gemini_config
        self.reviewer_id = reviewer_id
        self.proposals_path = dataset_root / "proposals.jsonl"
        self.verified_path = dataset_root / "verified.jsonl"
        self.from_ep = from_ep
        self.queue: list[Proposal] = []
        self.queue_idx = 0
        self.current_segments: list[Segment] = []
        self.current_boundaries: list[Boundary] = []
        self.actions_log: list[str] = []
        self.start_time: float = 0.0
        self.reprompt_used = False

    def on_mount(self) -> None:
        self._load_queue()
        if self.queue:
            self._render_current()

    def _load_queue(self) -> None:
        all_proposals = load_proposals(self.proposals_path)
        verified = already_verified_ids(self.verified_path)
        self.queue = [p for p in all_proposals if p.ep_id not in verified]
        if self.from_ep:
            for i, p in enumerate(self.queue):
                if p.ep_id == self.from_ep:
                    self.queue_idx = i
                    return
        self.queue_idx = 0

    def _current(self) -> Proposal | None:
        if 0 <= self.queue_idx < len(self.queue):
            return self.queue[self.queue_idx]
        return None

    def _min_ep_confidence(self, prop: Proposal) -> float:
        confs = (
            [s.quality_conf for s in prop.segments]
            + [s.success_conf for s in prop.segments]
        )
        return min(confs) if confs else 1.0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(id="main")
        yield KeybindFooter(id="footer")

    def _render_current(self) -> None:
        prop = self._current()
        main = self.query_one("#main", Vertical)
        for child in list(main.children):
            child.remove()
        if prop is None:
            main.mount(Static("Queue empty - all eps verified."))
            return

        self.current_segments = list(prop.segments)
        self.current_boundaries = list(prop.boundaries)
        self.actions_log = []
        self.reprompt_used = False
        self.start_time = time.monotonic()

        ep_conf = self._min_ep_confidence(prop)
        defaults = defaults_for(ep_conf)
        band = classify(ep_conf)
        header = Static(
            f"ep {self.queue_idx + 1}/{len(self.queue)}  "
            f"id={prop.ep_id}  conf {band.value} ({ep_conf:.0%})"
        )
        timeline = Timeline(
            prop.duration_s, self.current_boundaries, self.current_segments,
        )
        frames = FrameView(
            [self.dataset_root / fp for fp in prop.frame_paths],
            self.current_boundaries,
            prop.duration_s,
        )
        panel = VLMPanel(self.current_segments, prop.task)

        main.mount(header)
        main.mount(timeline)
        main.mount(frames)
        main.mount(panel)

        if defaults.cursor is CursorTarget.REPLAY:
            frames.enter_replay()
        elif defaults.auto_enter_scrub:
            frames.enter_scrub()

    def on_key(self, event) -> None:
        action = dispatch(event.key)
        if action is Action.NOOP:
            return
        if action is Action.HELP_TOGGLE:
            self.query_one("#footer", KeybindFooter).toggle()
            return
        if action in {
            Action.QUALITY_INC, Action.QUALITY_DEC,
            Action.QUALITY_SET_1, Action.QUALITY_SET_2, Action.QUALITY_SET_3,
            Action.QUALITY_SET_4, Action.QUALITY_SET_5,
            Action.SUCCESS_TOGGLE,
        }:
            self._apply_panel_action(action)
            return
        if action is Action.MISTAKE_ADD:
            self.push_screen(
                MistakeModal(default_t_s=0.0),
                self._on_mistake_added,
            )
            return
        if action is Action.MISTAKE_DELETE:
            self._delete_current_mistake()
            return
        if action is Action.EDIT_TASK:
            panel = self.query_one(VLMPanel)
            self.push_screen(
                TextInputModal("Edit task description", panel.task),
                self._on_task_edited,
            )
            return
        if action is Action.FULL_FRAME:
            frames = self.query_one(FrameView)
            if frames.mode is FrameMode.SCRUB:
                frames.enter_grid()
            else:
                frames.enter_scrub()
            return
        if action is Action.NAV_NEXT:
            self._advance(1)
            return
        if action is Action.NAV_PREV:
            self._advance(-1)
            return
        if action is Action.SAVE_QUIT:
            self.exit()
            return
        if action in {Action.COMMIT_NEXT, Action.ACCEPT_ALL}:
            self._commit_and_next()
            return
        if action is Action.REPROMPT:
            asyncio.create_task(self._reprompt())
            return

    def _apply_panel_action(self, action: Action) -> None:
        panel = self.query_one(VLMPanel)
        if action is Action.QUALITY_INC:
            panel.adjust_quality(+1)
            self.actions_log.append("q+1")
        elif action is Action.QUALITY_DEC:
            panel.adjust_quality(-1)
            self.actions_log.append("q-1")
        elif action is Action.SUCCESS_TOGGLE:
            panel.toggle_success()
            self.actions_log.append("toggle_success")
        elif action.value.startswith("QUALITY_SET_"):
            q = int(action.value[-1])
            panel.set_quality(q)
            self.actions_log.append(f"q={q}")

    def _advance(self, delta: int) -> None:
        new_idx = self.queue_idx + delta
        if 0 <= new_idx < len(self.queue):
            self.queue_idx = new_idx
            self._render_current()

    def _commit_and_next(self) -> None:
        prop = self._current()
        if prop is None:
            return
        panel = self.query_one(VLMPanel)
        if panel.has_unfixed_generic_notes():
            self.bell()
            return
        verified = Verified(
            ep_id=prop.ep_id,
            task=panel.task,
            boundaries=self.current_boundaries,
            segments=list(panel.segments),
            review=Review(
                reviewer_id=self.reviewer_id,
                review_seconds=time.monotonic() - self.start_time,
                actions=self.actions_log,
                reprompt_used=self.reprompt_used,
            ),
            verified_at=utc_now_iso(),
        )
        with self.verified_path.open("a") as f:
            f.write(verified.to_json() + "\n")
        self._advance(1)

    async def _reprompt(self) -> None:
        prop = self._current()
        if prop is None:
            return
        video_path = self.dataset_root / prop.video_path
        try:
            task, boundaries = await pass1_boundaries(
                self.gemini_config, video_path, MODEL_PRO,
            )
            segments = await pass2_labels(
                self.gemini_config, video_path, boundaries, prop.duration_s, MODEL_PRO,
            )
        except (PassError, GeminiError) as e:
            try:
                self.notify(f"Pro reprompt failed: {e}", severity="error")
            except Exception:
                pass
            return
        self.current_boundaries = boundaries
        self.current_segments = segments
        self.reprompt_used = True
        try:
            panel = self.query_one(VLMPanel)
            panel.task = task
            panel.segments = list(segments)
            panel.refresh_text()
            timeline = self.query_one(Timeline)
            timeline.update_state(boundaries, segments)
        except Exception:
            pass
        self.actions_log.append("reprompt:pro")

    def _on_mistake_added(self, mistake: Mistake | None) -> None:
        if mistake is None:
            return
        panel = self.query_one(VLMPanel)
        panel.add_mistake(mistake)
        self.actions_log.append(f"add_mistake:{mistake.type}")

    def _on_task_edited(self, new_task: str | None) -> None:
        if new_task is None:
            return
        panel = self.query_one(VLMPanel)
        panel.update_task(new_task)
        self.actions_log.append("edit_task")

    def _delete_current_mistake(self) -> None:
        panel = self.query_one(VLMPanel)
        seg = panel.segments[panel.seg_index]
        if not seg.mistakes:
            return
        panel.delete_mistake(len(seg.mistakes) - 1)
        self.actions_log.append("delete_mistake")

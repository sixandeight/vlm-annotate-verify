from pathlib import Path

import pytest

from vlm_annotate_verify.proposer.gemini import GeminiConfig
from vlm_annotate_verify.reviewer.app import MistakeModal, ReviewerApp, TextInputModal
from vlm_annotate_verify.schemas import Mistake


def test_mistake_modal_constructor_takes_default_t_s():
    modal = MistakeModal(default_t_s=3.14)
    assert modal._default_t_s == 3.14


def test_text_input_modal_constructor_fields():
    modal = TextInputModal("Edit task", "old value")
    assert modal._prompt == "Edit task"
    assert modal._initial == "old value"


def test_reviewer_app_mistake_callback_appends(tmp_path):
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )

    added = {}

    class _FakePanel:
        def add_mistake(self, m):
            added["m"] = m

    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    m = Mistake(type="slip", t_s=3.0, note="wobble here", confidence=1.0)
    app._on_mistake_added(m)
    assert added["m"] == m
    assert app.actions_log == ["add_mistake:slip"]


def test_reviewer_app_mistake_callback_none_is_noop(tmp_path):
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )

    called = {"n": 0}

    class _FakePanel:
        def add_mistake(self, _m):
            called["n"] += 1

    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    app._on_mistake_added(None)
    assert called["n"] == 0
    assert app.actions_log == []


def test_reviewer_app_task_edit_callback(tmp_path):
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )

    captured = {}

    class _FakePanel:
        def update_task(self, t):
            captured["t"] = t

    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    app._on_task_edited("new task")
    assert captured["t"] == "new task"
    assert app.actions_log == ["edit_task"]


def test_reviewer_app_task_edit_none_is_noop(tmp_path):
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )

    called = {"n": 0}

    class _FakePanel:
        def update_task(self, _t):
            called["n"] += 1

    app.query_one = lambda _cls=None: _FakePanel()  # type: ignore[assignment]
    app._on_task_edited(None)
    assert called["n"] == 0
    assert app.actions_log == []


def test_reviewer_app_mistake_delete_pops_last(tmp_path):
    """Pressing x on a segment with two mistakes should delete the last one
    and append delete_mistake to actions_log."""
    from vlm_annotate_verify.schemas import Segment
    app = ReviewerApp(
        dataset_root=tmp_path,
        gemini_config=GeminiConfig(api_key="x"),
        reviewer_id="nathan",
    )
    mistakes = [
        Mistake(type="slip", t_s=1.0, note="a good note", confidence=0.8),
        Mistake(type="drop", t_s=2.0, note="another good note", confidence=0.9),
    ]
    seg = Segment(idx=0, start_s=0, end_s=5, quality=4, quality_conf=0.8,
                  success=True, success_conf=0.9, mistakes=mistakes)

    deleted = {"idx": None}

    class _FakePanel:
        def __init__(self):
            self.seg_index = 0
            self.segments = [seg]
        def delete_mistake(self, idx):
            deleted["idx"] = idx

    panel = _FakePanel()
    app.query_one = lambda _cls=None: panel  # type: ignore[assignment]
    app._delete_current_mistake()
    assert deleted["idx"] == 1  # last index = len(mistakes) - 1
    assert app.actions_log == ["delete_mistake"]

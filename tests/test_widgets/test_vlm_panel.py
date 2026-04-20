import pytest

from vlm_annotate_verify.reviewer.widgets.vlm_panel import (
    GENERIC_NOTE_PATTERNS, VLMPanel, is_generic_note,
)
from vlm_annotate_verify.schemas import Mistake, Segment


def _make_seg(quality=4, success=True, mistakes=None):
    return Segment(
        idx=0, start_s=0.0, end_s=5.0,
        quality=quality, quality_conf=0.8,
        success=success, success_conf=0.9,
        mistakes=mistakes or [],
        reasoning="smooth grasp",
    )


def test_is_generic_note_detects_common_garbage():
    assert is_generic_note("minor issue") is True
    assert is_generic_note("slight imperfection") is True
    assert is_generic_note("small problem") is True
    assert is_generic_note("x") is True  # too short


def test_is_generic_note_accepts_specific_notes():
    assert is_generic_note("cube wobble during transfer") is False
    assert is_generic_note("gripper slipped at 3.2s") is False


def test_generic_note_patterns_non_empty():
    assert len(GENERIC_NOTE_PATTERNS) > 0


def test_vlm_panel_set_quality_mutates_segment():
    seg = _make_seg(quality=3)
    panel = VLMPanel([seg], task="place cube")
    panel.set_quality(5)
    assert panel.segments[0].quality == 5
    assert panel.segments[0].quality_conf == 1.0


def test_vlm_panel_adjust_quality_clamps():
    seg = _make_seg(quality=5)
    panel = VLMPanel([seg], task="x")
    panel.adjust_quality(+1)
    assert panel.segments[0].quality == 5  # clamped at 5
    panel.set_quality(1)
    panel.adjust_quality(-1)
    assert panel.segments[0].quality == 1  # clamped at 1


def test_vlm_panel_toggle_success():
    seg = _make_seg(success=True)
    panel = VLMPanel([seg], task="x")
    panel.toggle_success()
    assert panel.segments[0].success is False
    panel.toggle_success()
    assert panel.segments[0].success is True


def test_vlm_panel_delete_mistake():
    m0 = Mistake(type="slip", t_s=1.0, note="wobble here", confidence=0.6)
    m1 = Mistake(type="drop", t_s=2.0, note="bounced on floor", confidence=0.7)
    seg = _make_seg(mistakes=[m0, m1])
    panel = VLMPanel([seg], task="x")
    panel.delete_mistake(0)
    assert len(panel.segments[0].mistakes) == 1
    assert panel.segments[0].mistakes[0].type == "drop"


def test_vlm_panel_delete_mistake_out_of_range_is_noop():
    seg = _make_seg(mistakes=[])
    panel = VLMPanel([seg], task="x")
    panel.delete_mistake(5)  # must not raise
    assert panel.segments[0].mistakes == []


def test_vlm_panel_add_mistake():
    seg = _make_seg(mistakes=[])
    panel = VLMPanel([seg], task="x")
    panel.add_mistake(Mistake(type="miss", t_s=0.5, note="grabbed air", confidence=0.9))
    assert len(panel.segments[0].mistakes) == 1
    assert panel.segments[0].mistakes[0].type == "miss"


def test_vlm_panel_update_task():
    seg = _make_seg()
    panel = VLMPanel([seg], task="old task")
    panel.update_task("new task")
    assert panel.task == "new task"


def test_has_unfixed_generic_notes_true_when_any_mistake_is_generic():
    bad = Mistake(type="slip", t_s=1.0, note="minor issue", confidence=0.5)
    good = Mistake(type="drop", t_s=2.0, note="cube bounced on floor", confidence=0.7)
    panel = VLMPanel([_make_seg(mistakes=[good, bad])], task="x")
    assert panel.has_unfixed_generic_notes() is True


def test_has_unfixed_generic_notes_false_when_all_specific():
    good1 = Mistake(type="slip", t_s=1.0, note="cube wobbled during transfer",
                    confidence=0.5)
    good2 = Mistake(type="drop", t_s=2.0, note="cube bounced on floor",
                    confidence=0.7)
    panel = VLMPanel([_make_seg(mistakes=[good1, good2])], task="x")
    assert panel.has_unfixed_generic_notes() is False


def test_has_unfixed_generic_notes_false_on_no_mistakes():
    panel = VLMPanel([_make_seg(mistakes=[])], task="x")
    assert panel.has_unfixed_generic_notes() is False

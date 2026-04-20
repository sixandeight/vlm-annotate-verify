import pytest

from vlm_annotate_verify.reviewer.confidence import (
    BOUNDARY_AUTOACCEPT_THRESHOLD,
    Band, CursorTarget, Defaults,
    boundary_visible, classify, defaults_for,
)


def test_classify_high_band():
    assert classify(0.95) is Band.HIGH
    assert classify(0.86) is Band.HIGH


def test_classify_medium_band():
    assert classify(0.85) is Band.MEDIUM
    assert classify(0.50) is Band.MEDIUM
    assert classify(0.72) is Band.MEDIUM


def test_classify_low_band():
    assert classify(0.49) is Band.LOW
    assert classify(0.0) is Band.LOW


def test_defaults_for_high_targets_space():
    d = defaults_for(0.95)
    assert d.band is Band.HIGH
    assert d.cursor is CursorTarget.SPACE
    assert d.show_reasoning is False
    assert d.show_frames is False
    assert d.auto_enter_scrub is False


def test_defaults_for_medium_targets_edit():
    d = defaults_for(0.72)
    assert d.band is Band.MEDIUM
    assert d.cursor is CursorTarget.EDIT
    assert d.show_reasoning is True
    assert d.show_frames is True
    assert d.auto_enter_scrub is False


def test_defaults_for_low_targets_replay_and_auto_scrub():
    d = defaults_for(0.30)
    assert d.band is Band.LOW
    assert d.cursor is CursorTarget.REPLAY
    assert d.show_reasoning is True
    assert d.show_frames is True
    assert d.auto_enter_scrub is True


def test_boundary_visible_above_threshold_is_hidden():
    assert boundary_visible(0.99) is False


def test_boundary_visible_at_or_below_threshold_is_shown():
    assert boundary_visible(BOUNDARY_AUTOACCEPT_THRESHOLD) is True
    assert boundary_visible(0.0) is True


def test_defaults_dataclass_is_frozen():
    d = defaults_for(0.95)
    with pytest.raises(Exception):
        d.cursor = CursorTarget.EDIT

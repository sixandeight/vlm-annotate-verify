import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from vlm_annotate_verify.reviewer.widgets.footer import (
    COMPACT, FULL, KeybindFooter,
)


class _HostApp(App):
    def compose(self) -> ComposeResult:
        yield KeybindFooter(id="footer")


def test_compact_and_full_strings_are_non_empty():
    assert len(COMPACT.strip()) > 0
    assert len(FULL.strip()) > 0


def test_compact_mentions_core_actions():
    for needle in ("SPACE", "r=", "j/k", "?=help"):
        assert needle in COMPACT


def test_full_mentions_quality_and_mistakes_and_task():
    assert "Quality" in FULL
    assert "Mistakes" in FULL
    assert "Task" in FULL


@pytest.mark.asyncio
async def test_footer_starts_compact():
    async with _HostApp().run_test() as pilot:
        footer = pilot.app.query_one(KeybindFooter)
        assert footer.expanded is False
        text = pilot.app.query_one("#footer-text", Static).render()
        assert str(text) == COMPACT


@pytest.mark.asyncio
async def test_footer_toggle_expands():
    async with _HostApp().run_test() as pilot:
        footer = pilot.app.query_one(KeybindFooter)
        footer.toggle()
        await pilot.pause()
        assert footer.expanded is True
        text = pilot.app.query_one("#footer-text", Static).render()
        assert str(text) == FULL


@pytest.mark.asyncio
async def test_footer_toggle_twice_is_compact_again():
    async with _HostApp().run_test() as pilot:
        footer = pilot.app.query_one(KeybindFooter)
        footer.toggle()
        footer.toggle()
        await pilot.pause()
        assert footer.expanded is False

"""Always-visible keybind footer with a toggle to expand into full help."""
from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


COMPACT = "KEYS: SPACE=accept  m=+mistake  r=re-prompt  j/k=nav  ?=help"

FULL = """\
Quality:    + -    or 1 2 3 4 5
Success:    s (toggle)
Mistakes:   m=add  x=delete
Task:       e=edit task description
Actions:    SPACE=accept all   ENTER=commit+next   r=re-prompt VLM
Nav:        j=next  k=prev  q=save+quit  f=full-screen frame  ?=help"""


class KeybindFooter(Widget):
    expanded: reactive[bool] = reactive(False)

    DEFAULT_CSS = """
    KeybindFooter {
        height: auto;
        background: $boost;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Container(Static(COMPACT, id="footer-text"))

    def watch_expanded(self, expanded: bool) -> None:
        try:
            text = self.query_one("#footer-text", Static)
        except Exception:
            return
        text.update(FULL if expanded else COMPACT)

    def toggle(self) -> None:
        self.expanded = not self.expanded

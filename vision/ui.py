"""Terminal UI pieces for Vision: the input box, the model picker, message rendering."""
from __future__ import annotations

import os
from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

ACCENT = "bright_cyan"
USER_BG = "on grey23"
PREFIX_W = 9  # width of the "Vision ›" / "you ›" gutter

PT_STYLE = Style.from_dict({
    "frame.border": "ansicyan",
    "frame.label": "ansicyan",
    "placeholder": "#6b7580 italic",
    "status": "#8a939c",
    "status.key": "#c9d1d9 bold",
    "pick.title": "bold",
    "pick.cursor": "ansicyan bold",
    "pick.selected": "ansicyan bold",
    "pick.hint": "#6b7580",
    "pick.desc": "#8a939c",
})

MODEL_CHOICES = [
    ("", "default", "whatever your Claude Code default is"),
    ("fable", "Fable 5.1", "most capable, slowest; best for hard problems"),
    ("opus", "Opus 5", "very capable, good all-rounder"),
    ("sonnet", "Sonnet 5", "fast and strong; best for voice conversations"),
    ("haiku", "Haiku 4.5", "fastest and lightest"),
]


# ---------------------------------------------------------------- header
def show_header(console: Console, *lines: str) -> None:
    """A clean 'Vision' header: sets the terminal title, draws a rule, then dim detail lines."""
    console.set_window_title("Vision")
    console.print()
    console.rule(f"[bold {ACCENT}]Vision[/bold {ACCENT}]", style=ACCENT)
    for line in lines:
        console.print(Text(line, style="dim"), justify="center")
    console.print()


# ---------------------------------------------------------------- transcript rendering
def show_user(console: Console, text: str) -> None:
    """The user's message as a full-width highlighted band, like Claude Code."""
    body = Text.assemble(("you › ", "bold yellow"), (text, "bold"))
    console.print(Padding(body, (0, 1), style=USER_BG, expand=True))


def _reply_grid(content, label: str = "Vision ›") -> Table:
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=PREFIX_W - 1, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(Text(label, style=f"bold {ACCENT}"), content)
    return grid


class ReplyView:
    """Live-updating reply that starts beside the 'Vision ›' prefix and wraps under it."""

    def __init__(self, console: Console, markdown: bool = True):
        self.console = console
        self.markdown = markdown
        self.buf = ""
        self.status = ""
        self._live = Live(_reply_grid(Text("")), console=console, refresh_per_second=12, vertical_overflow="visible")

    def __enter__(self):
        self._live.__enter__()
        return self

    def _render(self):
        body = Markdown(self.buf) if self.markdown else Text(self.buf)
        if self.status:
            body = Group(body, Text(self.status, style="dim italic"))
        self._live.update(_reply_grid(body))

    def append(self, delta: str) -> None:
        self.buf += delta
        self._render()

    def set_status(self, text: str) -> None:
        self.status = text
        self._render()

    def __exit__(self, *exc):
        self.status = ""
        if not self.buf.strip():
            self.buf = "…"
        self._render()
        self._live.__exit__(*exc)
        self.console.print()
        return False


# ---------------------------------------------------------------- input box
class InputBox:
    """A framed message box at the bottom of the terminal with a status line beneath it.

    Enter sends, Ctrl-J inserts a newline, Up/Down walk history, Ctrl-C / Ctrl-D leave.
    """

    def __init__(self, history_path: str, status: Callable[[], str]):
        self.status_fn = status
        self.area = TextArea(
            multiline=True,
            wrap_lines=True,
            history=FileHistory(history_path),
            prompt=[("class:frame.label", "› ")],
            height=lambda: min(6, max(1, self.area.text.count("\n") + 1)),
            accept_handler=None,
        )
        kb = KeyBindings()

        @kb.add("enter")
        def _send(event):
            text = self.area.text
            if text.strip():
                self.area.buffer.append_to_history()
            event.app.exit(result=text)

        @kb.add("c-j")
        def _newline(event):
            self.area.buffer.insert_text("\n")

        @kb.add("c-c")
        def _cancel(event):
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _eof(event):
            if not self.area.text:
                event.app.exit(exception=EOFError())

        empty = Condition(lambda: not self.area.text)
        placeholder = ConditionalContainer(
            Window(FormattedTextControl([("class:placeholder", "Message Vision…  (Enter to send · Ctrl-J newline · /help)")]), height=1),
            filter=empty,
        )
        frame = Frame(FloatContainer(self.area, floats=[Float(placeholder, top=0, left=2)]), title=[("class:frame.label", " Vision ")])
        status_win = Window(FormattedTextControl(lambda: FormattedText([("class:status", "  " + self.status_fn())])), height=1)
        self.app = Application(
            layout=Layout(HSplit([frame, status_win]), focused_element=self.area),
            key_bindings=kb,
            style=PT_STYLE,
            mouse_support=False,
            erase_when_done=True,
        )

    def read(self) -> str:
        """Blocks until Enter. Raises KeyboardInterrupt / EOFError to leave."""
        self.area.text = ""
        return self.app.run()


# ---------------------------------------------------------------- picker
def pick(title: str, options: list[tuple[str, str, str]], current: str = "") -> str | None:
    """Inline arrow-key selector. options = [(value, label, description)]. Returns value or None."""
    idx = next((i for i, o in enumerate(options) if o[0] == current), 0)
    state = {"i": idx}

    def render():
        out = [("class:pick.title", f" {title}\n")]
        for i, (val, label, desc) in enumerate(options):
            cur = i == state["i"]
            marker = "❯ " if cur else "  "
            mark_now = "  (current)" if val == current else ""
            out.append(("class:pick.cursor" if cur else "", f" {marker}"))
            out.append(("class:pick.selected" if cur else "", f"{label:<12}"))
            out.append(("class:pick.desc", f" {desc}{mark_now}\n"))
        out.append(("class:pick.hint", "  ↑/↓ move · Enter select · Esc cancel"))
        return FormattedText(out)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(e):
        state["i"] = (state["i"] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(e):
        state["i"] = (state["i"] + 1) % len(options)

    @kb.add("enter")
    def _ok(e):
        e.app.exit(result=options[state["i"]][0])

    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(e):
        e.app.exit(result=None)

    for n in range(1, min(9, len(options)) + 1):
        def _num(e, n=n):
            e.app.exit(result=options[n - 1][0])
        kb.add(str(n))(_num)

    app = Application(
        layout=Layout(Window(FormattedTextControl(render), height=len(options) + 2, always_hide_cursor=True)),
        key_bindings=kb, style=PT_STYLE, erase_when_done=True,
    )
    return app.run()


def short_path(p: str) -> str:
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home) else p

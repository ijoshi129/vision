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


def header_renderable(*lines: str):
    """The same clean header as show_header, as a renderable for the full-screen transcript."""
    from rich.align import Align
    from rich.rule import Rule

    parts = [Text(""), Rule(f"[bold {ACCENT}]Vision[/bold {ACCENT}]", style=ACCENT)]
    parts += [Align.center(Text(line, style="dim")) for line in lines]
    parts.append(Text(""))
    return Group(*parts)


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


# ---------------------------------------------------------------- full-screen chat
import io
import threading
import time

from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.data_structures import Point as _Point  # noqa: F401  (older import path)

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def render_ansi(renderable, width: int) -> str:
    """Render a rich renderable to an ANSI string at the given width."""
    buf = io.StringIO()
    c = Console(file=buf, width=max(20, width), force_terminal=True, color_system="truecolor", highlight=False, soft_wrap=False)
    c.print(renderable)
    return buf.getvalue()


class _Entry:
    def __init__(self, renderable, gap_before: bool = False):
        self.renderable = renderable
        self.gap_before = gap_before
        self._cache: dict[int, str] = {}

    def ansi(self, width: int) -> str:
        if width not in self._cache:
            body = Group(Text(""), self.renderable) if self.gap_before else self.renderable
            self._cache[width] = render_ansi(body, width)
        return self._cache[width]

    def invalidate(self):
        self._cache.clear()


class _ReplyEntry(_Entry):
    def __init__(self, markdown: bool):
        super().__init__(None)
        self.markdown = markdown
        self.buf = ""
        self.status = ""
        self.done = False

    def renderable_now(self):
        body = Markdown(self.buf) if self.markdown else Text(self.buf)
        if self.status:
            body = Group(body, Text(self.status, style="dim italic"))
        if not self.buf and not self.status:
            body = Text("…", style="dim")
        grid = _reply_grid(body)
        return Group(grid, Text("")) if self.done else grid

    def ansi(self, width: int) -> str:
        if width not in self._cache:
            self._cache[width] = render_ansi(self.renderable_now(), width)
        return self._cache[width]


class ChatScreen:
    """Claude-Code-style layout: scrolling transcript on top, framed input pinned to the bottom,
    status line under it. Runs full-screen; `run()` returns a request tuple when the caller must
    do something outside the screen (voice modes), or ("quit",) to leave."""

    def __init__(self, history_path: str, status: Callable[[], str]):
        self.status_fn = status
        self.entries: list[_Entry] = []
        self.busy = False
        self.busy_label = ""
        self.scroll_up = 0  # lines scrolled up from the bottom (0 = follow)
        self.on_submit: Callable[[str], None] = lambda text: None
        self.on_cancel: Callable[[], None] = lambda: None
        self._picker = None  # dict(title, options, current, idx, cb)
        self._lock = threading.Lock()
        self._last_stream_render = 0.0

        self.area = TextArea(
            multiline=True, wrap_lines=True, history=FileHistory(history_path),
            prompt=[("class:frame.label", "› ")],
            height=lambda: Dimension.exact(min(8, max(1, self.area.text.count("\n") + 1))),
        )
        picker_open = Condition(lambda: self._picker is not None)
        kb = KeyBindings()

        @kb.add("enter", filter=~picker_open)
        def _send(event):
            text = self.area.text
            if not text.strip():
                return
            if self.busy and not text.lstrip().startswith("/"):
                self.busy_label = "still replying… (Esc cancels)"
                return
            self.area.buffer.append_to_history()
            self.area.text = ""
            self.on_submit(text)

        @kb.add("c-j")
        def _newline(event):
            self.area.buffer.insert_text("\n")

        @kb.add("escape", eager=True, filter=~picker_open)
        def _esc(event):
            if self.busy:
                self.on_cancel()

        @kb.add("c-c", filter=~picker_open)
        def _ctrl_c(event):
            if self.busy:
                self.on_cancel()
            elif self.area.text:
                self.area.text = ""
            else:
                event.app.exit(result=("quit",))

        @kb.add("c-d")
        def _eof(event):
            if not self.area.text:
                event.app.exit(result=("quit",))

        @kb.add("pageup")
        def _pgup(event):
            self.scroll_up += max(1, self._transcript_height() - 2)

        @kb.add("pagedown")
        def _pgdn(event):
            self.scroll_up = max(0, self.scroll_up - max(1, self._transcript_height() - 2))

        @kb.add("c-end")
        @kb.add("escape", "end")
        def _bottom(event):
            self.scroll_up = 0

        # picker keys
        @kb.add("up", filter=picker_open)
        @kb.add("k", filter=picker_open)
        def _pk_up(event):
            p = self._picker
            p["idx"] = (p["idx"] - 1) % len(p["options"])

        @kb.add("down", filter=picker_open)
        @kb.add("j", filter=picker_open)
        def _pk_down(event):
            p = self._picker
            p["idx"] = (p["idx"] + 1) % len(p["options"])

        @kb.add("enter", filter=picker_open)
        def _pk_ok(event):
            p, self._picker = self._picker, None
            p["cb"](p["options"][p["idx"]][0])

        @kb.add("escape", filter=picker_open, eager=True)
        @kb.add("c-c", filter=picker_open)
        @kb.add("q", filter=picker_open)
        def _pk_cancel(event):
            self._picker = None

        for n in range(1, 10):
            def _num(event, n=n):
                p = self._picker
                if p and n <= len(p["options"]):
                    self._picker = None
                    p["cb"](p["options"][n - 1][0])
            kb.add(str(n), filter=picker_open)(_num)

        empty = Condition(lambda: not self.area.text)
        placeholder = ConditionalContainer(
            Window(FormattedTextControl([("class:placeholder", "Message Vision…  (Enter to send · Ctrl-J newline · PgUp/PgDn scroll · /help)")]), height=1),
            filter=empty,
        )
        self.transcript = Window(
            FormattedTextControl(self._transcript_text, get_cursor_position=self._cursor, show_cursor=False, focusable=False),
            wrap_lines=True, always_hide_cursor=True,
        )
        frame = Frame(FloatContainer(self.area, floats=[Float(placeholder, top=0, left=2)]), title=[("class:frame.label", " Vision ")])
        status_win = Window(FormattedTextControl(self._status_text), height=1)
        picker_win = ConditionalContainer(
            Frame(Window(FormattedTextControl(self._picker_text), height=lambda: (len(self._picker["options"]) + 2) if self._picker else 1)),
            filter=picker_open,
        )
        body = HSplit([self.transcript, picker_win, frame, status_win])
        self.app = Application(
            layout=Layout(body, focused_element=self.area),
            key_bindings=kb, style=PT_STYLE, full_screen=True, mouse_support=False,
            refresh_interval=0.15,
        )

    # -- geometry helpers
    def _width(self) -> int:
        try:
            return self.app.output.get_size().columns
        except Exception:
            return 100

    def _transcript_height(self) -> int:
        try:
            return max(3, self.app.output.get_size().rows - 6)
        except Exception:
            return 20

    # -- rendering
    def _transcript_text(self):
        w = self._width()
        with self._lock:
            text = "".join(e.ansi(w) for e in self.entries)
        self._nlines = text.count("\n")
        return ANSI(text)

    def _cursor(self):
        n = getattr(self, "_nlines", 0)
        return Point(x=0, y=max(0, n - 1 - self.scroll_up))

    def _status_text(self):
        parts = [("class:status", "  " + self.status_fn())]
        if self.busy:
            spin = SPINNER[int(time.time() * 8) % len(SPINNER)]
            parts.append(("class:status.key", f"   {spin} {self.busy_label or 'Vision is thinking…'}"))
        if self.scroll_up:
            parts.append(("class:status", "   ↓ scrolled up · PgDn / Alt-End to follow"))
        return FormattedText(parts)

    def _picker_text(self):
        p = self._picker
        if not p:
            return ""
        out = [("class:pick.title", f" {p['title']}\n")]
        for i, (val, label, desc) in enumerate(p["options"]):
            cur = i == p["idx"]
            out.append(("class:pick.cursor" if cur else "", f" {'❯ ' if cur else '  '}"))
            out.append(("class:pick.selected" if cur else "", f"{label:<12}"))
            out.append(("class:pick.desc", f" {desc}{'  (current)' if val == p['current'] else ''}\n"))
        out.append(("class:pick.hint", "  ↑/↓ move · Enter select · Esc cancel"))
        return FormattedText(out)

    # -- public API (thread-safe)
    def add(self, renderable, gap_before: bool = False) -> None:
        with self._lock:
            self.entries.append(_Entry(renderable, gap_before))
        self.scroll_up = 0
        self.app.invalidate()

    def start_reply(self, markdown: bool = True) -> _ReplyEntry:
        e = _ReplyEntry(markdown)
        with self._lock:
            self.entries.append(e)
        self.busy = True
        self.busy_label = ""
        self.app.invalidate()
        return e

    def update_reply(self, e: _ReplyEntry, delta: str = "", status: str | None = None, force: bool = False) -> None:
        if delta:
            e.buf += delta
            e.status = ""
        if status is not None:
            e.status = status
        now = time.time()
        if force or now - self._last_stream_render > 0.08:
            self._last_stream_render = now
            e.invalidate()
            self.scroll_up = 0
            self.app.invalidate()

    def end_reply(self, e: _ReplyEntry) -> None:
        self.busy = False
        self.busy_label = ""
        if not e.buf.strip() and not e.status:
            e.buf = "…"
        e.status = ""
        e.done = True
        e.invalidate()
        self.app.invalidate()

    def open_picker(self, title: str, options, current: str, cb: Callable[[str], None]) -> None:
        idx = next((i for i, o in enumerate(options) if o[0] == current), 0)
        self._picker = {"title": title, "options": options, "current": current, "idx": idx, "cb": cb}
        self.app.invalidate()

    def exit(self, result) -> None:
        self.app.exit(result=result)

    def run(self):
        try:
            self.app.output.set_title("Vision")
        except Exception:
            pass
        return self.app.run()

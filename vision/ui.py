"""Terminal UI pieces for Vision: the input box, the model picker, message rendering."""
from __future__ import annotations

import base64
import os
import re
import sys
from collections.abc import Callable
from functools import lru_cache

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame, Label, TextArea
from prompt_toolkit.widgets.base import Border
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.color import Color
from rich.live import Live
from rich.markdown import CodeBlock, Heading, ImageItem, ListItem, Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.syntax import Syntax
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from vision.buddy import HEIGHT as BUDDY_HEIGHT, MIN_COLUMNS as BUDDY_MIN_COLUMNS, WIDTH as BUDDY_WIDTH, Buddy, FACE_BG, FACE_X0, FACE_X1, idle_sprite
from vision.brain import clock

ACCENT = "bright_cyan"
USER_STYLE = "bold"  # the user's own lines: bold behind a `›` mark
USER_BG = "on grey23"  # ...on a full-width grey band, like Claude Code (back on 2026-09-18)
SEL_STYLE = "bg:#264f78"  # a mouse selection over the transcript: Claude Code's dark-theme selectionBg, rgb(38, 79, 120)
PREFIX_W = BUDDY_WIDTH  # Pip's input gutter width
# The transcript's two-cell gutter, as Claude Code and Codex draw theirs: `› ` before what the user
# said, `● ` before Vision's prose, `◉ ` while the mic is open; tool rows carry their own `⏺`.
USER_MARK, REPLY_MARK, HEAR_MARK = "› ", "● ", "◉ "
GUTTER = len(REPLY_MARK)
GUTTER_MARKS = {USER_MARK, REPLY_MARK, HEAR_MARK}
NOTICE_MARKS = {"info": ("· ", "dim"), "warn": ("⚠ ", "yellow"), "error": ("✗ ", "red")}

MODE_BADGES = {"auto": "⏵⏵ auto", "plan": "⏸ plan"}  # what the bottom-left corner shows

# A lone Esc byte could be the start of an escape sequence, so prompt_toolkit holds it for
# `ttimeoutlen` (0.5 s by default) before treating it as the Escape key. Terminals deliver whole
# sequences in one read, so a short wait is plenty and Esc closes menus without the lag.
ESC_TIMEOUT = 0.05


def _snappy(app):
    """Set the short Esc wait on an Application and hand it back."""
    app.ttimeoutlen = ESC_TIMEOUT
    return app


PT_STYLE = Style.from_dict({
    "frame.border": "ansicyan",
    "frame.label": "ansicyan",
    "frame.corner": "ansicyan",  # the model name in the input box's bottom-right border
    "placeholder": "#6b7580 italic",
    "queued": "#c9d1d9",  # the strip of messages waiting behind a reply, over the input box
    "queued.hint": "#6b7580 italic",
    "queued.sel": "ansicyan bold",
    "queued.edit": "#6b7580",
    "status": "#8a939c",
    "status.key": "#c9d1d9 bold",
    # mode indicator, bottom left (Shift-Tab)
    "mode.auto": "ansiyellow bold",
    "mode.plan": "ansibrightgreen bold",
    "pick.title": "bold",
    "pick.cursor": "ansicyan bold",
    "pick.selected": "ansicyan bold",
    "pick.hint": "#6b7580",
    "pick.desc": "#8a939c",
    "pick.num": "#6b7580",
    "pick.current": "ansicyan",
    "pick.tab": "#8a939c",
    "pick.tab.active": "bg:ansicyan #000000 bold",
    # Pip Obsidian (the inset face and animated cells supply their own flat colors).
    "buddy.body": "#899a90",
    "buddy.eye": "#c1ebd1",
    "buddy.mouth": "#c1ebd1",
    "buddy.key": "#d0e7d9",
    "buddy.think": "#dbcaa5",
    "buddy.listen": "#c1ebd1",
    "buddy.err": "#d8a191",
    "buddy.sleep": "#738378",
})

# ---------------------------------------------------------------- header
def show_header(
    console: Console, *, model: str, directory: str, voice: str, resumed: bool = False, extra: tuple[str, ...] = (),
) -> None:
    """Sets the terminal title, then the open card (see header_renderable). Extra dim lines sit under it."""
    console.set_window_title("Vision")
    console.print(header_renderable(model=model, directory=directory, voice=voice, resumed=resumed))
    for line in extra:
        if line:
            console.print(Text(line, style="dim"))


def _pip_mark() -> Text:
    """Still idle Pip, four rows, the same colours as the composer gutter."""
    rows, colors = idle_sprite()
    out = Text()
    for y, row in enumerate(rows):
        if y:
            out.append("\n")
        for x, char in enumerate(row):
            style = colors[y][x]
            if 1 <= y <= 2 and FACE_X0 <= x <= FACE_X1:
                style = f"{style} on {FACE_BG}"
            out.append(char, style=style)
    return out


def header_renderable(*, model: str, directory: str, voice: str, resumed: bool = False):
    """The open card: Pip on the left, `◆ Vision` and labeled model / directory / voice on the right,
    in a rounded cyan box sized to the content. First block in an empty transcript; /clear puts it back."""
    title = Text.assemble(("◆ Vision", f"bold {ACCENT}"))
    if resumed:
        title.append("   resumed", style="dim")
    facts = Text()
    facts.append_text(title)
    for name, value in (("model", model), ("directory", directory), ("voice", voice)):
        facts.append("\n")
        facts.append(f"{name}:".ljust(12), style="dim")
        facts.append(value)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(_pip_mark(), facts)
    return Panel(grid, box=ROUNDED, border_style=ACCENT, padding=(0, 1), expand=False)


# ---------------------------------------------------------------- input word wrap
class WordWrap(Processor):
    """Make the input box wrap between words instead of mid-word.

    prompt_toolkit wraps a line the moment the next character would cross the window's right
    edge, so it happily splits words. This walks the line the same way and, whenever a word
    would straddle the edge, pads the line with spaces up to the edge so the whole word drops
    down together. The padding is display-only: the position maps keep the cursor and mouse
    clicks on the real text, and because the padded line is exactly as wide as what gets
    drawn, the box's auto-height (see ChatScreen) still comes out right. Words wider than the
    window fall back to character wrapping, there is nothing better to do with those.
    """

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        width = ti.width
        chars = [(style, ch) for style, text, *_ in ti.fragments for ch in text]
        n = len(chars)
        if width <= 0 or n == 0:
            return Transformation(ti.fragments)

        out: list[tuple[str, str]] = []
        disp_of = [0] * (n + 1)  # display index of each source index (and of the end)
        x = pad_total = 0
        i = 0
        while i < n:
            if chars[i][1] != " ":
                # measure the word starting here; pad to the edge if it would be split
                j = i
                w = 0
                while j < n and chars[j][1] != " ":
                    w += get_cwidth(chars[j][1])
                    j += 1
                if 0 < x and x + w > width >= w:
                    out.append(("", " " * (width - x)))
                    pad_total += width - x
                    x = width
            style, ch = chars[i]
            cw = get_cwidth(ch)
            if x + cw > width:
                x = 0  # the window wraps here
            disp_of[i] = i + pad_total
            out.append((style, ch))
            x += cw
            i += 1
        disp_of[n] = n + pad_total
        if not pad_total:
            return Transformation(ti.fragments)

        from bisect import bisect_left

        def source_to_display(i: int) -> int:
            return disp_of[min(max(i, 0), n)]

        def display_to_source(d: int) -> int:
            return min(bisect_left(disp_of, d), n)

        return Transformation(out, source_to_display, display_to_source)


# ---------------------------------------------------------------- transcript rendering
def user_grid(text: str) -> Padding:
    """The user's message: bold behind a `›` mark, on a full-width highlighted band."""
    return Padding(_reply_grid(Text(text, style=USER_STYLE), USER_MARK), (0, 0), style=USER_BG, expand=True)


def reply_grid(text: str, markdown: bool = True) -> Group:
    """A finished reply as a static transcript block (a resumed conversation's earlier turns): the
    same `●` gutter as a live reply, with its blank line above and below."""
    return Group(Text(""), _reply_grid(ChatMarkdown(text) if markdown else Text(text), REPLY_MARK), Text(""))


def hearing_grid(text: str) -> Table:
    """The user's words as they are still being heard: a `◉` mark, the words so far in italics and
    a trailing ellipsis while the ears are catching up; `◉ listening…` before the first word."""
    body = Text(text, style="italic") if text else Text("listening", style="dim italic")
    body.append("…" if not text else " …", style="dim")
    return _reply_grid(body, HEAR_MARK)


def notice_grid(text: str, kind: str = "info"):
    """A system line in the transcript (a /command's answer, a warning, an error), told from the
    conversation by its mark: `·` dim for info, `⚠` yellow, `✗` red. A multi-line text hangs under
    the mark. Rich markup in `text` is honoured for info lines (the callers' `[yellow]…[/yellow]`)."""
    mark, style = NOTICE_MARKS.get(kind, NOTICE_MARKS["info"])
    if kind == "info":
        body = Text.from_markup(text, style="dim") if "[" in text else Text(text, style="dim")
    else:
        body = Text(text, style=style)
    return _reply_grid(body, mark, style)


def show_user(console: Console, text: str) -> None:
    console.print(user_grid(text))


def _reply_grid(content, mark: str = "", mark_style: str = ACCENT) -> Table:
    """A transcript block: a two-cell gutter holding `mark` on the first row (blank under it, so
    wrapped lines hang off the mark), then the content across the rest of the width. Without a mark
    the content is flush left."""
    grid = Table.grid(expand=True)
    if mark:
        grid.add_column(width=GUTTER, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(*([Text(mark, style=mark_style)] if mark else []), content)
    return grid


def agent_activity(runs, spin: str = "") -> Group:
    """The subagent rows, two per agent, the shape of a tool row (see tool_activity):
    `⏺ general-purpose(what it was asked)` then, under a `⎿`, how it is going. A spinner in front
    while it runs and `⎿  running · opus high · 12s · ↓1.2k` (the model the Agent call named, else
    the one it inherits; the effort is always inherited; the voice model's worker shows the tokens
    it has produced so far); a green dot once done and `⎿  done · opus high · 3 tools · 17.3s`
    (`↑12,340 ↓1,204` for the worker, which has usage of its own); a red dot and `failed` if it
    failed. The child's individual tool calls are not listed; the model only ever sees its final
    report. Lines are truncated, not wrapped, so a long label stays one row (see _one_row)."""
    text = _one_row()
    for run in list(runs):
        if run.done:
            text.append("⏺ ", style="red" if run.failed else "green")
        else:
            text.append((spin or "⏺") + " ", style=ACCENT)
        text.append(run.kind, style="bold")
        if run.label:
            text.append(f"({run.label})", style="dim")
        text.append("\n")
        bits = ["cut off" if run.cut_off else "failed" if run.failed else "done"] if run.done else ["running"]
        if run.model:  # which brain is spending (`Fable 5.1`, not the id a workflow reports)
            from vision.models import model_label

            bits.append(model_label(run.model) + (f" {run.effort}" if run.effort else ""))
        if run.done:
            bits.append(run.status)
        elif run.started:
            bits.append(clock(time.monotonic() - run.started, live=True))
            tokens = run.tokens_fn() if run.tokens_fn else getattr(run, "tokens", 0)
            if tokens:
                bits.append(f"↓{short_count(tokens)}")
        text.append("  ⎿  ", style="dim")
        text.append(" · ".join(bits) + "\n", style="red" if run.failed else "dim")
        for line in (run.details if run.done else [])[1:]:  # the first line repeats the state above
            text.append("     ", style="dim")
            text.append(line + "\n", style="red" if run.failed and line is run.details[1] else "dim")
    text.rstrip()
    return Group(text)


def _one_row() -> Text:
    """A Text whose lines are truncated with an ellipsis rather than wrapped. Hand it to the console
    inside a Group: `console.print(text)` folds Text objects into a fresh one and loses no_wrap."""
    return Text(no_wrap=True, overflow="ellipsis")


def tool_activity(calls, spin: str = "", expanded: set | None = None) -> Group:
    """The main conversation's tool calls, drawn as Claude Code draws its own: `⏺ Bash(git status)`
    (a spinner while it runs, a red dot when it failed), then under a `⎿` only how much came back,
    `12 lines · click to expand`, until a click puts the call's id in `expanded`, which shows the
    whole result (see _ReplyEntry.lines). None of the work is shown otherwise. Once the reply is
    done the block folds to one summary row (tool_fold). Lines are truncated, not wrapped, so a long
    command or output line stays one row."""
    text = _one_row()
    expanded = expanded or set()
    for call in list(calls):
        if not call.done:
            text.append((spin or "⏺") + " ", style=ACCENT)
        else:
            text.append("⏺ ", style="red" if call.is_error else "green")
        text.append(call.name, style="bold")
        if call.detail:
            text.append(f"({call.detail})", style="dim")
        text.append("\n")
        if call.done:
            lines = [ln for ln in call.output.splitlines() if ln.strip()]
            body_style = "red" if call.is_error else "dim"
            text.append("  ⎿  ", style="dim")
            if not lines:
                # Nothing to show or hide: say so on the result row and leave it at that.
                text.append(("(error)" if call.is_error else "(no output)") + "\n", style=body_style)
            elif call.id not in expanded:
                text.append(f"{len(lines)} line{'s' if len(lines) != 1 else ''} · click to expand\n", style="dim italic")
            else:
                for i, ln in enumerate(lines):
                    if i:
                        text.append("     ", style="dim")
                    text.append(ln.rstrip() + "\n", style=body_style)
                text.append("     click to show less\n", style="dim italic")
    text.rstrip()
    return Group(text)


_TOOL_VERBS = {  # how the folded line counts each tool: (verb, noun); tools sharing a pair are added together
    "Read": ("read", "file"), "Edit": ("edited", "file"), "Write": ("edited", "file"), "NotebookEdit": ("edited", "file"),
    "Bash": ("ran", "command"), "Grep": ("searched", "pattern"), "Glob": ("searched", "pattern"),
    "WebSearch": ("searched the web", "time"), "WebFetch": ("fetched", "page"), "Agent": ("ran", "agent"), "Task": ("ran", "agent"),
}


def tool_summary(calls) -> str:
    """The turn's tool calls and subagents in one line, as Claude Code folds its own once a turn is
    over: `Read 3 files, ran 2 commands, ran 5 agents`, in the order they were first used, and
    `, 1 failed` at the end if any errored. Empty without calls."""
    counts: dict[tuple[str, str], int] = {}
    failed = 0
    for call in list(calls):
        name = call.name if _is_tool(call) else "Agent"
        key = _TOOL_VERBS.get(name, ("called", name))
        counts[key] = counts.get(key, 0) + 1
        failed += bool(call.is_error if _is_tool(call) else call.failed)
    parts = []
    for (verb, noun), n in counts.items():
        s = "" if n == 1 else "s"
        parts.append(f"called {noun} {n} time{s}" if verb == "called" else f"{verb} {n} {noun}{s}")
    if failed:
        parts.append(f"{failed} failed")
    line = ", ".join(parts)
    return line[:1].upper() + line[1:]


def tool_fold(calls) -> Group:
    """The folded block: one row, `⏺ Read 3 files, ran 2 agents · click or ctrl-o to expand` (a red
    dot if any failed), shown in place of the tool and agent rows once the reply is done."""
    calls = list(calls)
    text = _one_row()
    text.append("⏺ ", style="red" if any(c.is_error if _is_tool(c) else c.failed for c in calls) else "green")
    text.append(tool_summary(calls), style="dim")
    text.append(" · click or ctrl-o to expand", style="dim italic")
    return Group(text)


def _is_tool(item) -> bool:
    """A ToolCall rather than an AgentRun (see vision.brain; ui does not import it)."""
    return hasattr(item, "output")


def split_at_marks(buf: str, marks: list) -> list[tuple]:
    """The reply in the order things happened: `("text", start, end)` for each run of prose between
    the marks (offsets into `buf`; blank runs dropped) and `("marks", [k, …])` for each run of marks
    (indexes into `marks`, whose entries are (offset, AgentRun | ToolCall)) with no prose between
    them. A mark past the end of `buf` is left out, and so is everything after it: ReplyView hands
    in the text cut to what the voice has said, and a subagent launched after that point has not
    happened yet as far as the listener knows."""
    out: list[tuple] = []
    at = 0
    for k, (off, _) in enumerate(marks):
        if off > len(buf):
            break
        if buf[at:off].strip():
            out.append(("text", at, off))
        if out and out[-1][0] == "marks":
            out[-1][1].append(k)
        else:
            out.append(("marks", [k]))
        at = off
    if buf[at:].strip():
        out.append(("text", at, len(buf)))
    return out


def word_cut(text: str, n: int) -> int:
    """Index just past the word that holds the n-th char, so a cut never splits a word."""
    if n >= len(text):
        return len(text)
    for i in range(max(n, 0), len(text)):
        if text[i].isspace():
            return i
    return len(text)


class _CodeBlock(CodeBlock):
    """A fenced block flush with the prose, as Claude Code draws it: no padding lines or columns
    (rich's default pads one of each, which also lands in a copied selection)."""

    def __rich_console__(self, console, options):
        yield Syntax(str(self.text).rstrip(), self.lexer_name, theme=self.theme, word_wrap=True, padding=0)


class _ListItem(ListItem):
    """`- item` and `1. item` flush with the prose, as Claude Code draws them (rich indents ` • `)."""

    def render_bullet(self, console, options):
        lines = console.render_lines(self.elements, options.update(width=options.max_width - 2), style=self.style)
        style = console.get_style("markdown.item.bullet", default="none")
        for i, line in enumerate(lines):
            yield Segment("- " if i == 0 else "  ", style)
            yield from line
            yield Segment("\n")

    def render_number(self, console, options, number, last_number):
        width = len(str(last_number)) + 2  # "1. "
        lines = console.render_lines(self.elements, options.update(width=options.max_width - width), style=self.style)
        style = console.get_style("markdown.item.number", default="none")
        for i, line in enumerate(lines):
            yield Segment(f"{number}.".rjust(width - 1) + " " if i == 0 else " " * width, style)
            yield from line
            yield Segment("\n")


class _Heading(Heading):
    """Headings left-aligned (rich centres an h1, which a copied selection would keep as spaces)."""

    def __rich_console__(self, console, options):
        text = self.text.copy()
        text.justify = "left"
        yield text


_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")  # a Windows path: C:/x.png or C:\x.png


class _Image(ImageItem):
    """`![caption](/abs/path.png)`: the caption and the path, one click away (a file:// link the terminal
    opens in the image viewer). The iPhone app draws the picture itself."""

    def __rich_console__(self, console, options):
        path = self.destination
        if _DRIVE_PATH.match(path):
            link = RichStyle(link="file:///" + path.replace("\\", "/"))
            caption = self.text.plain.strip() or path.replace("\\", "/").rsplit("/", 1)[-1]
        else:
            link = RichStyle(link=("file://" + path) if path.startswith("/") else path or None)
            caption = self.text.plain.strip() or path.rsplit("/", 1)[-1]
        yield Text.assemble(("🖼 ", "none"), (caption, link + RichStyle(bold=True)), ("  ", "none"), (path, link + RichStyle(dim=True)), end="")


class ChatMarkdown(Markdown):
    """Rich markdown laid out like Claude Code's replies, so what is shown (and copied) is the
    reply's own text: code and lists flush with the prose, single blank lines between blocks."""

    elements = {**Markdown.elements, "fence": _CodeBlock, "code_block": _CodeBlock, "list_item_open": _ListItem, "heading_open": _Heading, "image": _Image}


class ReplyView:
    """Live-updating reply, flush left.

    With `gate` (a callable giving how many source chars the voice has said so far, see
    StreamingSpeaker.spoken) the text is shown only up to that point, so it keeps step with the speech."""

    def __init__(self, console: Console, markdown: bool = True, gate: Callable[[], int] | None = None):
        self.console = console
        self.markdown = markdown
        self.gate = gate
        self._gate_at = -1  # gate value the last render used
        self._closed = False
        self.buf = ""
        self._partial_word = ""
        self.status = ""
        self.agents: dict = {}  # subagents this turn, by id (AgentRun, see vision.brain)
        self.marks: list[tuple[int, object]] = []  # (len(buf) when it started, AgentRun): where each one's rows sit in the text
        self._live = Live(_reply_grid(Text(""), REPLY_MARK), console=console, refresh_per_second=12, vertical_overflow="visible")

    def __enter__(self):
        self.console.print()  # a blank line between the user's band and the reply
        self._live.__enter__()
        if self.gate is not None:
            threading.Thread(target=self._follow, daemon=True).start()
        return self

    def _follow(self):
        """Redraw as the voice gets through the text (the brain's deltas alone would stop early)."""
        while not self._closed:
            time.sleep(0.05)
            if self.gate is not None and self.gate() != self._gate_at:
                self._render()

    def _render(self):
        buf = self.buf
        if self.gate is not None:
            self._gate_at = self.gate()
            buf = buf[: word_cut(buf, self._gate_at)]
        # In the order things happened: a `●` block per run of prose, the subagents' rows where they
        # were launched (flush left, so their dot lines up with the mark), a blank line between.
        blocks: list = []
        for part in split_at_marks(buf, self.marks):
            if part[0] == "text":
                piece = buf[part[1]:part[2]].strip("\n")
                block = _reply_grid(ChatMarkdown(piece) if self.markdown else Text(piece), REPLY_MARK)
            else:
                block = agent_activity([self.marks[k][1] for k in part[1]])
            blocks += [Text(""), block] if blocks else [block]
        if not blocks:
            blocks.append(_reply_grid(Text(""), REPLY_MARK))
        if self.status:
            blocks.append(Padding(Text(self.status, style="dim italic"), (0, 0, 0, GUTTER)))
        self._live.update(Group(*blocks))

    def set_status(self, text: str) -> None:
        self.status = text
        self._render()

    def set_agent(self, run) -> None:
        if run.id not in self.agents:
            self.marks.append((len(self.buf), run))
        self.agents[run.id] = run  # the brain mutates the same AgentRun as it goes; redraw it
        self._render()

    def ask(self, questions: list[dict]) -> dict[str, str] | None:
        """Pause the live reply, run the question form below it, print the answer under the reply
        and carry on in a fresh one (the answer is a block of its own, as in Claude Code)."""
        self.status = ""
        if self.buf.strip() or self.agents:
            self._render()
        else:
            self._live.update(Text(""))  # nothing said yet: nothing above the form
        self._live.stop()
        try:
            answers = ask_questions(questions)
        except BaseException:
            self._live.start()
            raise
        self.console.print(answered_grid(questions, answers))
        self.console.print()
        self.buf, self.agents, self.marks = "", {}, []
        self._render()
        self._live.start()
        return answers

    def append(self, delta: str) -> None:
        if not self.buf:
            delta = delta.lstrip()  # see ChatScreen.update_reply
        # Keep the in-flight word off-screen until its trailing boundary arrives. This makes
        # streamed replies land as readable words instead of flickering token fragments.
        text = self._partial_word + delta
        match = re.search(r"\s+$", text)
        if match:
            self.buf += text
            self._partial_word = ""
        else:
            cut = max(text.rfind(" "), text.rfind("\n"))
            if cut >= 0:
                self.buf += text[:cut + 1]
                self._partial_word = text[cut + 1:]
            else:
                self._partial_word = text
        self._render()

    def __exit__(self, *exc):
        self.status = ""
        self._closed = True
        self.gate = None  # whatever is left shows now
        if self._partial_word:
            self.buf += self._partial_word
            self._partial_word = ""
        if not self.buf.strip():
            self.buf = "…"
        self._render()
        self._live.__exit__(*exc)
        self.console.print()
        return False


# ---------------------------------------------------------------- input box
class ConversationHistory(History):
    """Up/Down walks this conversation's user messages.

    Nothing is written to disk: a resumed session is loaded from the provider's session files,
    and each new user turn (typed or spoken) is appended in memory.
    """

    def __init__(self, turns: list[str] | None = None):
        super().__init__()
        self._storage = list(turns or [])  # oldest first

    def load_history_strings(self):
        yield from reversed(self._storage)

    def store_string(self, string: str) -> None:
        self._storage.append(string)

    def replace(self, turns: list[str]) -> None:
        """Swap in another conversation's user turns (oldest first) and mark the cache loaded."""
        self._storage = list(turns)
        self._loaded_strings = list(reversed(self._storage))
        self._loaded = True


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
            height=Dimension(min=2, max=6),  # grows with the wrapped text, see ChatScreen
            dont_extend_height=True,
            accept_handler=None,
            input_processors=[WordWrap()],
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

        frame = Frame(self.area)
        status_win = Window(FormattedTextControl(lambda: FormattedText([("class:status", "  " + self.status_fn())])), height=1)
        self.app = _snappy(Application(
            layout=Layout(HSplit([frame, status_win]), focused_element=self.area),
            key_bindings=kb,
            style=PT_STYLE,
            mouse_support=False,
            erase_when_done=True,
        ))

    def read(self) -> str:
        """Blocks until Enter. Raises KeyboardInterrupt / EOFError to leave."""
        self.area.text = ""
        return self.app.run()


# ---------------------------------------------------------------- picker
def _as_tabs(options) -> list[tuple[str, list[tuple[str, str, str]], str]]:
    """Picker options are either flat rows [(value, label, desc)] or tabs [(name, rows)] /
    [(name, rows, note)]; always return (name, rows, note) tabs."""
    if options and isinstance(options[0][1], list):
        return [(t[0], list(t[1]), t[2] if len(t) > 2 else "") for t in options]
    return [("", list(options), "")]


def _locate(tabs, current: str, fallback: str = "") -> tuple[int, int]:
    """(tab, row) of `current`, or the named `fallback` tab, or (0, 0)."""
    if current:
        for t, (_, rows, _) in enumerate(tabs):
            for i, o in enumerate(rows):
                if o[0] == current:
                    return t, i
    if fallback:
        for t, (name, _, _) in enumerate(tabs):
            if name.lower() == fallback.lower():
                return t, 0
    return 0, 0


def _picker_lines(title: str, tabs, tab: int, idx: int, current: str, framed: bool = False) -> list[tuple[str, str]]:
    """The picker body. With several tabs the first line is a provider switcher (active tab
    highlighted, its note beside it). Rows are padded to the tallest tab so the box keeps its size
    when switching provider. `framed` drops the title line (the frame border carries it)."""
    out = [] if framed else [("class:pick.title", f" {title}\n")]
    label_w = max((len(r[1]) for _, rows, _ in tabs for r in rows), default=8) + 2
    if len(tabs) > 1:
        out.append(("", "  "))
        for t, (name, _, _) in enumerate(tabs):
            out.append(("class:pick.tab.active" if t == tab else "class:pick.tab", f" {name} "))
            out.append(("", " "))
        if tabs[tab][2]:
            out.append(("class:pick.desc", f"  {tabs[tab][2]}"))
        out.append(("", "\n"))
    rows = tabs[tab][1]
    for i, (val, label, desc) in enumerate(rows):
        cur = i == idx
        is_current = val == current
        out.append(("class:pick.cursor" if cur else "", f" {'❯' if cur else ' '} "))
        out.append(("class:pick.num", f"{i + 1} " if i < 9 else "  "))
        out.append(("class:pick.selected" if cur else "class:pick.current" if is_current else "", f"{label:<{label_w}}"))
        out.append(("class:pick.desc", desc))
        if is_current:
            out.append(("class:pick.current", "  (current)"))
        out.append(("", "\n"))
    out.extend([("", "\n")] * (_max_rows(tabs) - len(rows)))
    out.append(("class:pick.hint", ("  ←/→ provider · " if len(tabs) > 1 else "  ") + "↑/↓ move · 1-9 pick · Enter select · Esc cancel"))
    return out


def _max_rows(tabs) -> int:
    return max((len(rows) for _, rows, _ in tabs), default=0)


def _picker_height(tabs, framed: bool = False) -> int:
    return _max_rows(tabs) + 1 + (0 if framed else 1) + (1 if len(tabs) > 1 else 0)


def pick(title: str, options, current: str = "", prefer_tab: str = "") -> str | None:
    """Inline arrow-key selector. options = [(value, label, description)] or [(tab_name, rows)];
    with tabs, ←/→ switches between them. Returns value or None."""
    tabs = _as_tabs(options)
    tab, idx = _locate(tabs, current, prefer_tab)
    state = {"tab": tab, "i": idx}

    def rows():
        return tabs[state["tab"]][1]

    def render():
        return FormattedText(_picker_lines(title, tabs, state["tab"], state["i"], current))

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(e):
        n = len(rows())
        if n:
            state["i"] = (state["i"] - 1) % n

    @kb.add("down")
    @kb.add("j")
    def _down(e):
        n = len(rows())
        if n:
            state["i"] = (state["i"] + 1) % n

    def _switch(step: int):
        if len(tabs) < 2:
            return
        state["tab"] = (state["tab"] + step) % len(tabs)
        t, i = _locate(tabs, current)
        state["i"] = i if t == state["tab"] else 0

    @kb.add("left")
    @kb.add("h")
    @kb.add("s-tab")
    def _left(e):
        _switch(-1)

    @kb.add("right")
    @kb.add("l")
    @kb.add("tab")
    def _right(e):
        _switch(1)

    @kb.add("enter")
    def _ok(e):
        r = rows()
        if r:
            e.app.exit(result=r[state["i"]][0])

    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(e):
        e.app.exit(result=None)

    for n in range(1, 10):
        def _num(e, n=n):
            if n <= len(rows()):
                e.app.exit(result=rows()[n - 1][0])
        kb.add(str(n))(_num)

    app = _snappy(Application(
        layout=Layout(Window(FormattedTextControl(render), height=_picker_height(tabs), always_hide_cursor=True)),
        key_bindings=kb, style=PT_STYLE, erase_when_done=True,
    ))
    return app.run()


# ---------------------------------------------------------------- question form (AskUserQuestion)
class QuestionForm:
    """State for Claude's AskUserQuestion: one or more questions, each single- or multi-select,
    with a free-text "type your own" escape hatch. The chat screen overlays it above the input box
    (typed text comes from the box); `ask_questions()` wraps it in its own small application.
    Answers come out as {question: "Label"} or "Label A, Label B", the format Claude Code uses."""

    def __init__(self, questions: list[dict]):
        self.qs = [q for q in questions if q.get("options")] or [{"question": "?", "header": "", "options": [], "multiSelect": False}]
        self.q = 0
        self.idx = 0
        self.picks: list[set[int]] = [set() for _ in self.qs]
        self.typed: list[str] = ["" for _ in self.qs]
        self.done: list[bool] = [False for _ in self.qs]

    # -- state
    @property
    def cur(self) -> dict:
        return self.qs[self.q]

    @property
    def multi(self) -> bool:
        return bool(self.cur.get("multiSelect"))

    @property
    def options(self) -> list[dict]:
        return self.cur["options"]

    @property
    def own(self) -> int:
        """Index of the "type your own" row, which sits after the last option."""
        return len(self.options)

    def up(self) -> None:
        self.idx = (self.idx - 1) % (len(self.options) + 1)

    def down(self) -> None:
        self.idx = (self.idx + 1) % (len(self.options) + 1)

    def goto(self, step: int) -> None:
        self.q = (self.q + step) % len(self.qs)
        if self.typed[self.q] and not self.picks[self.q]:
            self.idx = self.own
        else:
            self.idx = min(self.picks[self.q] or {0}) if not self.multi else 0

    def typing(self, text: str) -> None:
        """Input box changed: any text drags the cursor onto the ✎ row (like Claude Code's "Other")."""
        if text.strip():
            self.idx = self.own

    def toggle(self, i: int | None = None) -> None:
        i = self.idx if i is None else i
        if i >= len(self.options):
            return  # the ✎ row is answered by typing, not ticking
        if self.multi:
            self.picks[self.q] ^= {i}
        else:
            self.picks[self.q] = {i}
            self.idx = i

    def answer_text(self, q: int) -> str:
        labels = [self.qs[q]["options"][i]["label"] for i in sorted(self.picks[q])]
        if self.typed[q]:
            labels.append(self.typed[q])
        return ", ".join(labels)

    def confirm(self, typed: str = "") -> bool:
        """Enter: settle the current question (cursor row for single-select, ticked rows for multi,
        plus any typed text). Advances to the next open question; True when every question is answered."""
        typed = typed.strip()
        if typed:
            self.typed[self.q] = typed
        if not self.multi and not typed and self.idx < self.own:
            self.picks[self.q] = {self.idx}
        if not self.picks[self.q] and not self.typed[self.q]:
            return False  # nothing chosen yet (Enter on an empty ✎ row is a no-op)
        self.done[self.q] = True
        if all(self.done):
            return True
        self.goto(next((k for k in range(1, len(self.qs) + 1) if not self.done[(self.q + k) % len(self.qs)]), 1))
        return False

    def answers(self) -> dict[str, str]:
        return {q["question"]: self.answer_text(i) for i, q in enumerate(self.qs)}

    # -- rendering
    def lines(self, framed: bool = False, typed: str = "") -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if len(self.qs) > 1:
            out.append(("", "  "))
            for i, q in enumerate(self.qs):
                name = q.get("header") or f"Q{i + 1}"
                out.append(("class:pick.tab.active" if i == self.q else "class:pick.tab", f" {'✓ ' if self.done[i] else ''}{name} "))
                out.append(("", " "))
            out.append(("class:pick.desc", f"  {self.q + 1}/{len(self.qs)}"))
            out.append(("", "\n"))
        out.append(("class:pick.title", f" {self.cur.get('question', '')}\n"))
        label_w = max((len(o.get("label", "")) for o in self.options), default=8) + 2
        picks = self.picks[self.q]
        for i, o in enumerate(self.options):
            cur = i == self.idx
            on = i in picks
            out.append(("class:pick.cursor" if cur else "", f" {'❯' if cur else ' '} "))
            out.append(("class:pick.num", f"{i + 1} " if i < 9 else "  "))
            if self.multi:
                out.append(("class:pick.current" if on else "class:pick.num", "◉ " if on else "○ "))
            out.append(("class:pick.selected" if cur else "class:pick.current" if on else "", f"{o.get('label', ''):<{label_w}}"))
            out.append(("class:pick.desc", o.get("description", "")))
            out.append(("", "\n"))
        own = typed.strip() or self.typed[self.q]
        cur = self.idx == self.own
        out.append(("class:pick.cursor" if cur else "", f" {'❯' if cur else ' '} "))
        out.append(("class:pick.num", "✎ "))
        if own:
            out.append(("class:pick.selected" if cur else "class:pick.current", own + ("▏" if cur and typed.strip() else "")))
        else:
            out.append(("class:pick.selected" if cur else "class:pick.desc", "Type your own answer…" if cur else "Other (type your own)"))
        out.append(("", "\n"))
        keys = "↑/↓ move · " + ("Space tick · " if self.multi else "") + "1-9 pick · type for ✎ · Enter " + ("next · " if len(self.qs) > 1 else "confirm · ")
        keys += ("←/→ question · " if len(self.qs) > 1 else "") + "Esc cancel"
        out.append(("class:pick.hint", "  " + keys))
        return out

    def height(self, framed: bool = False) -> int:
        return len(self.options) + 3 + (1 if len(self.qs) > 1 else 0)


def ask_questions(questions: list[dict]) -> dict[str, str] | None:
    """Standalone AskUserQuestion form (for `vision ask` / talk mode). Returns answers or None."""
    form = QuestionForm(questions)
    box = TextArea(multiline=False, prompt=[("class:frame.label", "› ")], height=1)
    box.buffer.on_text_changed += lambda _buf: form.typing(box.text)
    kb = KeyBindings()
    empty = Condition(lambda: not box.text)

    @kb.add("up")
    def _up(e):
        form.up()

    @kb.add("down")
    def _down(e):
        form.down()

    @kb.add("left")
    @kb.add("s-tab")
    def _prev(e):
        form.goto(-1)

    @kb.add("right")
    @kb.add("tab")
    def _next(e):
        form.goto(1)

    @kb.add(" ", filter=empty)
    def _space(e):
        if form.multi:
            form.toggle()
        else:
            e.app.current_buffer.insert_text(" ")

    @kb.add("enter")
    def _ok(e):
        if form.confirm(box.text):
            e.app.exit(result=form.answers())
        box.text = ""

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _cancel(e):
        e.app.exit(result=None)

    for n in range(1, 10):
        def _num(e, n=n):
            if n <= len(form.options):
                form.toggle(n - 1)
                if not form.multi and form.confirm():
                    e.app.exit(result=form.answers())
        kb.add(str(n), filter=empty)(_num)

    body = HSplit([
        Window(FormattedTextControl(lambda: FormattedText(form.lines(typed=box.text))), height=lambda: form.height(), always_hide_cursor=True),
        box,
    ])
    app = _snappy(Application(
        layout=Layout(Frame(body, title=[("class:frame.label", " Vision asks ")]), focused_element=box),
        key_bindings=kb, style=PT_STYLE, erase_when_done=True,
    ))
    return app.run()


def answered_grid(questions: list[dict], answers: dict[str, str] | None):
    """What a settled question form leaves in the transcript, the way Claude Code records an
    answered AskUserQuestion: a headline, then one dim `· question → answer` line per question
    hanging off a ⎿ hook. A dismissed form lists the questions with their options instead, so
    the transcript still shows what was asked."""
    if answers:
        head = f"You answered Vision's question{'s' if len(answers) > 1 else ''}:"
        rows = [f"· {q} → {a}" for q, a in answers.items()]
    else:
        head = f"You declined to answer Vision's question{'s' if len(questions) > 1 else ''}"
        rows = [f"· {q.get('question', '')} ({' / '.join(o.get('label', '') for o in q.get('options', []))})" for q in questions]
    grid = Table.grid(padding=0, expand=True)
    grid.add_column(width=5, no_wrap=True)  # "  ⎿  " on the first row, blank under it so wrapped lines keep the indent
    grid.add_column(ratio=1, overflow="fold")
    for i, row in enumerate(rows):
        grid.add_row(Text("  ⎿  " if i == 0 else "", style="dim"), Text(row, style="dim"))
    return Group(Text.assemble("● ", head), grid)


def short_path(p: str) -> str:
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home) else p


# ---------------------------------------------------------------- slash-command menu
from dataclasses import dataclass

MENU_ROWS = 8  # visible rows in the slash menu before it scrolls


@dataclass
class SlashCommand:
    """One /command for the menu that pops up as you type. `args` (optional) returns
    (value, description) choices for the first argument, shown after '/name '."""

    name: str
    desc: str
    args: Callable[[], list[tuple[str, str]]] | None = None
    aliases: tuple[str, ...] = ()


@dataclass
class MenuRow:
    text: str  # what the input becomes when this row is chosen ("/model" or "/model sonnet")
    label: str
    desc: str
    is_arg: bool = False


def slash_menu_rows(commands: list[SlashCommand], text: str) -> list[MenuRow]:
    """Rows for the input `text`: commands matching '/na…' (prefix matches first, then substring),
    or the argument choices of a fully typed command after '/name '. Empty when no menu applies."""
    if not text.startswith("/") or "\n" in text:
        return []
    name, sp, arg = text[1:].partition(" ")
    if not sp:
        needle = name.lower()
        prefix = [c for c in commands if c.name.startswith(needle) or any(a.startswith(needle) for a in c.aliases)]
        inner = [c for c in commands if c not in prefix and needle and needle in c.name]
        return [MenuRow("/" + c.name, "/" + c.name, c.desc) for c in prefix + inner]
    cmd = next((c for c in commands if c.name == name.lower() or name.lower() in c.aliases), None)
    if not cmd or not cmd.args or " " in arg:
        return []
    needle = arg.lower()
    choices = cmd.args()
    prefix = [(v, d) for v, d in choices if v.lower().startswith(needle)]
    inner = [(v, d) for v, d in choices if needle and needle in v.lower() and (v, d) not in prefix]
    return [MenuRow(f"/{cmd.name} {v}", v, d, is_arg=True) for v, d in prefix + inner]


def _menu_lines(rows: list[MenuRow], idx: int) -> list[tuple[str, str]]:
    """The menu body, windowed around the cursor when there are more rows than MENU_ROWS."""
    start = max(0, min(idx - MENU_ROWS // 2, len(rows) - MENU_ROWS))
    shown = rows[start : start + MENU_ROWS]
    label_w = max((len(r.label) for r in shown), default=8) + 2
    out: list[tuple[str, str]] = []
    for i, r in enumerate(shown, start):
        cur = i == idx
        out.append(("class:pick.cursor" if cur else "", f" {'❯' if cur else ' '} "))
        out.append(("class:pick.selected" if cur else "", f"{r.label:<{label_w}}"))
        out.append(("class:pick.desc", r.desc))
        out.append(("", "\n"))
    if out:
        out.pop()  # no trailing newline: the window is sized to the rows
    return out


# ---------------------------------------------------------------- full-screen chat
import bisect
import io
import math
import re
import threading
import time

from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEventType

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TITLE_SPINNER = "✢✳✶✻✽✻✶✳"  # the terminal tab's "working" mark, Claude Code's glyphs; slow so Ptyxis is not redrawn all day
WORKING = "working…"  # the status while the brain is busy and has said nothing about what it is doing; the live line rotates it through WORKING_VERBS
WORKING_VERBS = ("working…", "on it…", "still on it…")  # content-free on purpose: "thinking…" is shown only while a reasoning block streams
VERB_SECONDS = 4  # how long each verb stays up


def short_count(n: int) -> str:
    """`842`, `1.2k`, `13k`, `1.1M`: a token count the width of a word."""
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    return f"{n / 1_000_000:.1f}M"
WHEEL_LINES = 3  # transcript lines per mouse-wheel notch


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
        self._lines: dict[int, list[list]] = {}

    def ansi(self, width: int) -> str:
        if width not in self._cache:
            body = Group(Text(""), self.renderable) if self.gap_before else self.renderable
            self._cache[width] = render_ansi(body, width)
        return self._cache[width]

    def lines(self, width: int) -> list[list]:
        """The entry as one prompt_toolkit fragment list per line, parsed once per width so a
        long transcript is neither re-parsed nor re-laid-out on every frame."""
        if width not in self._lines:
            self._lines[width] = _rows(self.ansi(width))
        return self._lines[width]

    def invalidate(self):
        self._cache.clear()
        self._lines.clear()


_OSC8 = re.compile(r"\x1b\]8;([^;\x07\x1b]*);([^\x07\x1b]*)(?:\x1b\\|\x07)")
_CSI = re.compile(r"(\x1b\[[0-9;]*[A-Za-z])")
LINK_OFF = "\x1b]8;;\x1b\\"


def _linked(chunk: str, link: str) -> str:
    if not link:
        return chunk
    parts = _CSI.split(chunk)
    for i in range(0, len(parts), 2):  # even parts are text, odd ones the SGR codes between
        parts[i] = "".join(ch if ch == "\n" else f"\001{link}\002{ch}" for ch in parts[i])
    return "".join(parts)


def _links_to_escapes(ansi: str) -> str:
    """rich's OSC 8 hyperlinks as prompt_toolkit zero-width escapes. Its ANSI parser knows no OSC:
    it drops the ESC and prints the rest (`8;id=…;file://…`). Every linked character carries its
    own opening, so a repaint that starts mid-link still links; the run's end closes it, and
    _close_links shuts it whenever the renderer jumps, so a partial repaint can't spill it."""
    if "\x1b]8;" not in ansi:
        return ansi
    out, link, pos = [], "", 0
    for m in _OSC8.finditer(ansi):
        out.append(_linked(ansi[pos:m.start()], link))
        pos = m.end()
        was, link = link, (m.group(0) if m.group(2) else "")
        if was and not link:
            out.append(f"\001{LINK_OFF}\002")
    out.append(_linked(ansi[pos:], link))
    return "".join(out).replace("\002\001", "")  # the parser takes a \001 straight after \002 as text


def _close_links(output):
    """Close any open hyperlink before every cursor jump and attribute reset (see _links_to_escapes)."""
    for name in ("reset_attributes", "cursor_goto", "cursor_up", "cursor_down", "cursor_forward", "cursor_backward"):
        def closing(*args, _method=getattr(output, name), **kwargs):
            output.write_raw(LINK_OFF)
            return _method(*args, **kwargs)
        setattr(output, name, closing)
    return output


def _rows(ansi: str) -> list[list]:
    """Parsed ANSI as one prompt_toolkit fragment list per line (rich's trailing newline dropped).
    prompt_toolkit's ANSI parser emits one fragment per character; runs of the same style are
    merged here, which makes every later per-frame pass over the fragments ~10x cheaper."""
    rows = []
    for row in split_lines(ANSI(_links_to_escapes(ansi)).__pt_formatted_text__()):
        merged: list = []
        for frag in row:
            if merged and merged[-1][0] == frag[0] and len(frag) == 2:
                merged[-1] = (frag[0], merged[-1][1] + frag[1])
            else:
                merged.append(frag)
        rows.append(merged)
    if ansi.endswith("\n"):
        rows.pop()
    return rows


def _vis(frag) -> int:
    """Visible characters in a fragment: blanks and zero-width escapes are free, so a code block's
    background row fills as one and the count is width-independent for prose."""
    if "[ZeroWidthEscape]" in frag[0]:
        return 0
    return len(frag[1]) - frag[1].count(" ")


@lru_cache(maxsize=512)
def _reveal_style(style: str, level: int) -> str:
    """Ease the incoming glyph up to its normal colour over sixteen small steps."""
    foreground, background = "#c9d1d9", "#20252b"
    for token in style.split():
        if token.startswith("bg:#"):
            background = token[3:]
        elif token.startswith(("#", "fg:#")):
            foreground = token.removeprefix("fg:")
        elif token.startswith("ansi") and token != "ansidefault":
            name = token[4:]
            foreground = "bright_" + name[6:] if name.startswith("bright") else name
            if foreground == "gray":
                foreground = "white"
    fg = Color.parse(foreground).get_truecolor()
    bg = Color.parse(background).get_truecolor()
    amount = (level / 16) ** 0.7
    rgb = tuple(round(a + (b - a) * amount) for a, b in zip(bg, fg))
    return f"{style} fg:#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class _Reveal:
    """A rendered reply, parsed into fragment lines once, that can be cut after its first n visible
    characters. Cutting is a list slice plus one truncated fragment, cheap enough for every frame."""

    def __init__(self, ansi: str, src_len: int):
        self.rows = _rows(ansi)
        self.src_len = src_len  # len(buf) this was rendered from
        self.at = time.monotonic()
        self._before: list[int] = []  # visible chars ahead of each counted fragment
        self._where: list[tuple[int, int]] = []  # (row, column) of that fragment
        total = 0
        for r, row in enumerate(self.rows):
            for c, frag in enumerate(row):
                v = _vis(frag)
                if v:
                    self._before.append(total)
                    self._where.append((r, c))
                    total += v
        self.total = total

    def cut(self, n: float) -> list[list]:
        if n <= 0 or self.total == 0:
            return []
        if n >= self.total:
            return self.rows
        whole = int(n)
        fraction = n - whole
        end = math.ceil(n)
        i = bisect.bisect_left(self._before, end) - 1  # fragment holding the last visible char
        r, c = self._where[i]
        row = self.rows[r]
        style, text = row[c][0], row[c][1]
        k, j = end - self._before[i], 0
        while k:
            if text[j] != " ":
                k -= 1
            j += 1
        if fraction:
            # Only the incoming glyph changes colour; completed text keeps its original
            # styles and layout. Fractional speech progress gives us intermediate frames
            # even when the voice only advances a dozen characters per second.
            edge = _reveal_style(style, max(1, min(16, round(fraction * 16))))
            return [*self.rows[:r], [*row[:c], (style, text[:j - 1]), (edge, text[j - 1:j])]]
        if not text[j:].strip() and not any(_vis(f) for f in row[c + 1:]):
            return self.rows[: r + 1]  # rest of the line is blank: take the whole row (keeps its background)
        return [*self.rows[:r], [*row[:c], (style, text[:j])]]


class _Segments:
    """The rendered reply in the order things happened: a _Reveal per run of prose, with the
    subagent and tool rows (the entry's marks, drawn live by _ReplyEntry.lines) slotted in where
    they arrived. The typewriter's count runs through the prose pieces in order, so a mark's rows
    come into view once the text before them is out, not before."""

    def __init__(self, buf: str, marks: list, markdown: bool, width: int):
        self.src_len = len(buf)  # len(buf) this was rendered from...
        self.nmarks = len(marks)  # ...and how many marks it knew of
        self.at = time.monotonic()
        self.parts: list[tuple] = []  # ("text", start, end, _Reveal, visible chars before it) | ("marks", [k, …], chars before)
        total = 0
        for part in split_at_marks(buf, marks):
            if part[0] == "text":
                a, b = part[1], part[2]
                piece = buf[a:b].strip("\n")  # the "\n\n" the brain puts between text blocks means nothing at the top of a piece
                reveal = _Reveal(render_ansi(ChatMarkdown(piece) if markdown else Text(piece), width), b - a)
                self.parts.append(("text", a, b, reveal, total))
                total += reveal.total
            else:
                self.parts.append(("marks", part[1], total))
        self.total = total

    def stale(self, buf: str, marks: list) -> bool:
        return self.src_len != len(buf) or self.nmarks != len(marks)

    def reach(self, said: float) -> float:
        """Rendered visible chars covered by the first `said` source chars: each piece's in
        proportion, which is close enough for a lag of a word or two."""
        out = 0.0
        for part in self.parts:
            if part[0] == "text":
                _, a, b, reveal, _ = part
                if said >= b:
                    out += reveal.total
                elif said > a:
                    out += reveal.total * (said - a) / (b - a)
        return out

    def cut(self, n: float) -> list[tuple]:
        """The first n visible chars: `("text", rows)` per run of prose that has begun (see
        _Reveal.cut) and `("marks", [k, …])` for each run of marks the count has reached."""
        out: list[tuple] = []
        for part in self.parts:
            if part[0] == "text":
                rows = part[3].cut(n - part[4])
                if rows:
                    out.append(("text", rows))
            elif n >= part[2]:
                out.append(("marks", part[1]))
        return out


_ROW_CACHE: dict[tuple, list[list]] = {}


def _styled_rows(text: str, style: str, width: int = 40) -> list[list]:
    key = (text, style, width)
    if key not in _ROW_CACHE:
        _ROW_CACHE[key] = _rows(render_ansi(Text(text, style=style), width))
    return _ROW_CACHE[key]


class _ReplyEntry(_Entry):
    """A streaming reply. The whole received text is rendered (markdown and all) and the typewriter
    reveals that *rendered* output, so nothing reflows as bold, lists or code fences resolve: the
    pacer lag doubles as look-ahead, and the text simply appears in its final shape."""

    def __init__(self, markdown: bool):
        super().__init__(None)
        self.markdown = markdown
        self.buf = ""
        self.shown = 0.0  # fractional progress also animates the incoming glyph between letters
        self.status = WORKING  # spinner line in the transcript: "working…" / "thinking…" / "using Bash…"; "" while text flows
        self.agents: dict = {}  # subagents this turn, by id (AgentRun), see agent_activity
        self.tools: dict = {}  # the main conversation's tool calls, by id (ToolCall), see tool_activity
        self.marks: list[tuple[int, object]] = []  # (len(buf) when it arrived, AgentRun | ToolCall): where each one's rows sit in the text, see place
        self.expanded: set[str] = set()  # tool calls whose whole result is shown (a click on the row toggles it)
        self.unfolded = False  # the done reply's tool calls are back to one row each (a click on the folded line, or ctrl-o)
        self.on_click: Callable[[], None] | None = None  # the screen's redraw, called after a toggle
        self.finished = False  # the brain has stopped sending
        self.done = False  # ...and everything is revealed
        self.started = time.monotonic()  # the timer under the reply runs from here...
        self.ended: float | None = None  # ...until the brain is done (the typewriter tail is not counted)
        self.cancelled = False  # Esc froze this reply before the brain finished
        self.gate: Callable[[], float] | None = None  # source chars said, including fractional progress
        self.tokens_fn: Callable[[], int] | None = None  # output tokens received so far (the `↓ 1.2k` on the live line); None = unknown
        self.footer = ""  # the done line under the reply: `4.8s`, or `4.8s · cancelled`
        self._full: tuple[int, _Segments] | None = None  # (width, render) of the whole buffer
        self._render_lock = threading.Lock()
        self._spin: tuple = ()  # (spinner frame, timer text) the cached lines were built with
        self._line_progress = -1.0
        self._floor: dict[int, int] = {}  # per width: most rows shown so far (never shrink mid-reply)

    def full(self, width: int, fresh: bool = True) -> _Segments:
        """The rendered whole buffer at `width`, in pieces around the marks. With fresh=False a
        render of an older buffer is handed back as is (the pacer refreshes on its own schedule,
        see _pace)."""
        with self._render_lock:
            cur = self._full[1] if self._full and self._full[0] == width else None
            if cur is None or (fresh and cur.stale(self.buf, self.marks)):
                cur = _Segments(self.buf, list(self.marks), self.markdown, max(20, width - GUTTER - 1))
                self._full = (width, cur)
            return cur

    def place(self, item) -> None:
        """A subagent (AgentRun) or a tool call (ToolCall) of this reply, slotted in at the point in
        the text where it arrived, so the transcript keeps the order things happened: a model that
        says "sending an agent off" and then launches one reads that way, not the other way round.
        The brain mutates the same object as it goes, so a later call with it is just a redraw."""
        table = self.tools if _is_tool(item) else self.agents
        if item.id not in table:
            self.marks.append((len(self.buf), item))
        table[item.id] = item

    def reveal_all(self) -> None:
        self.shown = 1 << 30

    def _click(self, action: Callable[[], None]) -> Callable:
        """A mouse handler for a row: `action` on a click (the button coming up), then a redraw."""

        def handler(mouse_event):
            if mouse_event.event_type != MouseEventType.MOUSE_UP:
                return NotImplemented  # presses, drags and the wheel are not clicks
            action()
            # A deliberate collapse is a new baseline, not a reflow bounce to pad over (see the floor
            # in lines): left in place, the padding would fill the bottom-anchored view with blank
            # rows until the turn ended.
            self._floor.clear()
            self.invalidate()
            if self.on_click:
                self.on_click()
            return None

        return handler

    def _toggle(self, call_id: str) -> Callable:
        return self._click(lambda: self.expanded.__ixor__({call_id}))

    def _mark_rows(self, ks: list[int], spin: str, inner: int) -> list[list]:
        """The rows of a run of marks with no prose between them: each subagent's two rows and each
        tool call's block, every fragment of a tool's carrying a mouse handler (the third element)
        so a click on any of them shows or hides the whole result. Once the reply is done the run
        folds into one summary row (tool_fold) that a click unfolds, with the way back under the
        last run when they are unfolded on purpose."""
        rows: list[list] = []
        if self.folded:
            handler = self._click(self.toggle_fold)
            for row in _rows(render_ansi(tool_fold([self.marks[k][1] for k in ks]), inner)):
                rows.append([(style, txt, handler) for style, txt, *_ in row])
            return rows
        for k in ks:
            item = self.marks[k][1]
            if not _is_tool(item):
                rows += _rows(render_ansi(agent_activity([item], spin), inner))
                continue
            handler = self._toggle(item.id)
            for row in _rows(render_ansi(tool_activity([item], spin, self.expanded), inner)):
                rows.append([(style, txt, handler) for style, txt, *_ in row])
        if self.done and ks[-1] == len(self.marks) - 1:
            handler = self._click(self.toggle_fold)
            for row in _styled_rows("     click or ctrl-o to fold", "dim italic", inner):
                rows.append([(style, txt, handler) for style, txt, *_ in row])
        return rows

    @property
    def folded(self) -> bool:
        """Once the reply is done its tool calls and subagents fold into one summary row per run of
        them, as Claude Code's do, until a click on such a row or ctrl-o unfolds them (see tool_fold)."""
        return bool(self.marks) and self.done and not self.unfolded

    def toggle_fold(self) -> None:
        """Ctrl-o: unfold the done reply's tool calls and subagents, or fold them again."""
        if not self.marks or not self.done:
            return
        self.unfolded = not self.unfolded
        self._floor.clear()
        self.invalidate()
        if self.on_click:
            self.on_click()

    @property
    def elapsed(self) -> float:
        """Seconds the brain has been (or was) working on this reply."""
        return (self.ended if self.ended is not None else time.monotonic()) - self.started

    def live_label(self) -> str:
        """What the live line says: the status as given (`using Bash…`, `waiting for your answer…`),
        except that the bare `working…` rotates through Vision's own verbs every few seconds, as
        Claude Code's do, so a long wait visibly keeps moving."""
        running = sum(not r.done for r in self.agents.values())
        if running and self.status in (WORKING, "using Agent…", "using Task…"):
            # Subagents at work: their rows (and timers) may have scrolled off above the streaming
            # text, so the live line at the bottom says the wait is on them.
            return f"waiting on {running} agents…" if running > 1 else "waiting on an agent…"
        if self.status != WORKING:
            return self.status
        return WORKING_VERBS[int(self.elapsed // VERB_SECONDS) % len(WORKING_VERBS)]

    def lines(self, width: int) -> list[list]:
        animating = not self.done or any(not r.done for r in self.agents.values()) or any(not c.done for c in self.tools.values())
        spin = SPINNER[int(time.time() * 8) % len(SPINNER)] if animating else ""
        timer = f"{self.elapsed:.1f}s" if self.ended is None else ""
        shown = self.shown
        if (spin, timer) != self._spin or shown != self._line_progress:
            self._lines.clear()  # animate the spinner and the timer (the app repaints every refresh_interval)
            self._spin = (spin, timer)
            self._line_progress = shown
        if width not in self._lines:
            inner = max(20, width - 1)
            mark = _styled_rows(REPLY_MARK, f"bold {ACCENT}")[0]
            rows: list[list] = []
            for kind, payload in self.full(width, fresh=self.done).cut(shown):
                if kind == "text":
                    # The `●` gutter: the mark on the first row of each run of prose, two blanks under it.
                    block = [[*mark, *payload[0]], *[[("", " " * GUTTER), *row] for row in payload[1:]]]
                else:
                    # The subagents' and tool calls' rows where they happened, flush left so their
                    # dot lines up with the mark.
                    block = self._mark_rows(payload, spin, inner)
                rows += [[], *block] if rows else block  # a blank line between prose and the rows
            if self.ended is None:
                # Live line while the brain works: `⠋ having a look… · 3.2s · ↓ 1.2k · esc to stop`, or
                # just the timer while text flows. Not through _ROW_CACHE: it changes ten times a second.
                label = self.live_label()
                parts = [f"{spin} {label}" if label else spin, timer]
                got = self.tokens_fn() if self.tokens_fn else 0
                if got:
                    parts.append(f"↓ {short_count(got)}")
                parts.append("esc to stop")
                rows = [*rows, *_rows(render_ansi(Text(" · ".join(parts), style="dim italic"), inner))]
            elif self.footer and (self.done or self.cancelled):
                rows = [*rows, *_styled_rows(" " * GUTTER + self.footer, "dim", inner)]
            elif not rows:
                rows = _styled_rows("…", "dim")
            if not self.done:
                # A re-render of the longer buffer can shorten the already-revealed part by a row (a code
                # fence closing, a table gaining a row, a status line clearing). Pad instead of shrinking,
                # or the bottom-anchored transcript bounces up and down as the text streams.
                floor = self._floor.get(width, 0)
                if len(rows) < floor:
                    rows = [*rows, *([[]] * (floor - len(rows)))]
                self._floor[width] = len(rows)
            out = [[], *rows]  # a blank line between the user's band and the reply, as under it
            if self.done:
                out.append([])
            self._lines[width] = out
        return self._lines[width]


def copy_to_clipboard(text: str, output=None) -> str:
    """Put `text` on the system clipboard: wl-copy on Wayland, xclip or xsel on X11, otherwise the
    OSC 52 escape through the terminal (works over SSH where the terminal allows it). Returns the
    way it went, "" if none worked."""
    import shutil
    import subprocess

    if sys.platform == "win32":
        from vision.compat import copy_text_windows

        try:
            if copy_text_windows(text):
                return "clipboard"
        except OSError:
            pass
    cmds = []
    if os.environ.get("WAYLAND_DISPLAY"):
        cmds.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        cmds += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in cmds:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return cmd[0]
            except Exception:  # noqa: BLE001
                continue
    if output is not None:
        try:
            output.write_raw("\x1b]52;c;" + base64.b64encode(text.encode()).decode() + "\x07")
            output.flush()
            return "osc52"
        except Exception:  # noqa: BLE001
            pass
    return ""


def _cells(text: str) -> list[int]:
    """Cumulative cell width before each char of `text`, plus the total at the end."""
    out, w = [0], 0
    for ch in text:
        w += get_cwidth(ch)
        out.append(w)
    return out


def _slice_row(row: list, c0: int, c1: int, style: str | None = None) -> tuple[list, str]:
    """The row with cells [c0, c1) restyled (style appended, e.g. "reverse") and the text of those
    cells. Fragments are split at the boundaries; a fragment's mouse handler stays on its parts."""
    out: list = []
    picked: list[str] = []
    at = 0
    for frag in row:
        st, txt, *rest = frag
        if "[ZeroWidthEscape]" in st:  # a link's escape code: no cells, not part of the copied text
            out.append(frag)
            continue
        cells = _cells(txt)
        end = at + cells[-1]
        if end <= c0 or at >= c1:
            out.append(frag)
        else:
            # first char index at or after c0, and first char index at or after c1
            i0 = next((i for i, c in enumerate(cells) if at + c >= c0), len(txt))
            i1 = next((i for i, c in enumerate(cells) if at + c >= c1), len(txt))
            if i0 > 0:
                out.append((st, txt[:i0], *rest))
            if i1 > i0:
                out.append((f"{st} {style}" if style else st, txt[i0:i1], *rest))
                picked.append(txt[i0:i1])
            if i1 < len(txt):
                out.append((st, txt[i1:], *rest))
        at = end
    return out, "".join(picked)


def dedent_rows(rows: list[tuple[int, str]]) -> str:
    """Selected rows as (cell column they start at, their text) → the text without the transcript's
    margin. The leftmost column where content starts, over the non-blank rows, becomes column 0:
    a reply's paragraphs lose any margin while a code block keeps its indentation
    relative to the prose around it. Blank rows are blank lines; blank rows at the ends are dropped."""
    starts = [col + len(text) - len(text.lstrip(" ")) for col, text in rows if text.strip()]
    if not starts:
        return ""
    edge = min(starts)
    out = []
    for col, text in rows:
        lead = len(text) - len(text.lstrip(" "))
        out.append(text[min(lead, max(0, edge - col)):])
    return "\n".join(out).strip("\n")


def _sel_cols(sel, line: int) -> tuple[int, int] | None:
    """The [c0, c1) cell span of `line` inside selection `sel` ((line, col) first and last cells,
    inclusive); None when the line is outside it. Middle lines are whole."""
    (l0, c0), (l1, c1) = sel
    if line < l0 or line > l1:
        return None
    return (c0 if line == l0 else 0, c1 + 1 if line == l1 else 1 << 30)


class _TranscriptControl(FormattedTextControl):
    """The transcript's control: the wheel scrolls, a left drag selects (the screen's on_mouse gets
    every event first; when it declines, a click goes to the fragment's own handler, if any)."""

    def __init__(self, scroll_by: Callable[[int], None], on_mouse: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self._scroll_by = scroll_by
        self._on_mouse = on_mouse

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_by(-WHEEL_LINES)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_by(WHEEL_LINES)
            return None
        if self._on_mouse is not None and self._on_mouse(mouse_event) is None:
            return None
        return super().mouse_handler(mouse_event)


PLACEHOLDER = "Message Vision…"
QUIT_WINDOW = 2.0  # seconds a second idle Ctrl-C has to arrive in to quit (the first one only warns)


def _private_output():
    """The screen's own handle on the terminal: a duplicate of stdout's descriptor, so that a worker
    thread sending the process's fds 1/2 to /dev/null for a while (tts._quiet around the voice model
    load, on the first spoken reply) does not blank the screen and leave it stale when they return."""
    from prompt_toolkit.output import create_output

    try:
        tty = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding=sys.stdout.encoding or "utf-8", errors="replace")
        return _close_links(create_output(stdout=tty))
    except (OSError, ValueError, AttributeError):
        return None  # not a real terminal (tests, pipes): prompt_toolkit's default output


class CornerFrame:
    """The input box's frame with a label set into its bottom-right border, the way the Grok CLI
    names its model there: `└──── Sonnet 5 (high) ─┘`. `corner` is read on every repaint; an empty
    string draws a plain border. Same border characters and styles as prompt_toolkit's Frame."""

    def __init__(self, body, corner: Callable[[], str]):
        from functools import partial

        fill = partial(Window, style="class:frame.border")
        self.body = body
        self.corner_fn = corner
        label = Label(
            lambda: [("class:frame.corner", f" {text} " if (text := self.corner_fn() or "") else "")],
            style="class:frame.border", dont_extend_width=True,
        )
        self.container = HSplit(
            [
                VSplit([fill(width=1, height=1, char=Border.TOP_LEFT), fill(char=Border.HORIZONTAL), fill(width=1, height=1, char=Border.TOP_RIGHT)], height=1),
                VSplit([fill(width=1, char=Border.VERTICAL), body, fill(width=1, char=Border.VERTICAL)], padding=0),
                VSplit(
                    [
                        fill(width=1, height=1, char=Border.BOTTOM_LEFT),
                        fill(char=Border.HORIZONTAL),
                        label,
                        fill(width=1, height=1, char=Border.HORIZONTAL),
                        fill(width=1, height=1, char=Border.BOTTOM_RIGHT),
                    ],
                    height=1,
                ),
            ],
            style="class:frame",
        )

    def __pt_container__(self):
        return self.container


class ChatScreen:
    """Claude-Code-style layout: scrolling transcript on top, framed input pinned to the bottom,
    status line under it (the current mode at its left edge). Runs full-screen until `exit()`; voice
    modes run on worker threads inside it, so `run()` only returns ("quit",)."""

    def __init__(self, history_path: str, status: Callable[[], str], buddy: Buddy | None = None, mode: Callable[[], str] | None = None, status_right: Callable[[], str] | None = None, corner: Callable[[], str] | None = None):
        # history_path is unused: Up/Down is this conversation (ConversationHistory), not a global file.
        self.status_fn = status
        self.corner_fn = corner  # set into the input box's bottom-right border: the model typed input goes to
        self.status_right_fn = status_right  # pinned to the bottom-right corner of the status row (the wake-word switch)
        self.mode_fn = mode  # "auto" | "plan"; None hides the indicator
        self.buddy = buddy
        self.entries: list[_Entry] = []
        self.busy = False
        self.busy_label = ""
        self.scroll_top: int | None = None  # first visible transcript line; None = follow the bottom
        self._view = (0, 0, 0)  # (top line, window height, line count) as of the last render
        self._visible_rows = 0
        # Mouse selection over the transcript: [(line, col) pressed, (line, col) now] in transcript
        # coordinates, copied when the button comes up (or the drag pauses, in case it comes up
        # outside the transcript). The highlight stays after release, as Claude Code's does, until
        # Ctrl-C copies it again and drops it, or the next press starts over.
        self._sel: list[tuple[int, int]] | None = None
        self._sel_dragged = False
        self._sel_copied = ""
        self._sel_timer: threading.Timer | None = None
        self._notice: tuple[str, float] = ("", 0.0)  # a short status-row message and when it expires
        self._quit_armed = 0.0  # when an idle Ctrl-C on an empty box last happened; a second within QUIT_WINDOW quits
        # Messages sent while a reply runs, oldest first: shown in a dim strip over the input box until
        # their turn starts (or Ctrl-X sends them into the running one). See queue / unqueue.
        self.turns = None  # vision.turnqueue.TurnQueue, set by the chat (cli.py)
        self.can_steer_fn: Callable[[], bool] = lambda: False  # can Ctrl-X send into the running reply?
        self._qsel: int | None = None  # the strip row ↑ picked (index among the shown messages)
        self._qedit: dict | None = None  # the message pulled into the box to edit: {"item", "index"}
        self.on_submit: Callable[[str], None] = lambda text: None
        # Ctrl-X mid-reply: send the message into the running turn now rather than queue it behind
        # it ("" = send the queued ones now). Without a reply running it is a plain send.
        self.on_steer: Callable[[str], None] = lambda text: self.on_submit(text) if text else None
        self.on_cancel: Callable[[], None] = lambda: None
        self.on_toggle_mode: Callable[[], None] = lambda: None  # Shift-Tab: auto ⇄ plan
        # Esc / Ctrl-C while nothing is being replied (e.g. to leave a voice conversation); True = consumed
        self.on_interrupt: Callable[[], bool] = lambda: False
        self.placeholder = PLACEHOLDER
        # Keep the transcript at a human pace.  Some providers genuinely stream tokens while
        # others (notably Codex exec) send a completed message in one lump; treating both as a
        # fast dump makes the second one feel like a pasted answer rather than Vision talking.
        # 42 cps is roughly conversational speech once spaces and punctuation are included.
        self.reveal_cps = 42  # spoken replies only; typed replies reveal as they arrive (see _pace)
        self.catch_up = 2.5  # target drain time for a burst, subject to the cap below
        self.max_reveal_cps = 72  # never race ahead of the voice just because a reply arrived whole
        self._revealing: _ReplyEntry | None = None
        self._picker = None  # dict(title, options, current, idx, cb)
        self._form: QuestionForm | None = None  # AskUserQuestion overlay; answers go to _form_cb
        self._form_cb: Callable[[dict[str, str] | None], None] = lambda a: None
        self._lock = threading.Lock()
        self._last_stream_render = 0.0
        # Slash menu: pops up while the input starts with "/", filters as you type. Set `commands`.
        self.commands: list[SlashCommand] = []
        # Terminal tab: `Vision - <conversation title>` with a spinning mark in front while a reply is
        # in progress (see _sync_title). Set `title_fn` to the conversation's current title ("" = none).
        self.title_fn: Callable[[], str] | None = None
        self._title = ""
        self._menu_idx = 0
        self._menu_moved = False  # cursor moved by hand → Enter picks the row even for arguments
        self._menu_dismissed: str | None = None  # Esc hides the menu until the text changes
        self._menu_cache: tuple[str, list[MenuRow]] = ("", [])

        # No fixed height: with min/max only, prompt_toolkit sizes the box from the *wrapped*
        # text, so long lines grow it just like Ctrl-J newlines do (dont_extend keeps it snug).
        self.area = TextArea(
            multiline=True, wrap_lines=True, history=ConversationHistory(),
            prompt=[("class:frame.label", "› ")],
            height=Dimension(min=2, max=8), dont_extend_height=True,
            input_processors=[WordWrap()],
        )
        form_open = Condition(lambda: self._form is not None)
        picker_open = Condition(lambda: self._picker is not None)
        menu_open = Condition(lambda: self._picker is None and self._form is None and bool(self._menu_rows()))
        overlay = picker_open | form_open
        kb = KeyBindings()

        @kb.add("enter", filter=~overlay & ~menu_open)
        def _send(event):
            self._submit(self.area.text)

        @kb.add("c-j")
        def _newline(event):
            self.area.buffer.insert_text("\n")

        @kb.add("c-x", filter=~overlay & ~menu_open)
        def _send_now(event):
            self._submit(self.area.text, now=True)

        # The queued strip: ↑ on an empty box picks a waiting message (the queue holds meanwhile).
        picking = Condition(lambda: self._qsel is not None)
        editing = Condition(lambda: self._qedit is not None)
        can_pick = Condition(lambda: self._qsel is None and self._qedit is None and not self.area.text and bool(self.queued))

        @kb.add("up", filter=~overlay & ~menu_open & can_pick)
        def _pick_queued(event):
            self._queue_pick()

        @kb.add("up", filter=picking)
        def _queued_up(event):
            self._queue_move(-1)

        @kb.add("down", filter=picking)
        def _queued_down(event):
            self._queue_move(1)

        @kb.add("enter", filter=picking)
        def _queued_edit(event):
            self._queue_edit()

        @kb.add("delete", filter=picking)
        @kb.add("backspace", filter=picking)
        def _queued_remove(event):
            self._queue_remove()

        @kb.add("c-x", filter=picking)
        def _queued_now(event):
            self._queue_send_now()

        @kb.add("escape", eager=True, filter=picking)
        def _queued_leave(event):
            self._queue_done()

        @kb.add("escape", eager=True, filter=editing & ~overlay & ~menu_open)
        def _queued_keep(event):
            self._queue_edit_finish(None)

        @kb.add("escape", eager=True, filter=~overlay & ~menu_open & ~picking & ~editing)
        def _esc(event):
            if self.busy:
                self._cancel()
            else:
                self.on_interrupt()

        @kb.add("s-tab", filter=~overlay)
        def _toggle_mode(event):
            self.on_toggle_mode()

        @kb.add("c-o", filter=~overlay & ~menu_open)
        def _fold(event):
            self.toggle_tools()

        # slash-menu keys
        @kb.add("up", filter=menu_open)
        @kb.add("c-p", filter=menu_open)
        def _menu_up(event):
            self._menu_idx = (self._menu_idx - 1) % len(self._menu_rows())
            self._menu_moved = True

        @kb.add("down", filter=menu_open)
        @kb.add("c-n", filter=menu_open)
        def _menu_down(event):
            self._menu_idx = (self._menu_idx + 1) % len(self._menu_rows())
            self._menu_moved = True

        @kb.add("tab", filter=menu_open)
        def _menu_fill(event):
            row = self._menu_rows()[self._menu_idx]
            cmd = self._command_named(row.text[1:])
            self._set_text(row.text + (" " if not row.is_arg and cmd and cmd.args else ""))

        @kb.add("enter", filter=menu_open)
        def _menu_choose(event):
            rows = self._menu_rows()
            row = rows[self._menu_idx]
            text = self.area.text
            _, _, arg = text[1:].partition(" ")
            if row.is_arg and not arg and not self._menu_moved:
                self._submit(text)  # bare "/model " → the command itself (its picker), not the first choice
            else:
                self._submit(row.text)

        @kb.add("escape", filter=menu_open, eager=True)
        def _menu_close(event):
            self._menu_dismissed = self.area.text

        @kb.add("c-c", filter=~overlay)
        def _ctrl_c(event):
            if self._sel_range():
                # A highlighted selection takes the key (like Claude Code): copy it, drop the highlight,
                # and leave the busy/quit logic alone, so it neither cancels a reply nor arms the exit.
                self._sel_copied = ""  # already copied on release: copy again so the notice shows
                self._copy_selection()
                self._sel = None
                self.app.invalidate()
                return
            if self.busy:
                self._cancel()
            elif self.on_interrupt():
                pass
            elif self.area.text:
                self.area.text = ""
            elif time.monotonic() - self._quit_armed < QUIT_WINDOW:
                event.app.exit(result=("quit",))
            else:
                # Like Claude Code / Grok: one stray Ctrl-C only warns, the second one leaves.
                self._quit_armed = time.monotonic()
                self.notice("Press Ctrl-C again to exit", QUIT_WINDOW)

        @kb.add("c-d")
        def _eof(event):
            if not self.area.text:
                event.app.exit(result=("quit",))

        @kb.add("pageup")
        def _pgup(event):
            self.scroll_by(-self._page())

        @kb.add("pagedown")
        def _pgdn(event):
            self.scroll_by(self._page())

        @kb.add("s-up")
        def _line_up(event):
            self.scroll_by(-1)

        @kb.add("s-down")
        def _line_down(event):
            self.scroll_by(1)

        @kb.add("c-home")
        @kb.add("escape", "home")
        def _top(event):
            self.scroll_top = 0

        @kb.add("c-end")
        @kb.add("escape", "end")
        def _bottom(event):
            self.scroll_top = None

        # picker keys
        @kb.add("up", filter=picker_open)
        @kb.add("k", filter=picker_open)
        def _pk_up(event):
            p = self._picker
            n = len(self._picker_rows())
            if n:
                p["idx"] = (p["idx"] - 1) % n

        @kb.add("down", filter=picker_open)
        @kb.add("j", filter=picker_open)
        def _pk_down(event):
            p = self._picker
            n = len(self._picker_rows())
            if n:
                p["idx"] = (p["idx"] + 1) % n

        def _pk_switch(step: int):
            p = self._picker
            if len(p["tabs"]) < 2:
                return
            p["tab"] = (p["tab"] + step) % len(p["tabs"])
            t, i = _locate(p["tabs"], p["current"])
            p["idx"] = i if t == p["tab"] else 0

        @kb.add("left", filter=picker_open)
        @kb.add("h", filter=picker_open)
        @kb.add("s-tab", filter=picker_open)
        def _pk_left(event):
            _pk_switch(-1)

        @kb.add("right", filter=picker_open)
        @kb.add("l", filter=picker_open)
        @kb.add("tab", filter=picker_open)
        def _pk_right(event):
            _pk_switch(1)

        @kb.add("enter", filter=picker_open)
        def _pk_ok(event):
            rows = self._picker_rows()
            if not rows:
                return
            p, self._picker = self._picker, None
            p["cb"](rows[p["idx"]][0])

        @kb.add("escape", filter=picker_open, eager=True)
        @kb.add("c-c", filter=picker_open)
        @kb.add("q", filter=picker_open)
        def _pk_cancel(event):
            self._picker = None

        for n in range(1, 10):
            def _num(event, n=n):
                p = self._picker
                rows = self._picker_rows()
                if p and n <= len(rows):
                    self._picker = None
                    p["cb"](rows[n - 1][0])
            kb.add(str(n), filter=picker_open)(_num)

        # question-form keys (letters still reach the input box: that is the "type your own" path)
        box_empty = Condition(lambda: not self.area.text)

        @kb.add("up", filter=form_open)
        def _fm_up(event):
            self._form.up()

        @kb.add("down", filter=form_open)
        def _fm_down(event):
            self._form.down()

        @kb.add("left", filter=form_open & box_empty)
        @kb.add("s-tab", filter=form_open)
        def _fm_prev(event):
            self._form.goto(-1)

        @kb.add("right", filter=form_open & box_empty)
        @kb.add("tab", filter=form_open)
        def _fm_next(event):
            self._form.goto(1)

        @kb.add(" ", filter=form_open & box_empty)
        def _fm_space(event):
            if self._form.multi:
                self._form.toggle()
            else:
                self.area.buffer.insert_text(" ")

        @kb.add("enter", filter=form_open)
        def _fm_ok(event):
            if self._form.confirm(self.area.text):
                self._finish_form()
            self.area.text = ""

        @kb.add("escape", filter=form_open, eager=True)
        @kb.add("c-c", filter=form_open)
        def _fm_cancel(event):
            self._finish_form(cancel=True)

        for n in range(1, 10):
            def _fnum(event, n=n):
                f = self._form
                if f and n <= len(f.options):
                    f.toggle(n - 1)
                    if not f.multi and f.confirm():
                        self._finish_form()
            kb.add(str(n), filter=form_open & box_empty)(_fnum)

        self.area.buffer.on_text_changed += self._on_text_changed
        menu_win = ConditionalContainer(
            Window(FormattedTextControl(self._menu_text, focusable=False), height=lambda: min(MENU_ROWS, len(self._menu_rows())) or 1),
            filter=menu_open,
        )
        queued_win = ConditionalContainer(
            Window(FormattedTextControl(self._queued_text, focusable=False), height=lambda: len(self._queued_lines()) or 1),
            filter=Condition(lambda: bool(self.queued) or self._qedit is not None),
        )
        self.transcript = Window(
            _TranscriptControl(self.scroll_by, self._mouse, text=self._transcript_text, get_cursor_position=self._cursor, show_cursor=False, focusable=False),
            wrap_lines=False, always_hide_cursor=True,  # rich already wrapped at the width; one line = one row keeps the scroll maths exact
        )
        frame = CornerFrame(self.area, self.corner_fn) if self.corner_fn else Frame(self.area)
        status_win = Window(FormattedTextControl(self._status_text), height=1)
        # content-width window at the far right (the wake switch); the main status
        # row takes whatever is left
        right_win = Window(FormattedTextControl(self._status_right_text, focusable=False), height=1, dont_extend_width=True)
        status_win = VSplit([status_win, right_win])
        picker_win = ConditionalContainer(
            Frame(
                Window(FormattedTextControl(self._picker_text), height=lambda: _picker_height(self._picker["tabs"], framed=True) if self._picker else 1),
                title=lambda: [("class:frame.label", f" {self._picker['title']} ")] if self._picker else "",
            ),
            filter=picker_open,
        )
        form_win = ConditionalContainer(
            Frame(
                Window(FormattedTextControl(self._form_text), height=lambda: self._form.height() if self._form else 1),
                title=[("class:frame.label", " Vision asks ")],
            ),
            filter=form_open,
        )
        composer = frame
        if self.buddy:
            # Pip owns a 16-column gutter on the bottom-left, beside the input box (the transcript
            # above is flush left). He stays bottom-aligned while the input grows;
            # narrow terminals fold him into the status row instead (see _status_text).
            buddy_win = Window(
                FormattedTextControl(lambda: self.buddy.render(self.busy), focusable=False),
                width=Dimension.exact(BUDDY_WIDTH), height=Dimension.exact(BUDDY_HEIGHT), dont_extend_width=True,
            )
            filler = Window(height=Dimension(preferred=0))
            wide = Condition(lambda: self._width() >= BUDDY_MIN_COLUMNS)
            buddy_gutter = ConditionalContainer(HSplit([filler, buddy_win]), filter=wide)
            composer = VSplit([buddy_gutter, frame])
            self.area.buffer.on_text_changed += lambda _buf: self.buddy.touch()
        bottom = HSplit([composer, status_win])  # the status row runs the full width, under Pip too
        body = HSplit([self.transcript, picker_win, form_win, menu_win, queued_win, bottom])
        self.app = _snappy(Application(
            layout=Layout(body, focused_element=self.area),
            # Mouse on: the wheel scrolls the transcript and a click on a tool row expands its
            # result. The terminal's own drag-to-select then needs Shift held (VTE/Ptyxis convention).
            key_bindings=kb, style=PT_STYLE, full_screen=True, mouse_support=True,
            refresh_interval=0.10, min_redraw_interval=1 / 60,
            max_render_postpone_time=0, output=_private_output(),
        ))

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

    def _page(self) -> int:
        return max(1, self._view[1] - 2)

    # -- scrolling
    def scroll_by(self, lines: int) -> None:
        """Move the transcript view by `lines` (negative = up). Reaching the bottom resumes following."""
        top, _, _ = self._view
        self.scroll_top = max(0, top + lines)  # clamped to the bottom (→ follow) at the next render
        self.app.invalidate()

    # -- rendering
    def _transcript_text(self):
        """Only the visible slice of the transcript: prompt_toolkit lays out every line it is
        given on every frame, so feeding it the whole history made frames slower as the chat
        grew (65 ms at ten messages). We do the scrolling arithmetic ourselves instead."""
        w = self._width()
        with self._lock:
            blocks = [e.lines(w) for e in self.entries]
        n = sum(len(b) for b in blocks)
        info = self.transcript.render_info
        h = info.window_height if info else self._transcript_height()
        max_top = max(0, n - h)
        if self.scroll_top is not None and self.scroll_top >= max_top:
            self.scroll_top = None
        top = max_top if self.scroll_top is None else min(self.scroll_top, max_top)
        frags: list = []
        rows = 0
        i = 0  # transcript line number at the start of the current block
        sel = self._sel_range()
        for b in blocks:
            if i + len(b) > top and rows < h:
                for row in b[max(0, top - i): max(0, top - i) + h - rows]:
                    if rows:
                        frags.append(("", "\n"))
                    span = _sel_cols(sel, top + rows) if sel else None
                    frags.extend(_slice_row(row, *span, SEL_STYLE)[0] if span else row)
                    rows += 1
            i += len(b)
        self.transcript.vertical_scroll = 0
        self._view = (top, h, n)
        self._visible_rows = rows
        return frags

    def _cursor(self):
        return Point(x=0, y=max(0, self._visible_rows - 1))

    # -- mouse selection → clipboard
    def _sel_range(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """The selection as (first, last) inclusive (line, col) cells, whichever way it was dragged."""
        if not self._sel or not self._sel_dragged:
            return None
        a, b = self._sel
        return (a, b) if a <= b else (b, a)

    def _mouse(self, ev):
        """Left press starts a selection (dropping any earlier one), drag extends it, release
        copies it and leaves it highlighted. A press-and-release without a drag is a click: nothing
        is selected and the row's own handler gets it (returns NotImplemented)."""
        if ev.button != MouseButton.LEFT:
            return NotImplemented
        top, _h, _n = self._view
        here = (top + ev.position.y, ev.position.x)
        if ev.event_type == MouseEventType.MOUSE_DOWN:
            self._sel, self._sel_dragged, self._sel_copied = [here, here], False, ""
            self.app.invalidate()
            return None
        if ev.event_type == MouseEventType.MOUSE_MOVE and self._sel:
            if here != self._sel[1]:
                self._sel[1] = here
                self._sel_dragged = True
                if self._sel_timer:
                    self._sel_timer.cancel()
                self._sel_timer = threading.Timer(0.4, self._copy_selection)  # the button may come up elsewhere
                self._sel_timer.daemon = True
                self._sel_timer.start()
                self.app.invalidate()
            return None
        if ev.event_type == MouseEventType.MOUSE_UP:
            dragged = bool(self._sel and self._sel_dragged)
            if dragged:
                self._copy_selection()
            if self._sel_timer:
                self._sel_timer.cancel()
                self._sel_timer = None
            if not dragged:
                self._sel = None
            self.app.invalidate()
            return None if dragged else NotImplemented
        return NotImplemented

    def selected_text(self) -> str:
        """What the selection covers, one line per transcript row, without the transcript's margin
        (see dedent_rows): what you get is the reply's own text, as copying from Claude Code gives."""
        sel = self._sel_range()
        if not sel:
            return ""
        w = self._width()
        with self._lock:
            lines = [row for e in self.entries for row in e.lines(w)]
        rows = []
        for ln in range(sel[0][0], min(sel[1][0], len(lines) - 1) + 1):
            c0, c1 = _sel_cols(sel, ln)
            text = _slice_row(lines[ln], c0, c1)[1].rstrip()
            if c0 == 0 and text[:GUTTER] in GUTTER_MARKS:
                text = " " * GUTTER + text[GUTTER:]  # the `›` / `●` gutter is layout, not the message: dedent_rows drops it with the margin
            rows.append((c0, text))
        return dedent_rows(rows)

    def _copy_selection(self) -> None:
        text = self.selected_text()
        if not text or text == self._sel_copied:
            return
        self._sel_copied = text
        how = copy_to_clipboard(text, self.app.output)
        n = text.count("\n") + 1
        self.notice(f"copied {n} line{'s' if n != 1 else ''}" if how else "could not reach the clipboard (no wl-copy/xclip)")

    def notice(self, msg: str, seconds: float = 2.5) -> None:
        """A short message on the status row (the row repaints on the refresh interval, so it fades by itself)."""
        self._notice = (msg, time.time() + seconds)
        self.app.invalidate()

    def _sync_title(self) -> None:
        """Keep the terminal's title current. Called on every repaint (the app redraws every
        refresh_interval), so the spinner turns while busy and a title Claude names after the fact
        shows up on its own; the escape sequence is only written when the text changes."""
        name = ""
        if self.title_fn:
            try:
                name = self.title_fn() or ""
            except Exception:
                name = ""
        title = f"Vision - {name}" if name else "Vision"
        if self.busy:
            title = f"{TITLE_SPINNER[int(time.time() * 4) % len(TITLE_SPINNER)]} {title}"
        if title != self._title:
            self._title = title
            try:
                self.app.output.set_title(title)
            except Exception:
                pass

    def _notice_active(self) -> bool:
        return bool(self._notice[0]) and time.time() < self._notice[1]

    def _status_text(self):
        self._sync_title()
        if self._notice_active():
            # A notice takes the whole row: the usual status line is long enough that a message
            # appended to its end can land off-screen, and the point is that it gets seen.
            return FormattedText([("class:status", "  "), ("class:status.key", self._notice[0])])
        parts = [("class:status", "  ")]
        if self.mode_fn:
            mode = self.mode_fn()
            parts += [(f"class:mode.{mode}", MODE_BADGES.get(mode, mode)), ("class:status", "  ")]
        parts.append(("class:status", self.status_fn()))
        if self.busy and self.busy_label:  # thinking/tool progress lives in the transcript (see _ReplyEntry)
            spin = SPINNER[int(time.time() * 8) % len(SPINNER)]
            parts.append(("class:status.key", f"   {spin} {self.busy_label}"))
        if self.scroll_top is not None:
            top, h, n = self._view
            parts.append(("class:status", f"   ↑ {max(0, n - top - h)} lines below · PgDn / Alt-End to follow"))
        if self.buddy and self._width() < BUDDY_MIN_COLUMNS:
            parts.append(("", "   "))
            parts.extend(self.buddy.render_inline(self.busy))
        return FormattedText(parts)

    def _status_right_text(self):
        """The right end of the status row: `wake on/off` (no key hints: the keys are in /help)."""
        if self._notice_active() or not self.status_right_fn:
            return FormattedText([])  # a notice owns the row (see _status_text)
        text = self.status_right_fn()
        return FormattedText([("class:status" if text.endswith("off") else "class:status.key", text), ("class:status", "  ")])

    # -- slash menu
    def _menu_rows(self) -> list[MenuRow]:
        text = self.area.text
        if not self.commands or text == self._menu_dismissed:
            return []
        if self._menu_cache[0] != text:
            self._menu_cache = (text, slash_menu_rows(self.commands, text))
        return self._menu_cache[1]

    @property
    def queued(self) -> list[str]:
        """The messages waiting behind the running reply, oldest first (the strip's rows)."""
        return [it.text for it in self.turns.shown()] if self.turns else []

    QUEUED_ROWS = 4  # strip rows shown at once; the rest fold into `+ 2 more` around the picked one

    def _queued_lines(self) -> list[tuple[str, str]]:
        items = self.queued
        rows: list[tuple[str, str]] = [("class:queued", t) for t in items]
        if self._qedit is not None:
            at = min(self._qedit["index"], len(rows))
            rows.insert(at, ("class:queued.edit", "✎ in the box below"))
        width = max(20, self._width() - 16)
        sel = self._qsel
        first = 0 if sel is None else max(0, min(sel - self.QUEUED_ROWS + 1, len(rows) - self.QUEUED_ROWS))
        lines = []
        if first:
            lines.append(("class:queued.hint", f"            + {first} more"))
        for i, (style, text) in enumerate(rows[first:first + self.QUEUED_ROWS], start=first):
            one = " ".join(text.split())
            one = one[: width - 1] + "…" if len(one) > width else one
            mark = "›" if i == sel else "⏸"
            lines.append(("class:queued.sel" if i == sel else style, f"  {mark} queued  {one}"))
        if len(rows) > first + self.QUEUED_ROWS:
            lines.append(("class:queued.hint", f"            + {len(rows) - first - self.QUEUED_ROWS} more"))
        if self._qedit is not None:
            hint = "Enter puts it back · empty removes it · Esc keeps the original · queue held while you edit"
        elif sel is not None:
            hint = "↑↓ pick · Enter edit · Del remove" + (" · Ctrl-X send now" if self.can_steer_fn() else "") + " · Esc done · queue held"
        else:
            now = ("Ctrl-X sends them into the reply now" if len(items) > 1 else "Ctrl-X sends it into the reply now") if self.can_steer_fn() else "goes when this reply ends"
            hint = f"{now} · ↑ to edit"
        lines.append(("class:queued.hint", f"            {hint}"))
        return lines

    def _queued_text(self):
        lines = self._queued_lines()
        return FormattedText([(style, line + ("\n" if i < len(lines) - 1 else "")) for i, (style, line) in enumerate(lines)])

    # -- picking and editing queued messages (the queue is held meanwhile, see vision.turnqueue)
    def _queue_pick(self) -> None:
        """↑ on an empty box: into the strip, on the newest message."""
        if self.turns and self.queued:
            self.turns.hold()
            self._qsel = len(self.queued) - 1
            self.app.invalidate()

    def _queue_move(self, step: int) -> None:
        n = len(self.queued)
        if self._qsel is None or not n:
            return self._queue_done()
        sel = self._qsel + step
        if sel >= n:
            return self._queue_done()  # ↓ past the last one: back to the box
        self._qsel = max(0, sel)
        self.app.invalidate()

    def _queue_done(self) -> None:
        """Out of the strip: the queue carries on."""
        self._qsel = None
        if self._qedit is None and self.turns:
            self.turns.release()
        self.app.invalidate()

    def _queue_selected(self):
        shown = self.turns.shown() if self.turns else []
        return shown[self._qsel] if self._qsel is not None and 0 <= self._qsel < len(shown) else None

    def _queue_remove(self) -> None:
        item = self._queue_selected()
        if item is not None and self.turns.take(item.id) is not None:
            self.notice("removed from the queue", 2)
        if not self.queued:
            return self._queue_done()
        self._qsel = min(self._qsel or 0, len(self.queued) - 1)
        self.app.invalidate()

    def _queue_edit(self) -> None:
        """Enter on a picked message: into the box to edit; the queue stays held until it goes back."""
        item = self._queue_selected()
        index = self._qsel or 0
        if item is None or self.turns.take(item.id) is None:
            self.notice("that one has already gone", 2)
            return self._queue_done()
        self._qedit = {"item": item, "index": index}
        self._qsel = None
        self.area.buffer.set_document(Document(item.text, len(item.text)), bypass_readonly=True)
        self.app.invalidate()

    def _queue_send_now(self) -> None:
        item = self._queue_selected()
        if item is None or self.turns.take(item.id) is None:
            return self._queue_done()
        self._queue_done()
        self.on_steer(item.text)

    def _queue_edit_finish(self, text: str | None) -> None:
        """Enter (the new text; empty drops it) or Esc (None: the original) while editing a queued message."""
        edit, self._qedit = self._qedit, None
        if edit is None:
            return
        item = edit["item"]
        if text is None or text.strip():
            if text is not None:
                item.text = text.strip()
            self.turns.insert(edit["index"], item)
        else:
            self.notice("removed from the queue", 2)
        self.area.buffer.reset()
        self.turns.release()
        self.app.invalidate()

    def _menu_text(self):
        rows = self._menu_rows()
        self._menu_idx = min(self._menu_idx, max(0, len(rows) - 1))
        return FormattedText(_menu_lines(rows, self._menu_idx))

    def _command_named(self, spec: str) -> SlashCommand | None:
        name = spec.split(" ", 1)[0].lower()
        return next((c for c in self.commands if c.name == name or name in c.aliases), None)

    def _on_text_changed(self, _buf) -> None:
        if self._qsel is not None and self.area.text:
            self._queue_done()  # typing: back to the box, the queue carries on
        if self._form:
            self._form.typing(self.area.text)
        self._menu_idx = 0
        self._menu_moved = False
        if self._menu_dismissed is not None and self.area.text != self._menu_dismissed:
            self._menu_dismissed = None

    def _set_text(self, text: str) -> None:
        self.area.buffer.document = Document(text, len(text))

    def _submit(self, text: str, now: bool = False) -> None:
        """Send the input (a message, or a /command which is allowed even mid-reply).

        A live reply has its own entry in ``_revealing``.  Messages submitted while that entry is
        active are follow-ups; the chat driver serialises them behind the current turn, unless
        ``now`` (Ctrl-X) sends them into it.  ``busy`` is also used for operations which cannot
        accept a message (model switches, usage lookups and voice warm-up), so those retain the old guard.
        """
        if self._qedit is not None and not text.lstrip().startswith("/"):
            if now and text.strip():  # Ctrl-X on an edited message: into the running reply now
                edit, self._qedit = self._qedit, None
                self.area.buffer.reset()
                self.turns.release()
                self.on_steer(text.strip())
                return
            self._queue_edit_finish(text)  # Enter: back in its place (empty: removed)
            return
        steer = now and self._revealing is not None and not text.lstrip().startswith("/")
        if steer and not text.strip():
            self.on_steer("")  # an empty box: the queued messages go in now
            return
        if not text.strip():
            return
        if self.busy and self._revealing is None and not text.lstrip().startswith("/"):
            self.busy_label = "still replying… (Esc cancels)"
            return
        self._set_text(text)
        # History navigation is for conversation messages.  Slash commands are UI
        # actions, and keeping them here makes Up/Down fall into the command menu
        # instead of continuing through the message history.
        if not text.lstrip().startswith("/"):
            self.area.buffer.append_to_history()
        self.area.buffer.reset()  # reload working lines so Up sees this turn, not a stale snapshot
        (self.on_steer if steer else self.on_submit)(text)

    def _picker_rows(self):
        p = self._picker
        return p["tabs"][p["tab"]][1] if p else []

    def _picker_text(self):
        p = self._picker
        if not p:
            return ""
        return FormattedText(_picker_lines(p["title"], p["tabs"], p["tab"], p["idx"], p["current"], framed=True))

    def _form_text(self):
        f = self._form
        return FormattedText(f.lines(framed=True, typed=self.area.text)) if f else ""

    def _finish_form(self, cancel: bool = False) -> None:
        f, self._form = self._form, None
        cb, self._form_cb = self._form_cb, lambda a: None
        self.scroll_top = None  # what the answer leads to (the plan being carried out) should be in view
        if f is not None:
            cb(None if cancel else f.answers())
        self.app.invalidate()

    def set_history(self, turns: list[str]) -> None:
        """Replace Up/Down history with this conversation's user messages (oldest first)."""
        hist = self.area.buffer.history
        if isinstance(hist, ConversationHistory):
            hist.replace(turns)
        self.area.buffer.reset()
        self.app.invalidate()

    def remember_user(self, text: str) -> None:
        """Record a user turn for Up/Down (spoken messages never go through the input box)."""
        text = (text or "").strip()
        if not text or text.lstrip().startswith("/"):
            return

        def apply() -> None:
            buf = self.area.buffer
            strings = buf.history.get_strings()
            if strings and strings[-1] == text:
                return
            buf.history.append_string(text)
            if not buf.text.strip():
                buf.reset()
            self.app.invalidate()

        loop = self.app.loop
        if loop is not None:
            loop.call_soon_threadsafe(apply)
        else:
            apply()

    # -- public API (thread-safe)
    def add(self, renderable, gap_before: bool = False) -> _Entry:
        e = _Entry(renderable, gap_before)
        with self._lock:
            self.entries.append(e)
        self.scroll_top = None
        self.app.invalidate()
        return e

    def update_entry(self, e: _Entry, renderable) -> None:
        """Redraw an entry with new content in place (the live transcript of what is being said)."""
        e.renderable = renderable
        e.invalidate()
        self.scroll_top = None
        self.app.invalidate()

    def remove_entry(self, e: _Entry) -> None:
        with self._lock:
            if e in self.entries:
                self.entries.remove(e)
        self.app.invalidate()

    def toggle_tools(self) -> None:
        """Ctrl-O: unfold (or fold again) the tool calls and subagents of the latest reply that had
        any, as in Claude Code. A reply still in progress shows them one per row anyway."""
        with self._lock:
            e = next((x for x in reversed(self.entries) if isinstance(x, _ReplyEntry) and x.marks), None)
        if e is not None:
            e.toggle_fold()

    def start_reply(self, markdown: bool = True) -> _ReplyEntry:
        e = _ReplyEntry(markdown)
        e.on_click = self.app.invalidate
        with self._lock:
            self.entries.append(e)
        self.busy = True
        self.busy_label = ""
        self._revealing = e
        if self.buddy:
            self.buddy.touch()
        self.app.invalidate()
        threading.Thread(target=self._pace, args=(e,), daemon=True).start()
        return e

    def split_reply(self, e: _ReplyEntry, *renderables) -> _ReplyEntry:
        """Close reply `e`, put `renderables` under it and open a fresh reply for what follows: a
        question's answer is a transcript block of its own (as in Claude Code), not a line inside
        the reply. A reply that had not said anything yet is dropped rather than left behind as a
        stray empty block."""
        with self._lock:
            dropped = not e.buf.strip() and not e.agents and not e.tools and e in self.entries
            if dropped:
                self.entries.remove(e)
        self.end_reply(e)
        for i, r in enumerate(renderables):
            self.add(r, gap_before=dropped and i == 0)  # a finished reply already ends in a blank line
        return self.start_reply(markdown=e.markdown)

    def update_reply(self, e: _ReplyEntry, delta: str = "", status: str | None = None, agent=None, tool=None, force: bool = False) -> None:
        if delta and not e.buf:
            delta = delta.lstrip()  # the "\n\n" the brain puts between text blocks means nothing at the top of a fresh reply
        if delta:
            e.buf += delta  # revealed by the pacer thread, not here
            e.status = ""
            if self.buddy:
                self.buddy.working()
        if agent is not None or tool is not None:
            e.place(agent if agent is not None else tool)
            e.invalidate()
            self.app.invalidate()
        if status is not None:
            e.status = status
            if self.buddy:
                if status.startswith("using "):
                    self.buddy.using(status[len("using "):].rstrip("…"))
                else:
                    self.buddy.working()
            e.invalidate()
            # deliberately no scroll reset: a reader who scrolled up mid-reply keeps their place
            self.app.invalidate()

    def _pace(self, e: _ReplyEntry) -> None:
        """Reveal loop. A typed reply (no gate) shows every delta as soon as it arrives. A spoken
        reply (gate set) is a typewriter: text shows at ~reveal_cps, never past what the voice has
        said, and a backlog drains within ~catch_up seconds so it never crawls. Runs at 60 fps
        with fractional progress so each glyph fades in between letters; the markdown render of
        the growing buffer happens here, off the UI thread."""
        tick, last = 1 / 60, time.monotonic()
        deadline = last + tick
        while True:
            time.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            dt, last = now - last, now
            deadline = max(deadline + tick, now + tick / 2)
            w = self._width()
            r = e.full(w, fresh=False)
            if r.stale(e.buf, e.marks) and (
                r.total - e.shown < self.reveal_cps * 0.5 or now - r.at > 0.5 or e.finished
            ):
                # Re-render only when the reveal is about to catch up with the last render (or
                # it is half a second old): the markdown pass is O(reply) and the lag hides it.
                r = e.full(w)
            total = r.total
            if e.gate is not None and e.buf and not e.finished:
                # Spoken replies: never show more than the voice has got to (see _Segments.reach).
                # Capped at the chars this render came from, not the live buffer: the render goes
                # stale while the voice holds the reveal back, and the buffer runs ahead of it.
                total = min(total, r.reach(min(e.gate(), r.src_len)))
            backlog = total - e.shown
            if backlog > 0:
                if e.gate is None:
                    # Typed replies are not held back: whatever the brain has sent is shown as it lands.
                    step = backlog
                else:
                    # Spoken replies keep pace with the voice: bursts catch up gently, capped so a
                    # reply that arrived whole still reads as live speech, not a paste.
                    step = min(self.max_reveal_cps * dt, max(self.reveal_cps * dt, backlog * dt / self.catch_up))
                e.shown = min(total, e.shown + step)
                e.invalidate()
                self.app.invalidate()
            elif e.finished:
                break
        self._settle(e)

    def _settle(self, e: _ReplyEntry) -> None:
        if not e.buf.strip() and not e.status:
            e.buf = "…"
        e.reveal_all()
        e.status = ""
        e.done = True
        if self._revealing is e:  # a later reply may already be streaming (the plan ends, the work starts)
            self.busy = False
            self.busy_label = ""
            self._revealing = None
            if self.buddy:
                self.buddy.rest()
        e.invalidate()
        self.app.invalidate()

    def end_reply(self, e: _ReplyEntry, stats: str = "", cancelled: bool = False) -> None:
        """The brain is done; the pacer finishes revealing and then clears busy. The timer stops here
        becomes the dim line under the reply (`stats`, if given, follows it; the chat screen sends none)."""
        e.cancelled = e.cancelled or cancelled
        if e.ended is None:
            e.ended = time.monotonic()
        e.footer = " · ".join(p for p in (f"{e.elapsed:.1f}s", "cancelled" if e.cancelled else "", stats) if p)
        e.finished = True

    def _cancel(self) -> None:
        """Esc / Ctrl-C while busy: show everything that has arrived at once, then cancel the brain."""
        e = self._revealing
        if e is not None:
            # Freeze the wall-clock figure now. The provider may take a moment to terminate and
            # report the partial usage, but the transcript should acknowledge Esc immediately.
            e.cancelled = True
            if e.ended is None:
                e.ended = time.monotonic()
            e.footer = f"{e.elapsed:.1f}s · cancelled"
            e.reveal_all()
            e.invalidate()
        self.on_cancel()

    def open_picker(self, title: str, options, current: str, cb: Callable[[str], None], prefer_tab: str = "") -> None:
        """options = [(value, label, desc)] or [(tab_name, rows)]; with tabs, ←/→ switches between them."""
        tabs = _as_tabs(options)
        tab, idx = _locate(tabs, current, prefer_tab)
        self._picker = {"title": title, "tabs": tabs, "tab": tab, "current": current, "idx": idx, "cb": cb}
        self.app.invalidate()

    def open_questions(self, questions: list[dict], cb: Callable[[dict[str, str] | None], None]) -> None:
        """Show an AskUserQuestion form; cb gets the answers, or None if the user dismissed it."""
        self._form = QuestionForm(questions)
        self._form_cb = cb
        self.area.text = ""
        self.scroll_top = None
        if self.buddy:
            self.buddy.touch()
        self.app.invalidate()

    @property
    def form_open(self) -> bool:
        return self._form is not None

    @property
    def pending_questions(self) -> list[dict]:
        return self._form.qs if self._form is not None else []

    def answer_questions(self, answers: dict[str, str] | None) -> bool:
        """Close the open form with answers that arrived elsewhere (the phone). False if none is open."""
        f, self._form = self._form, None
        if f is None:
            return False
        cb, self._form_cb = self._form_cb, lambda a: None
        self.scroll_top = None
        cb(answers if isinstance(answers, dict) and answers else None)
        self.app.invalidate()
        return True

    def ask_questions(self, questions: list[dict], on_open: Callable[[], None] | None = None) -> dict[str, str] | None:
        """Blocking variant for worker threads: open the form and wait for the user. `on_open` runs once
        the form is showing (so `form_open` is already true for anything it reports)."""
        done = threading.Event()
        box: dict = {}

        def cb(answers):
            box["a"] = answers
            done.set()

        self.open_questions(questions, cb)
        if on_open:
            on_open()
        done.wait()
        return box.get("a")

    def exit(self, result) -> None:
        self.app.exit(result=result)

    def run(self, pre_run=None):
        self._sync_title()
        return self.app.run(pre_run=pre_run)

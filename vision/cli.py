"""Vision command-line interface."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from vision import __version__
from vision.ui import MODEL_CHOICES, USER_BG, ChatScreen, ReplyView, header_renderable, pick, short_path, show_header, show_user
from vision.config import (
    CONFIG_PATH,
    KOKORO_MODEL_URL,
    KOKORO_VOICES_URL,
    MODELS_DIR,
    VOICE_PRESETS,
    WORKSPACE_DIR,
    Config,
    load_config,
)

app = typer.Typer(
    help="Vision: your local voice + text AI assistant, driven by your Claude Code subscription.",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
)
console = Console()
ACCENT = "bright_cyan"
NAME = f"[bold {ACCENT}]Vision[/bold {ACCENT}]"

_EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "goodbye vision", "bye vision", "that's all", "that is all", "stop listening"}


# ---------------------------------------------------------------- helpers
def _cfg(model: Optional[str], voice: Optional[str], effort: Optional[str] = None) -> Config:
    cfg = load_config()
    if model:
        cfg.brain.model = model
    if effort:
        cfg.brain.effort = effort
    if voice:
        cfg.voice.voice = voice
    return cfg


def _brain(cfg: Config, voice_mode: bool, cont: bool, new: bool):
    from vision.brain import Brain

    sid = None
    if cont and not new:
        sid = Brain.last_session_id()
    return Brain(cfg.brain, voice_mode=voice_mode, session_id=sid)


def _speaker(cfg: Config):
    from vision.tts import Speaker

    return Speaker(cfg.voice)


def _status(msg: str):
    console.print(f"  [dim]{msg}[/dim]")


def _run_turn(brain, text: str, speaker=None, markdown: bool = True):
    """One brain turn rendered beside 'Vision ›', optionally spoken. Returns Turn or None if cancelled."""
    ss = None
    if speaker is not None:
        from vision.tts import StreamingSpeaker

        ss = StreamingSpeaker(speaker)
    with ReplyView(console, markdown=markdown) as view:

        def on_text(delta: str):
            if view.status:
                view.status = ""
            view.append(delta)
            if ss:
                ss.feed(delta)

        def on_status(tool: str):
            view.set_status(f"using {tool}…")

        try:
            turn = brain.ask(text, on_text=on_text, on_status=on_status)
        except KeyboardInterrupt:
            if ss:
                ss.stop()
            brain.cancel()
            view.append("\n\n*(cancelled)*" if markdown else "\n(cancelled)")
            return None
        if turn.is_error and not turn.text:
            view.append(f"**Vision could not answer:** {turn.error}" if markdown else f"Vision could not answer: {turn.error}")
            return turn
    if ss:
        try:
            ss.finish()
        except KeyboardInterrupt:
            ss.stop()
    return turn


# ---------------------------------------------------------------- usage
_USAGE_LINE = re.compile(r"^(?P<label>[^:]+):\s+(?P<pct>\d+)% used(?:\s+·\s+resets\s+(?P<reset>.+))?$")


def _bar(pct: float) -> str:
    n = int(round(pct / 5))
    colour = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
    return f"[{colour}]{'█' * n}[/{colour}][dim]{'░' * (20 - n)}[/dim]"


def _usage_renderable(text: str, full: bool = False):
    """Claude Code's /usage report (session, week, per-model such as Fable) as bars."""
    t = Table(title="Claude subscription usage", header_style=ACCENT, show_edge=False)
    t.add_column("window"), t.add_column("used", justify="right"), t.add_column("resets")
    rest = []
    for line in text.splitlines():
        m = _USAGE_LINE.match(line.strip())
        if m:
            pct = float(m.group("pct"))
            t.add_row(m.group("label"), f"{_bar(pct)} {pct:4.0f}%", (m.group("reset") or "").replace(" (America/New_York)", ""))
        else:
            rest.append(line)
    if full:
        tail = Text("\n".join(l for l in rest if l.strip() and not l.startswith("You are currently")), style="dim")
    else:
        tail = Text("`vision usage --full` (or /usage full) shows what has been contributing to these numbers.", style="dim")
    return Group(t, tail)


def _render_usage_text(text: str, full: bool = False) -> None:
    console.print(_usage_renderable(text, full))


def _usage_fallback_renderable(usage: dict | None):
    """Fallback using the rate-limit event attached to the last reply."""
    from datetime import datetime

    if not usage:
        return Text("No usage data yet. Ask Vision something first, or run `vision usage`.", style="yellow")
    info, at = usage.get("info", {}), usage.get("at", 0)
    wins = info.get("unifiedWindows") or {}
    t = Table(title="Claude subscription usage", header_style=ACCENT, show_edge=False)
    t.add_column("window"), t.add_column("used", justify="right"), t.add_column("resets")
    labels = {"five_hour": "Current session", "seven_day": "Current week (all models)", "seven_day_overage_included": "Current week (incl. extra usage)"}
    for key, w in wins.items():
        pct = float(w.get("utilization") or 0) * 100
        reset = w.get("resetsAt")
        when = datetime.fromtimestamp(reset).strftime("%b %d, %H:%M") if reset else "?"
        t.add_row(labels.get(key, key), f"{_bar(pct)} {pct:4.0f}%", when)
    stamp = datetime.fromtimestamp(at).strftime("%H:%M:%S") if at else "?"
    return Group(t, Text(f"from the last reply's rate-limit event at {stamp}", style="dim"))


def _render_usage(usage: dict | None, refreshed: bool):
    console.print(_usage_fallback_renderable(usage))


def _usage_for(brain, full: bool = False):
    text = brain.usage_report()
    return _usage_renderable(text, full) if text else _usage_fallback_renderable(brain.last_usage or brain.cached_usage())


def _show_usage(brain, full: bool = False) -> None:
    with console.status("[dim]asking Claude Code for usage…[/dim]"):
        r = _usage_for(brain, full)
    console.print(r)


@app.command()
def usage(full: bool = typer.Option(False, "--full", help="Also show what has been contributing to usage.")):
    """Show your Claude subscription usage: session, week, and per-model windows (e.g. Fable)."""
    from vision.brain import Brain

    _show_usage(Brain(load_config().brain), full)


# ---------------------------------------------------------------- default: chat
@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    speak: bool = typer.Option(False, "--speak", "-s", help="Read replies aloud."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Claude model (alias or full name)."),
    effort: Optional[str] = typer.Option(None, "--effort", help="Effort level: low, medium, high."),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice preset, name, or blend."),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the last Vision conversation."),
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
):
    """Start an interactive text chat with Vision (default when no command is given)."""
    if version:
        console.print(f"Vision {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    chat(speak=speak, model=model, effort=effort, voice=voice, cont=cont)


def _model_label(alias: str) -> str:
    return next((label for v, label, _ in MODEL_CHOICES if v == alias), alias or "default")


HELP_TEXT = (
    "[bold]/model[/bold] pick a model (or /model sonnet) · [bold]/speak[/bold] toggle spoken replies · "
    "[bold]/voice <name>[/bold] change voice\n"
    "[bold]/listen[/bold] say one turn with the mic · [bold]/talk[/bold] switch to voice conversation · "
    "[bold]/say <text>[/bold] speak text\n"
    "[bold]/new[/bold] fresh conversation · [bold]/usage[/bold] subscription usage (add 'full' for detail) · "
    "[bold]/clear[/bold] clear the screen · [bold]/quit[/bold]\n"
    "[dim]Enter sends · Ctrl-J newline · PgUp/PgDn scroll · Esc cancels a reply · Ctrl-D quits[/dim]"
)


def _user_band(text: str):
    return Padding(Text.assemble(("you › ", "bold yellow"), (text, "bold")), (0, 1), style=USER_BG, expand=True)


def _banner(cfg: Config, brain, speak: bool):
    return header_renderable(
        f"model {_model_label(cfg.brain.model)} · speech {'on' if speak else 'off'} · {short_path(brain.workdir)}"
        + (" · resumed" if brain.session_id else ""),
        f"tools: {', '.join(cfg.brain.allowed_tools) or 'none'}",
        "/help for commands · Esc cancels a reply · Ctrl-D or /quit exits",
    )


def chat(speak: bool, model: Optional[str], effort: Optional[str], voice: Optional[str], cont: bool):
    from vision.config import STATE_DIR

    cfg = _cfg(model, voice, effort)
    brain = _brain(cfg, voice_mode=False, cont=cont, new=False)
    state = {"speak": speak, "speaker": None, "mic": None, "stt": None, "ss": None}
    if speak:
        state["speaker"] = _speaker(cfg)
        threading.Thread(target=state["speaker"]._load, daemon=True).start()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    def speaker():
        if state["speaker"] is None:
            state["speaker"] = _speaker(cfg)
        return state["speaker"]

    def status_line() -> str:
        return (
            f"model {_model_label(cfg.brain.model)}  ·  speech {'on' if state['speak'] else 'off'}  ·  "
            f"{short_path(brain.workdir)}  ·  /help"
        )

    screen = ChatScreen(str(STATE_DIR / "history"), status_line)
    screen.add(_banner(cfg, brain, speak))

    def note(msg: str, style: str = "dim"):
        screen.add(Text.from_markup(msg) if "[" in msg else Text(msg, style=style))

    # ---- one reply, run on a worker thread so the UI keeps drawing
    def run_turn(text: str):
        ss = None
        if state["speak"]:
            from vision.tts import StreamingSpeaker

            ss = StreamingSpeaker(speaker())
        state["ss"] = ss
        entry = screen.start_reply(markdown=True)

        def on_text(d):
            screen.update_reply(entry, delta=d)
            if ss:
                ss.feed(d)

        def on_status(tool):
            screen.update_reply(entry, status=f"using {tool}…", force=True)

        try:
            turn = brain.ask(text, on_text=on_text, on_status=on_status)
            if turn.is_error and not turn.text:
                msg = "*(cancelled)*" if turn.error == "cancelled" else f"**Vision could not answer:** {turn.error}"
                screen.update_reply(entry, delta=msg, force=True)
        except Exception as e:  # noqa: BLE001
            screen.update_reply(entry, delta=f"**Error:** {e}", force=True)
        finally:
            screen.end_reply(entry)
            if ss:
                try:
                    ss.finish()
                except Exception:
                    pass
            state["ss"] = None

    def cancel():
        brain.cancel()
        if state["ss"]:
            state["ss"].stop()
        elif state["speaker"]:
            state["speaker"].stop()

    def background(fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    # ---- slash commands
    def command(cmd: str, arg: str):
        if cmd in ("quit", "exit", "q"):
            screen.exit(("quit",))
        elif cmd == "help":
            note(HELP_TEXT)
        elif cmd == "clear":
            with screen._lock:
                screen.entries.clear()
            screen.add(_banner(cfg, brain, state["speak"]))
        elif cmd == "model":
            def set_model(choice):
                cfg.brain.model = "" if choice in ("default", "reset") else choice
                note(f"model → {_model_label(cfg.brain.model)}")
            if arg:
                set_model(arg)
            else:
                screen.open_picker("Choose a model", MODEL_CHOICES, cfg.brain.model, set_model)
        elif cmd == "usage":
            screen.busy, screen.busy_label = True, "asking Claude Code for usage…"

            def go():
                try:
                    screen.add(_usage_for(brain, full=arg == "full"))
                finally:
                    screen.busy = False
            background(go)
        elif cmd == "new":
            brain.new_session()
            note("new conversation")
        elif cmd == "speak":
            if state["speaker"] is None:
                speaker()
                state["speak"] = True
            else:
                state["speak"] = not state["speak"]
            note(f"speech {'on' if state['speak'] else 'off'}")
        elif cmd == "voice":
            def go():
                try:
                    speaker().set_voice(arg or cfg.voice.voice)
                    note(f"voice → {speaker().voice}")
                except Exception as e:
                    note(f"[red]{e}[/red]")
            background(go)
        elif cmd == "say":
            background(lambda: speaker().say(arg))
        elif cmd == "talk":
            screen.exit(("talk",))
        elif cmd == "listen":
            screen.exit(("listen",))
        else:
            note(f"[red]unknown command /{cmd}[/red]")

    def submit(text: str):
        text = text.strip()
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            command(cmd.lower(), arg.strip())
            return
        screen.add(_user_band(text), gap_before=True)
        background(run_turn, text)

    screen.on_submit = submit
    screen.on_cancel = cancel

    # ---- outer loop: the screen exits only for voice modes or quitting
    while True:
        try:
            result = screen.run()
        except (EOFError, KeyboardInterrupt):
            result = ("quit",)
        kind = result[0] if result else "quit"
        if kind == "quit":
            cancel()
            console.print(f"{NAME} [dim]signing off.[/dim]")
            break
        if kind == "talk":
            _talk_loop(cfg, brain, speaker(), ptt=False, echo=True)
            brain.voice_mode = False
            screen.add(Text("back from voice conversation", style="dim"))
        elif kind == "listen":
            if state["mic"] is None:
                from vision.audio import Microphone
                from vision.stt import Transcriber

                state["mic"], state["stt"] = Microphone(cfg.listen), Transcriber(cfg.listen)
                with console.status("[dim]loading speech recognition…[/dim]"):
                    state["stt"].warm_up()
            heard = _listen_once(cfg, state["mic"], state["stt"])
            if heard:
                screen.add(_user_band(heard), gap_before=True)
                background(run_turn, heard)
            else:
                screen.add(Text("didn't catch that", style="dim"))


# ---------------------------------------------------------------- talk (speech to speech)
def _listen_once(cfg: Config, mic, stt, ptt: bool = False, cancel: threading.Event | None = None) -> str:
    """Single spoken turn with a spinner (used by chat's /listen and `vision listen`)."""
    from vision.config import resolve_device
    from vision.tts import chime

    out_dev = resolve_device(cfg.voice.output_device, "output")
    if ptt:
        console.print("[dim]recording… press Enter to stop[/dim]")
        audio = mic.record_until_enter()
    else:
        if cfg.listen.chime:
            chime("listen", out_dev)
        with console.status(f"[{ACCENT}]listening…[/{ACCENT}]", spinner="dots") as st:
            audio = mic.record_utterance(on_speech_start=lambda: st.update(f"[{ACCENT}]hearing you…[/{ACCENT}]"), cancel=cancel)
    if audio is None or audio.size == 0:
        return ""
    with console.status("[dim]transcribing…[/dim]", spinner="dots"):
        return stt.transcribe(audio)


class _Keyboard:
    """Background stdin reader so you can type to Vision while it is listening."""

    def __init__(self):
        import queue

        self.q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in sys.stdin:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        self.q.put(None)  # EOF

    def poll(self, timeout: float = 0.1):
        """Returns a typed line, None on EOF, or raises queue.Empty if nothing yet."""
        return self.q.get(timeout=timeout)


def _next_input(cfg: Config, mic, stt, kb: _Keyboard, ptt: bool) -> str | None:
    """Wait for either a spoken utterance or a typed line. Returns text, or None on EOF."""
    import queue

    from vision.config import resolve_device
    from vision.tts import chime

    out_dev = resolve_device(cfg.voice.output_device, "output")

    if ptt:
        console.print("[dim]press Enter to talk, or type a message ›[/dim] ", end="")
        line = kb.q.get()
        if line is None:
            return None
        if line.strip():
            return line.strip()
        stop = threading.Event()
        console.print(f"[{ACCENT}]recording… press Enter to stop[/{ACCENT}]")
        threading.Thread(target=lambda: (kb.q.get(), stop.set()), daemon=True).start()
        audio = mic.record_until_enter(stop)
    else:
        if cfg.listen.chime:
            chime("listen", out_dev)
        console.print(f"[{ACCENT}]listening…[/{ACCENT}] [dim](or type a message and press Enter)[/dim]")
        cancel = threading.Event()
        result: dict = {}

        def capture():
            result["audio"] = mic.record_utterance(
                on_speech_start=lambda: console.print(f"[{ACCENT}]hearing you…[/{ACCENT}]"), cancel=cancel
            )

        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        typed = None
        while worker.is_alive():
            try:
                typed = kb.poll(0.1)
            except queue.Empty:
                continue
            cancel.set()
            worker.join()
            break
        if typed is not None or (not worker.is_alive() and cancel.is_set()):
            if typed is None:
                return None  # EOF (Ctrl-D)
            if typed.strip():
                return typed.strip()
            return ""  # empty Enter: just restart listening
        audio = result.get("audio")

    if audio is None or audio.size == 0:
        return ""
    with console.status("[dim]transcribing…[/dim]", spinner="dots"):
        return stt.transcribe(audio)


def _talk_loop(cfg: Config, brain, speaker, ptt: bool, echo: bool):
    from vision.audio import Microphone
    from vision.stt import Transcriber

    brain.voice_mode = True
    mic = Microphone(cfg.listen)
    stt = Transcriber(cfg.listen)
    with console.status("[dim]warming up ears and voice…[/dim]"):
        t_load = threading.Thread(target=speaker._load, daemon=True)
        t_load.start()
        stt.warm_up()
        t_load.join()
    show_header(
        console,
        f"listening · ears {stt.device} · voice {speaker.voice} on {speaker.device} · model {_model_label(cfg.brain.model)} · {short_path(brain.workdir)}",
        "Push-to-talk: press Enter to start and stop recording." if ptt else "Hands-free: just speak; pause to send.",
        "Type a message and press Enter at any time · Ctrl-C interrupts a reply · say or type “goodbye” to exit",
    )
    kb = _Keyboard()
    while True:
        try:
            heard = _next_input(cfg, mic, stt, kb, ptt)
        except KeyboardInterrupt:
            console.print(f"\n{NAME} [dim]signing off.[/dim]")
            break
        if heard is None:
            console.print(f"{NAME} [dim]signing off.[/dim]")
            break
        if not heard:
            continue
        if heard.startswith("/"):
            cmd, _, arg = heard[1:].partition(" ")
            if cmd == "usage":
                _show_usage(brain, full=arg.strip() == "full")
            elif cmd == "model":
                choice = arg.strip() or pick("Choose a model", MODEL_CHOICES, current=cfg.brain.model)
                if choice is not None:
                    cfg.brain.model = "" if choice in ("default", "reset") else choice
                    console.print(f"[dim]model → {_model_label(cfg.brain.model)}[/dim]")
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                console.print("[dim]in talk mode: /model, /usage, /quit[/dim]")
            continue
        show_user(console, heard)
        if heard.lower().strip(" .!?,") in _EXIT_PHRASES:
            try:
                speaker.say("Goodbye.")
            except KeyboardInterrupt:
                speaker.stop()
            break
        _run_turn(brain, heard, speaker, markdown=False)


@app.command()
def talk(
    ptt: bool = typer.Option(False, "--ptt", help="Push-to-talk instead of hands-free listening."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    effort: Optional[str] = typer.Option(None, "--effort"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v"),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the last Vision conversation."),
):
    """Speech-to-speech conversation: you talk, Vision talks back."""
    cfg = _cfg(model, voice, effort)
    brain = _brain(cfg, voice_mode=True, cont=cont, new=False)
    speaker = _speaker(cfg)
    _talk_loop(cfg, brain, speaker, ptt=ptt, echo=True)


# ---------------------------------------------------------------- one-shots
@app.command()
def ask(
    question: list[str] = typer.Argument(None, help="Question text. Omit to read from stdin."),
    speak: bool = typer.Option(False, "--speak", "-s", help="Also read the answer aloud."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    effort: Optional[str] = typer.Option(None, "--effort"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v"),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the last Vision conversation."),
    plain: bool = typer.Option(False, "--plain", help="Plain text output (for piping)."),
):
    """Ask Vision one question and exit."""
    text = " ".join(question) if question else sys.stdin.read()
    text = text.strip()
    if not text:
        raise typer.BadParameter("empty question")
    cfg = _cfg(model, voice, effort)
    brain = _brain(cfg, voice_mode=speak and not plain, cont=cont, new=False)
    if plain:
        from vision.brain import Brain  # noqa: F401

        turn = brain.ask(text)
        if turn.is_error and not turn.text:
            console.print(f"[red]{turn.error}[/red]", file=sys.stderr)
            raise typer.Exit(1)
        print(turn.text)
        if speak:
            _speaker(cfg).say(turn.text)
        return
    _run_turn(brain, text, _speaker(cfg) if speak else None)


@app.command()
def say(
    text: list[str] = typer.Argument(None, help="Text to speak. Omit to read from stdin."),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice preset, name, or blend."),
    speed: Optional[float] = typer.Option(None, "--speed", help="Speech rate multiplier (1.0 = normal)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write a WAV file instead of playing."),
):
    """Text-to-speech: speak text in Vision's voice (or save it to a WAV)."""
    msg = " ".join(text) if text else sys.stdin.read()
    msg = msg.strip()
    if not msg:
        raise typer.BadParameter("nothing to say")
    cfg = _cfg(None, voice)
    sp = _speaker(cfg)
    if out:
        p = sp.save(msg, out, speed)
        console.print(f"[dim]wrote[/dim] {p}")
        return
    try:
        sp.say(msg, speed)
    except KeyboardInterrupt:
        sp.stop()


@app.command()
def listen(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Transcribe an audio file instead of the mic."),
    ptt: bool = typer.Option(False, "--ptt", help="Record until Enter instead of auto end-pointing."),
    forever: bool = typer.Option(False, "--forever", help="Keep transcribing utterances until Ctrl-C."),
):
    """Speech-to-text: transcribe your microphone (or a file) and print the text."""
    from vision.stt import Transcriber

    cfg = load_config()
    stt = Transcriber(cfg.listen)
    if file:
        with console.status("[dim]transcribing…[/dim]"):
            print(stt.transcribe_file(str(file)))
        return
    from vision.audio import Microphone

    mic = Microphone(cfg.listen)
    with console.status("[dim]loading speech recognition…[/dim]"):
        stt.warm_up()
    try:
        while True:
            heard = _listen_once(cfg, mic, stt, ptt=ptt)
            if heard:
                print(heard, flush=True)
            if not forever:
                break
    except KeyboardInterrupt:
        pass


@app.command()
def voices(
    preview: bool = typer.Option(False, "--preview", "-p", help="Speak a sample line in each English voice."),
    text: str = typer.Option("Good afternoon. Vision online, all systems nominal.", "--text", help="Sample line for --preview."),
):
    """List available voices and presets."""
    sp = _speaker(load_config())
    names = sp.available_voices()
    t = Table(title="Presets", show_header=True, header_style=ACCENT)
    t.add_column("preset"), t.add_column("blend")
    for k, v in VOICE_PRESETS.items():
        t.add_row(k, v)
    console.print(t)
    english = [n for n in names if n[0] in "ab"]
    console.print("[bold]English voices[/bold] (a = American, b = British; f/m = female/male):")
    console.print("  " + ", ".join(english))
    console.print("[bold]Other languages:[/bold] " + ", ".join(n for n in names if n[0] not in "ab"))
    console.print("[dim]Blend syntax: --voice \"bf_emma:0.6,bf_isabella:0.4\"[/dim]")
    if preview:
        for n in ["friday"] + [v for v in english if v.startswith("bf_")] + ["af_heart", "af_bella"]:
            console.print(f"  ▶ {n}")
            sp.set_voice(n)
            try:
                sp.say(text)
            except KeyboardInterrupt:
                sp.stop()
                break


# ---------------------------------------------------------------- setup / doctor / config
def _download(url: str, dest: Path) -> None:
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    with console.status(f"[dim]downloading {dest.name}…[/dim]"):
        urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


@app.command()
def setup(whisper: bool = typer.Option(True, help="Also pre-download the Whisper model.")):
    """Download voice and speech models (idempotent)."""
    from vision.tts import model_files

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    m, v = model_files()
    if not m.exists():
        _download(KOKORO_MODEL_URL, m)
    if not v.exists():
        _download(KOKORO_VOICES_URL, v)
    console.print(f"[green]✓[/green] Kokoro voice model in {MODELS_DIR}")
    if whisper:
        from vision.stt import Transcriber

        cfg = load_config()
        stt = Transcriber(cfg.listen)
        with console.status("[dim]loading Whisper (downloads on first run)…[/dim]"):
            stt.load()
        console.print(f"[green]✓[/green] Whisper ready on {stt.device}")
    console.print(f"[green]✓[/green] config: {CONFIG_PATH}")


@app.command()
def doctor():
    """Check Claude Code login, models, GPU, and audio devices."""
    ok = "[green]✓[/green]"
    bad = "[red]✗[/red]"
    warn = "[yellow]![/yellow]"
    cfg = load_config()

    # Claude Code
    exe = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if os.path.exists(exe):
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
        console.print(f"{ok} Claude Code: {ver} ({exe})")
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        r = subprocess.run(
            [exe, "-p", "Reply with the single word OK", "--output-format", "json", "--no-session-persistence", "--model", "haiku"],
            capture_output=True, text=True, env=env, cwd=str(WORKSPACE_DIR), timeout=120,
        )
        if r.returncode == 0 and '"is_error":false' in r.stdout:
            console.print(f"{ok} Claude Code login works (headless round-trip succeeded)")
        else:
            console.print(f"{bad} Claude Code headless call failed: {(r.stderr or r.stdout).strip()[:300]}\n   Run `claude` once and log in.")
    else:
        console.print(f"{bad} Claude Code CLI not found. Install: npm i -g @anthropic-ai/claude-code, then run `claude` to log in.")

    # models
    from vision.tts import models_present

    console.print(f"{ok if models_present() else bad} Kokoro voice model in {MODELS_DIR}" + ("" if models_present() else "  → run `vision setup`"))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        console.print(f"{ok} GPU: {r.stdout.strip()}" if r.returncode == 0 else f"{warn} no NVIDIA GPU; Whisper will use CPU")
    except FileNotFoundError:
        console.print(f"{warn} nvidia-smi not found; Whisper will use CPU")
    try:
        from vision.stt import Transcriber

        stt = Transcriber(cfg.listen)
        with console.status("[dim]loading Whisper…[/dim]"):
            stt.load(quiet=True)
        console.print(f"{ok} Whisper ready: {stt.device}")
    except Exception as e:
        console.print(f"{bad} Whisper failed: {e}")
    console.print(f"{ok if shutil.which('espeak-ng') else warn} espeak-ng {'found' if shutil.which('espeak-ng') else 'missing (sudo dnf install espeak-ng) — needed for some words'}")

    # audio
    import sounddevice as sd

    devs = sd.query_devices()
    try:
        din, dout = sd.default.device
        console.print(f"{ok} default input:  {devs[din]['name']}")
        console.print(f"{ok} default output: {devs[dout]['name']}")
    except Exception as e:
        console.print(f"{bad} audio device query failed: {e}")
    t = Table(title="Audio devices", header_style=ACCENT)
    t.add_column("#"), t.add_column("name"), t.add_column("in"), t.add_column("out")
    for i, d in enumerate(devs):
        if d["max_input_channels"] or d["max_output_channels"]:
            t.add_row(str(i), d["name"], str(d["max_input_channels"]), str(d["max_output_channels"]))
    console.print(t)
    try:
        from vision.tts import Speaker

        sp = Speaker(cfg.voice)
        sp._load()
        console.print(f"{ok} voice '{cfg.voice.voice}' loads on {sp.device} ({sp.voice})")
    except Exception as e:
        console.print(f"{bad} voice failed: {e}")
    console.print(f"[dim]config: {CONFIG_PATH}[/dim]")


@app.command()
def config(edit: bool = typer.Option(False, "--edit", "-e", help="Open the config in $EDITOR.")):
    """Show (or edit) Vision's configuration."""
    load_config()
    if edit:
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        subprocess.call([editor, str(CONFIG_PATH)])
        return
    console.print(f"[dim]{CONFIG_PATH}[/dim]")
    console.print(Text(CONFIG_PATH.read_text()))


def main():
    try:
        app()
    except KeyboardInterrupt:
        console.print()


if __name__ == "__main__":
    main()

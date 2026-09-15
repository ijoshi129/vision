"""Vision command-line interface."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vision import __version__
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


class _Streamer:
    """Renders streamed markdown live, and optionally feeds a StreamingSpeaker."""

    def __init__(self, speaker=None, markdown: bool = True, prefix: bool = True):
        self.buf = ""
        self.markdown = markdown
        self.tools: list[str] = []
        self._live: Live | None = None
        self._ss = None
        if speaker is not None:
            from vision.tts import StreamingSpeaker

            self._ss = StreamingSpeaker(speaker)
        if prefix:
            console.print(f"{NAME}")

    def __enter__(self):
        if self.markdown:
            self._live = Live(Markdown(""), console=console, refresh_per_second=12, vertical_overflow="visible")
            self._live.__enter__()
        return self

    def on_text(self, delta: str):
        self.buf += delta
        if self._live:
            self._live.update(Markdown(self.buf))
        else:
            console.print(delta, end="", highlight=False, markup=False, soft_wrap=True)
        if self._ss:
            self._ss.feed(delta)

    def on_status(self, tool: str):
        self.tools.append(tool)
        if self._live:
            self._live.update(Markdown(self.buf + f"\n\n*using {tool}…*"))

    def __exit__(self, *exc):
        if self._live:
            self._live.update(Markdown(self.buf))
            self._live.__exit__(*exc)
        elif self.buf and not self.buf.endswith("\n"):
            console.print()
        return False

    def finish_speech(self):
        if self._ss:
            self._ss.finish()

    def stop_speech(self):
        if self._ss:
            self._ss.stop()


def _run_turn(brain, text: str, speaker=None, markdown: bool = True):
    """One brain turn with live rendering and optional speech. Returns Turn."""
    with _Streamer(speaker=speaker, markdown=markdown) as s:
        try:
            turn = brain.ask(text, on_text=s.on_text, on_status=s.on_status)
        except KeyboardInterrupt:
            s.stop_speech()
            brain.cancel()
            console.print("[dim](cancelled)[/dim]")
            return None
    if turn.is_error and not turn.text:
        console.print(f"[red]Vision could not answer:[/red] {turn.error}")
        return turn
    try:
        s.finish_speech()
    except KeyboardInterrupt:
        s.stop_speech()
    return turn


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


def chat(speak: bool, model: Optional[str], effort: Optional[str], voice: Optional[str], cont: bool):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    from vision.config import STATE_DIR

    cfg = _cfg(model, voice, effort)
    brain = _brain(cfg, voice_mode=False, cont=cont, new=False)
    speaker = None
    if speak:
        speaker = _speaker(cfg)
        threading.Thread(target=speaker._load, daemon=True).start()
    mic = None
    stt = None

    console.print(
        Panel.fit(
            f"{NAME} online.  [dim]model:[/dim] {cfg.brain.model or 'default'}   [dim]speech:[/dim] {'on' if speak else 'off'}"
            + ("   [dim]resumed[/dim]" if brain.session_id else "")
            + "\n[dim]/help for commands · Ctrl-C cancels a reply · Ctrl-D or /quit exits[/dim]",
            border_style=ACCENT,
        )
    )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(STATE_DIR / "history")))
    style = Style.from_dict({"prompt": "bold ansiyellow"})

    while True:
        try:
            text = session.prompt([("class:prompt", "you › ")], style=style).strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"{NAME} [dim]signing off.[/dim]")
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                console.print(
                    "[bold]/speak[/bold] toggle spoken replies · [bold]/voice <name>[/bold] change voice · "
                    "[bold]/listen[/bold] say one turn with the mic · [bold]/talk[/bold] switch to voice conversation\n"
                    "[bold]/new[/bold] fresh conversation · [bold]/model <name>[/bold] switch model · "
                    "[bold]/say <text>[/bold] speak text · [bold]/quit[/bold]"
                )
            elif cmd == "new":
                brain.new_session()
                console.print("[dim]new conversation[/dim]")
            elif cmd == "model":
                cfg.brain.model = arg.strip()
                console.print(f"[dim]model → {cfg.brain.model or 'default'}[/dim]")
            elif cmd == "speak":
                if speaker is None:
                    speaker = _speaker(cfg)
                    speak = True
                else:
                    speak = not speak
                console.print(f"[dim]speech {'on' if speak else 'off'}[/dim]")
            elif cmd == "voice":
                if speaker is None:
                    speaker = _speaker(cfg)
                try:
                    speaker.set_voice(arg.strip() or cfg.voice.voice)
                    console.print(f"[dim]voice → {speaker.voice}[/dim]")
                except Exception as e:
                    console.print(f"[red]{e}[/red]")
            elif cmd == "say":
                if speaker is None:
                    speaker = _speaker(cfg)
                try:
                    speaker.say(arg)
                except KeyboardInterrupt:
                    speaker.stop()
            elif cmd == "talk":
                _talk_loop(cfg, brain, speaker or _speaker(cfg), ptt=False, echo=True)
                brain.voice_mode = False
            elif cmd == "listen":
                if mic is None:
                    from vision.audio import Microphone
                    from vision.stt import Transcriber

                    mic, stt = Microphone(cfg.listen), Transcriber(cfg.listen)
                    with console.status("[dim]loading speech recognition…[/dim]"):
                        stt.warm_up()
                heard = _listen_once(cfg, mic, stt)
                if heard:
                    console.print(f"[bold yellow]you ›[/bold yellow] {heard}")
                    _run_turn(brain, heard, speaker if speak else None)
            else:
                console.print(f"[red]unknown command /{cmd}[/red]")
            continue
        _run_turn(brain, text, speaker if speak else None)


# ---------------------------------------------------------------- talk (speech to speech)
def _listen_once(cfg: Config, mic, stt, ptt: bool = False, cancel: threading.Event | None = None) -> str:
    from vision.tts import chime

    from vision.config import resolve_device

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
    console.print(
        Panel.fit(
            f"{NAME} is listening.  [dim]ears:[/dim] {stt.device}   [dim]voice:[/dim] {speaker.voice}   "
            f"[dim]model:[/dim] {cfg.brain.model or 'default'}\n"
            + ("[dim]Push-to-talk: press Enter to start and stop recording.[/dim]\n" if ptt else "[dim]Hands-free: just speak; pause to send.[/dim]\n")
            + "[dim]Ctrl-C while Vision is talking interrupts it · Ctrl-C while listening exits · say “goodbye” to exit[/dim]",
            border_style=ACCENT,
        )
    )
    while True:
        try:
            if ptt:
                try:
                    input("press Enter to talk › ")
                except EOFError:
                    break
            heard = _listen_once(cfg, mic, stt, ptt=ptt)
        except KeyboardInterrupt:
            console.print(f"\n{NAME} [dim]signing off.[/dim]")
            break
        if not heard:
            continue
        console.print(f"[bold yellow]you ›[/bold yellow] {heard}")
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
        console.print(f"{ok} voice '{cfg.voice.voice}' loads ({sp.voice})")
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

"""Vision command-line interface."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from vision import __version__
from vision.brain import BrainError, agent_frame, cache_figure, context_figure, tool_frame
from vision.buddy import Buddy
from vision.warmup import WarmupProgress, warm_voice
from vision import usage as usage_ui
from vision import cachettl, clis
from vision.config import (
    CONFIG_PATH,
    LLAMA_DIR,
    LLAMA_URLS,
    MODELS_DIR,
    MODES,
    ORPHEUS_MODEL_URL,
    QWEN_TTS_BASE,
    QWEN_TTS_DESIGN,
    SNAC_MODEL_URL,
    VOICE_DESIGNS,
    VOICES_DIR,
    WORKSPACE_DIR,
    BrainConfig,
    Config,
    load_config,
    save_brain_defaults,
    save_voice_default,
    save_input_device,
    save_wake_enabled,
    saved_voices,
    voice_choices,
    voice_dir,
)
from vision.models import CONVERSATION_TABS, MODEL_TABS, THINKING_OFF, coerce_effort, effort_choices, model_label, provider_default, provider_for, provider_label, replace_retired_models, supports_effort
from vision.reply import status_label
from rich.markup import escape
from vision.ui import PLACEHOLDER, ChatScreen, ReplyView, SlashCommand, answered_grid, header_renderable, notice_grid, pick, short_path, show_header, show_user, user_grid, reply_grid, hearing_grid

app = typer.Typer(
    help="Vision: your local voice + text AI assistant, driven by Claude Code, OpenAI Codex, or Grok.",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
)
console = Console()
ACCENT = "bright_cyan"
NAME = f"[bold {ACCENT}]Vision[/bold {ACCENT}]"

_EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "goodbye vision", "bye vision", "that's all", "that is all", "stop listening"}


# ---------------------------------------------------------------- helpers
RETIRED_NOTES: list[str] = []  # from the last _cfg: saved models their provider no longer lists


def _cfg(model: Optional[str], voice: Optional[str], effort: Optional[str] = None, *, quiet: bool = False) -> Config:
    """Load config.toml with the command-line overrides. A saved model that its provider has retired runs as
    that provider's default (models.replace_retired_models); the notes land in RETIRED_NOTES and, unless
    `quiet` (the chat screen shows them itself), are printed here."""
    cfg = load_config()
    RETIRED_NOTES[:] = replace_retired_models(cfg, brain=not model)
    if not quiet:
        for n in RETIRED_NOTES:
            console.print(f"[yellow]{n}[/yellow]")
    if model:
        cfg.brain.model = model
    if effort:
        effort = effort.lower()
        if not supports_effort(cfg.brain.model, effort):
            allowed = ", ".join(v for v, _, _ in effort_choices(cfg.brain.model)) or "none"
            raise typer.BadParameter(f"{model_label(cfg.brain.model)} supports effort: {allowed}")
        cfg.brain.effort = effort
    elif model:
        cfg.brain.effort, _ = coerce_effort(cfg.brain.model, cfg.brain.effort)
    if voice:
        cfg.voice.voice = voice
    return cfg


def _brain(cfg: Config, voice_mode: bool, cont: bool, new: bool):
    from vision.brain import create_brain

    return create_brain(cfg.brain, voice_mode=voice_mode, continue_session=cont and not new)


def _turn_brain(brain, conversation, from_voice: bool, text: str = "", talk: bool = False):
    """Input origin selects the model; enabling playback never changes that routing. While a voice
    conversation is on (`talk`), typed turns go to the conversation model too, so one model answers
    the whole chat. Otherwise a typed turn goes to the front end when the router is on (`[router]
    typed`), when the message is one of its commands (/agent, /local, /search, /weather, /cancel) or
    when one was set for the next message."""
    from vision.routing import Override, parse_command

    conversation.agent = brain
    conversation.next_channel = "voice" if from_voice else "text"
    if hasattr(conversation, "follow"):
        conversation.follow()  # the voice model is the chat's own pick whenever it can be
    if from_voice or talk:
        return conversation
    rc = getattr(getattr(conversation, "cfg", None), "router", None)
    if rc is not None and getattr(rc, "mode", None) == "on" and getattr(rc, "typed", False) is True:
        return conversation
    if isinstance(getattr(conversation, "pending_override", None), Override):
        return conversation
    try:
        if text and rc is not None and parse_command(text, rc) is not None:
            return conversation
    except Exception:  # noqa: BLE001  (a malformed command: the front end reports it)
        return conversation
    return brain


def _split_model_arg(arg: str) -> tuple[str, bool]:
    """Accept the legacy 'full' suffix; model switches always carry the transcript."""
    words = arg.split()
    full = any(w.lower() == "full" for w in words[1:])
    return (words[0] if words else ""), full


def _switch_model(cfg: Config, brain, model: str, voice_mode: bool, effort: str | None = None, full: bool = False):
    """Apply a model (and optionally effort), replacing the driver when providers differ.

    Same-provider switches keep the existing session. Cross-provider switches read the saved
    conversation text without calling the outgoing model. `full` is a compatibility-only argument.
    """
    from vision.brain import create_brain, local_handoff

    old_provider = brain.provider
    new_provider = provider_for(model)
    changed = model != cfg.brain.model
    transcript = getattr(brain, "handoff", None)
    if new_provider != old_provider and brain.session_id:
        transcript = local_handoff(brain)
    cfg.brain.model = model
    if new_provider == "claude" and model != "opus" and cfg.brain.fast:
        cfg.brain.fast = False  # Claude fast mode is Opus-only; /model away from Opus turns it off.
        fast_note = "fast mode off (Claude fast mode uses Opus)"
    else:
        fast_note = ""
    if effort is not None:
        cfg.brain.effort = effort
    cfg.brain.effort, effort_note = coerce_effort(model, cfg.brain.effort)
    session_note = ""
    if new_provider != old_provider:
        brain.cancel()
        brain = create_brain(cfg.brain, voice_mode=voice_mode)
        session_note = f"new {provider_label(new_provider)} conversation"
        session_note += " from the previous transcript" if transcript else ""
    elif changed and brain.session_id:
        session_note = "carrying the whole transcript (the new model reads it once)"
    if changed and getattr(brain, "context", None) and hasattr(brain, "context_window"):
        brain.context = (brain.context[0], brain.context_window())  # the gauge's size is the new model's
    brain.handoff = transcript
    notes = [n for n in (fast_note, effort_note, session_note) if n]
    return brain, "; ".join(notes)


def _apply_session(brain, info) -> str:
    """Resume `info` on this brain. Same provider keeps the native thread; another provider
    starts a fresh session here and injects that transcript on the next message."""
    from vision.sessions import format_transcript, session_history

    if info.provider == brain.provider:
        if info.id == brain.session_id:
            return "already in that conversation"
        brain.resume(info.id)
        return f"resumed “{info.title}” · {info.age()} · {info.short_id}"
    summary = format_transcript(session_history(info.provider, info.id, limit=0, include_context=True)) or None
    brain.new_session()
    brain.handoff = summary
    src = provider_label(info.provider)
    extra = " from that transcript" if summary else ""
    return f"continuing “{info.title}” from {src}{extra} · {info.short_id}"


def _cd_target(arg: str, current: str, previous: str | None) -> str:
    """Resolve /cd's argument like a shell: '~' or '~/x' is under home, '-' is the previous directory,
    anything else is a path (relative to the current one). Must be an existing directory."""
    word = arg.strip()
    if word == "-":
        if not previous:
            raise ValueError("no previous directory yet")
        word = previous
    target = os.path.expanduser(word)
    if not os.path.isabs(target):
        target = os.path.join(current, target)
    target = os.path.normpath(target)
    if not os.path.isdir(target):
        raise ValueError(f"not a directory: {short_path(target)}" if os.path.exists(target) else f"no such directory: {short_path(target)}")
    if not os.access(target, os.R_OK | os.X_OK):
        raise ValueError(f"cannot enter {short_path(target)}")
    return target


def _fast_value(arg: str, current: bool) -> bool:
    """Resolve /fast's optional on/off argument; bare /fast toggles."""
    word = arg.strip().lower()
    if not word:
        return not current
    if word in ("on", "true", "1"):
        return True
    if word in ("off", "false", "0"):
        return False
    raise ValueError("/fast, /fast on or /fast off")


def _set_fast(cfg: Config, brain, enabled: bool, voice_mode: bool):
    """Apply native fast mode, including Claude Code's automatic move to Opus."""
    detail = ""
    if enabled and brain.provider == "claude" and cfg.brain.model != "opus":
        brain, detail = _switch_model(cfg, brain, "opus", voice_mode=voice_mode, full=True)
    cfg.brain.fast = enabled
    return brain, detail


def _switch_spoken(cfg: Config, brain, model: str, effort: str | None = None, full: bool = False):
    """Switch models in the plain-console modes, preserving the conversation."""
    return _switch_model(cfg, brain, model, voice_mode=False, effort=effort, full=full)


def _speaker(cfg: Config):
    from vision.tts import Speaker

    return Speaker(cfg.voice)


def _status(msg: str):
    console.print(f"  [dim]{msg}[/dim]")


def _token_stats(usage: dict | None) -> str:
    """`in 12,345 (11,900 cached) · out 456` from one turn's usage; "" when the provider gave none.
    Claude Code counts cache reads/writes apart from `input_tokens`; Codex folds `cached_input_tokens`
    into its `input_tokens`. Both end up as: everything the model read, with the cached part noted."""
    if not usage:
        return ""
    if "cached_input_tokens" in usage:  # Codex
        total, cached = usage.get("input_tokens", 0), usage["cached_input_tokens"]
    else:  # Claude Code
        cached = usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
        total = usage.get("input_tokens", 0) + cached
    out = usage.get("output_tokens", 0)
    return f"in {total:,}" + (f" ({cached:,} cached)" if cached else "") + f" · out {out:,}"


def _footer(elapsed: float, turn, cancelled: bool = False) -> str:
    """The dim line under a reply: `4.8s · in 12,345 (11,900 cached) · out 456`, with `cancelled`
    after the time when the turn was cut short (the counts are then what had arrived by that point)."""
    return " · ".join(p for p in (f"{elapsed:.1f}s", "cancelled" if cancelled else "", _token_stats(turn.usage if turn else None)) if p)


def _failed_turn_delta(text: str | None, error: str, markdown: bool = True) -> str:
    """Extra line for a failed turn, or '' if the reply already is that error.

    Claude streams session-limit (and similar) as assistant text and again as result.error;
    wrapping the same string would print it twice.
    """
    err = (error or "").strip()
    if not err:
        return ""
    body = (text or "").strip()
    wrapped = f"Vision could not answer: {err}"
    marked = f"**Vision could not answer:** {err}"
    if body in (err, wrapped, marked):
        return ""
    note = marked if markdown else wrapped
    return (f"\n\n{note}" if body else note)


def _load_voice(speaker, vc) -> None:
    """Load the voice, then make its filler clips ("One sec.") while it is idle: they are cached on disk,
    so this costs a few seconds once per voice. A clip that fails is simply not available."""
    speaker._load()
    if vc.filler:
        speaker.prepare_fillers(vc.filler_phrases + vc.filler_later_phrases)


def _arm_filler(ss, speaker, vc) -> None:
    """Have this spoken reply say a filler if its first words are late (see StreamingSpeaker.arm_filler).
    Only clips already made are used: a voice whose fillers are not ready yet just stays quiet."""
    if not vc.filler or vc.filler_after_ms <= 0:
        return
    try:
        ss.arm_filler(speaker.fillers(vc.filler_phrases), vc.filler_after_ms / 1000,
                      later=speaker.fillers(vc.filler_later_phrases), again_s=vc.filler_again_ms / 1000)
    except Exception:  # noqa: BLE001  a filler is a nicety; never let it break a turn
        pass


def _run_turn(brain, text: str, speaker=None, markdown: bool = True, timing=None, voice_cfg=None):
    """One brain turn rendered in the transcript, optionally spoken. Returns Turn or None if cancelled.
    `voice_cfg` (a spoken request): the reply says a filler if its first words are late."""
    ss = None
    if speaker is not None:
        from vision.tts import StreamingSpeaker

        ss = StreamingSpeaker(speaker, **({"timing": timing} if timing else {}))
        if voice_cfg is not None:
            _arm_filler(ss, speaker, voice_cfg)
    started = time.monotonic()
    with ReplyView(console, markdown=markdown, gate=(lambda: ss.spoken) if ss else None) as view:

        def on_text(delta: str):
            if view.status:
                view.status = ""
            view.append(delta)
            if ss:
                ss.feed(delta)  # sentence by sentence; a voice reply streams in as the model writes it

        def on_status(tool: str):
            if ss and tool:
                ss.flush()  # the text block is over: say its last sentence while the tool runs
            view.set_status(status_label(tool))

        turn = None
        try:
            turn = brain.ask(text, on_text=on_text, on_status=on_status, on_question=view.ask, on_agent=view.set_agent,
                             **({"timing": timing} if timing else {}))
        except KeyboardInterrupt:
            if ss:
                ss.stop()
            brain.cancel()
        cancelled = turn is None or turn.error == "cancelled"
        if turn is not None and turn.is_error and not cancelled:
            extra = _failed_turn_delta(turn.text, turn.error, markdown)
            if extra:
                view.append(extra)
        if ss:
            try:
                ss.finish()  # after a cancel (Ctrl-C, a cut-in) this only winds its threads down and closes the stream
            except KeyboardInterrupt:
                ss.stop()
                ss.finish()
    if timing:
        timing.event("turn_finished")
        timing.write()
    _status(_footer(time.monotonic() - started, turn, cancelled))
    return turn


# ---------------------------------------------------------------- versions / updates
def _update_cell(check: clis.UpdateCheck | None):
    """'up to date' / '2.1.280 available' / dim reason, for the versions table."""
    if check is None:
        return Text("")
    if check.available is None:
        return Text(f"? {check.error or 'unknown'}"[:60], style="dim")
    if check.available:
        return Text(f"{check.latest} available", style="yellow")
    return Text("up to date", style="green")


def _versions_renderable(
    infos: list[clis.CliInfo] | None = None, providers=clis.PROVIDERS, vision: bool = True, checks: dict[str, clis.UpdateCheck] | None = None, check: bool = True
):
    """Vision's own version and the provider CLIs it drives (all three by default), as one small table.
    With `check`, an update column says whether a newer release is out (given `checks`, or looked up)."""
    infos = clis.cli_versions(providers) if infos is None else infos
    if check and checks is None:
        checks = {i.provider: clis.check_update(i.provider) for i in infos if i.ok}
    t = Table(header_style=ACCENT, show_edge=False)
    t.add_column("tool")
    t.add_column("version")
    if check:
        t.add_column("update")
    t.add_column("where", style="dim")
    if vision:
        t.add_row("Vision", __version__, *([""] if check else []), short_path(str(Path(__file__).resolve().parent)))
    for i in infos:
        if i.ok:
            t.add_row(i.label, i.version, *([_update_cell((checks or {}).get(i.provider))] if check else []), short_path(i.path or ""))
        else:
            t.add_row(i.label, Text("missing", style="red"), *([""] if check else []), Text(i.error or "", style="red"))
    return t


def _update_summary(results: list[clis.UpdateResult]) -> str:
    """One line per CLI, coloured by outcome, for the chat transcript or the terminal."""
    lines = []
    for r in results:
        if r.error or not r.ok:
            lines.append(f"[red]✗[/red] {escape(r.summary())}")
        elif r.changed:
            lines.append(f"[green]✓[/green] {escape(r.summary())}")
        else:
            lines.append(f"[dim]·[/dim] {escape(r.summary())}")
    return "\n".join(lines)


@app.command()
def version(
    which: Optional[list[str]] = typer.Argument(None, help="all (default), or any of vision, claude, codex, grok."),
):
    """Show Vision's version and the Claude Code, Codex and Grok CLI versions (same as --version)."""
    try:
        show_vision, providers = clis.parse_version_arg(" ".join(which or []))
    except ValueError as e:
        raise typer.BadParameter(str(e))
    console.print(_versions_renderable(providers=providers, vision=show_vision))


@app.command()
def update(
    provider: Optional[list[str]] = typer.Argument(None, help="all (default), or any of claude, codex, grok."),
):
    """Update the provider CLIs (Claude Code, Codex, Grok) with their own updaters."""
    try:
        which = clis.parse_update_arg(" ".join(provider or []))
    except ValueError as e:
        raise typer.BadParameter(str(e))
    results = []
    for p in which:
        console.rule(f"[{ACCENT}]{provider_label(p, cli=True)}[/{ACCENT}]", style="dim")
        results.append(clis.update_cli(p, stream=True))
    console.rule(style="dim")
    console.print(_update_summary(results))
    if any(not r.ok for r in results):
        raise typer.Exit(1)


# ---------------------------------------------------------------- usage
_USAGE_LINE = re.compile(r"^(?P<label>[^:]+):\s+(?P<pct>\d+)% used(?:\s+·\s+resets\s+(?P<reset>.+))?$")


def _bar(pct: float) -> str:
    return usage_ui.bar(pct)


def _usage_renderable(text: str, full: bool = False):
    """Claude Code's /usage report (session, week, per-model such as Fable) as bars."""
    t = usage_ui.usage_table("Claude")
    rest = []
    for line in text.splitlines():
        m = _USAGE_LINE.match(line.strip())
        if m:
            label = m.group("label")
            if label == "Current week (all models)":
                label = "Current week"
            elif label == "Current week (Fable)":
                label = "Fable"
            usage_ui.add_window(t, label, float(m.group("pct")), (m.group("reset") or "").replace(" (America/New_York)", "") or "?")
        else:
            rest.append(line)
    usage_ui.add_banked(t, usage_ui.claude_banked())
    parts = [t]
    if full:
        parts.append(Text("\n".join(l for l in rest if l.strip() and not l.startswith("You are currently")), style="dim"))
    return usage_ui.usage_group(*parts)


def _render_usage_text(text: str, full: bool = False) -> None:
    console.print(_usage_renderable(text, full))


def _usage_fallback_renderable(usage: dict | None):
    """Fallback using the rate-limit event attached to the last reply."""
    if not usage:
        return Text("No usage data yet. Ask Vision something first, or run `vision usage`.", style="yellow")
    info, at = usage.get("info", {}), usage.get("at", 0)
    wins = info.get("unifiedWindows") or {}
    t = usage_ui.usage_table("Claude")
    labels = {"five_hour": "Current session", "seven_day": "Current week", "seven_day_overage_included": "Current week (incl. extra usage)"}
    for key, w in wins.items():
        usage_ui.add_window(t, labels.get(key, key), float(w.get("utilization") or 0) * 100, w.get("resetsAt"))
    usage_ui.add_banked(t, usage_ui.claude_banked())
    return usage_ui.usage_group(t, usage_ui.footer("from the last reply's rate-limit event", at or None))


def _render_usage(usage: dict | None, refreshed: bool):
    console.print(_usage_fallback_renderable(usage))


def _usage_for(brain, full: bool = False):
    if brain.provider != "claude":
        return brain.usage_renderable(full)
    text = brain.usage_report()
    return _usage_renderable(text, full) if text else _usage_fallback_renderable(brain.last_usage or brain.cached_usage())


_USAGE_PROVIDERS = ("claude", "codex", "grok")  # each one's throwaway brain runs models.provider_default


def _split_usage_arg(arg: str) -> tuple[str | None, bool]:
    """'/usage all full' → ("all", True); '/usage' → (None, False). Unknown words raise ValueError."""
    provider, full = None, False
    for w in arg.split():
        w = w.lower()
        if w == "full":
            full = True
        elif w == "all" or w in _USAGE_PROVIDERS:
            provider = w
        else:
            raise ValueError(f"/usage takes all, claude, codex or grok (and 'full'), not '{w}'")
    return provider, full


def _usage_brain(cfg: Config, brain, provider: str | None):
    """The current brain, or a throwaway one for another provider (resuming its last thread, so
    Codex can read its rollout) without switching what answers the chat."""
    if brain is not None and (provider is None or provider == brain.provider):
        return brain
    from dataclasses import replace

    from vision.brain import create_brain

    return create_brain(replace(cfg.brain, model=provider_default(provider)), continue_session=True)


def _usage_selection(cfg: Config, brain, provider: str | None, full: bool = False):
    """Render one provider, or every provider in a stable order for `/usage all`."""
    if provider != "all":
        return _usage_for(_usage_brain(cfg, brain, provider), full)

    parts = []
    for name in _USAGE_PROVIDERS:
        try:
            parts.append(_usage_for(_usage_brain(cfg, brain, name), full))
        except Exception as e:  # one unavailable CLI should not hide the other providers
            parts.append(Text(f"{provider_label(name)} usage unavailable: {e}", style="dim"))
        parts.append(Text(""))
    return usage_ui.usage_group(*parts[:-1])


def _show_usage(cfg: Config, brain, provider: str | None = None, full: bool = False) -> None:
    label = "all provider" if provider == "all" else provider_label((provider or brain.provider), cli=True)
    with console.status(f"[dim]reading {label} usage…[/dim]"):
        r = _usage_selection(cfg, brain, provider, full)
    console.print(r)


@app.command()
def usage(
    provider: str = typer.Argument("", help="all, claude, codex or grok (default: the configured model's provider)."),
    full: bool = typer.Option(False, "--full", help="Also show what has been contributing to usage."),
):
    """Show subscription usage for a provider."""
    cfg = load_config()
    try:
        which, _ = _split_usage_arg(provider)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    brain = _brain(cfg, voice_mode=False, cont=True, new=False)
    _show_usage(cfg, brain, which, full)


# ---------------------------------------------------------------- default: chat
@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    speak: bool = typer.Option(False, "--speak", "-s", help="Read replies aloud."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Claude alias, Codex slug, or Grok id."),
    effort: Optional[str] = typer.Option(None, "--effort", help="Reasoning effort (model-dependent)."),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice preset, name, or blend."),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the last Vision conversation."),
    wake: bool = typer.Option(False, "--wake", "-w", help="Listen for Vision's name from the start (config: wake.enabled)."),
    join: Optional[str] = typer.Option(None, "--join", "-j", help="Join a chat open on the phone (a chat id or prefix; `vision serve` must be running)."),
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
):
    """Start an interactive text chat with Vision (default when no command is given)."""
    if version:
        console.print(_versions_renderable())
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    chat(speak=speak, model=model, effort=effort, voice=voice, cont=cont, wake=wake, join=join)


def _model_label(brain) -> str:
    """Name of the model actually in use, e.g. 'Opus 5'."""
    return model_label(brain.resolved_model()) or brain.cfg.model


def _effort_word(effort: str) -> str:
    """How an effort reads in a status line: the level, "thinking off" for a local model's off switch
    (a bare "off" next to a model name reads as the model being off), "effort off" when there is none."""
    if effort == THINKING_OFF:
        return "thinking off"
    return effort or "effort off"


def _brain_summary(cfg: Config, brain, sep: str = " · ") -> str:
    """'Opus 5 · high · fast': the model actually in use, effort and optional speed tier."""
    parts = [_model_label(brain), _effort_word(cfg.brain.effort)]
    if cfg.brain.fast:
        parts.append("fast")
    return sep.join(parts)


def _voice_summary(cfg: Config, sep: str = " · ") -> str:
    """'Qwen 3.6 35B-A3B · thinking off': the conversation model that answers what you say ([conversation])."""
    model = cfg.conversation.model
    return sep.join([model_label(model) or model, _effort_word(cfg.conversation.effort)])


def _voicemodel_note(cfg: Config, choice: str, saved: bool) -> str:
    """What /voicemodel did: it names the voice for Codex and Grok chats; a Claude or Local chat keeps
    talking through its own model (VoiceConversation.voice_model)."""
    tail = " (saved)" if saved else ""
    if cfg.conversation.model == choice:
        return f"voice model → {_voice_summary(cfg)}{tail}"
    return (f"voice for Codex and Grok chats → {model_label(choice) or choice}{tail} · "
            f"this chat talks through its own model, {_voice_summary(cfg)}")


def _corner_label(cfg: Config, brain) -> str:
    """'Sonnet 5 (high) · fast': the model typed input goes to, in the input box's bottom-right
    border like the Grok CLI's `Grok 4.6 (xhigh)`. Effort off leaves the parenthesis out."""
    label = _model_label(brain)
    if cfg.brain.effort:
        label += f" ({_effort_word(cfg.brain.effort)})"
    if cfg.brain.fast:
        label += " · fast"
    return label


def _active_summary(cfg: Config, brain, talk: bool, sep: str = " · ") -> str:
    """What answers the next turn: the brain, or while talk is on the conversation model, which then
    takes typed turns as well as spoken ones (the brain is its worker)."""
    return _voice_summary(cfg, sep) if talk else _brain_summary(cfg, brain, sep)


FACTORY_MODEL, FACTORY_EFFORT = BrainConfig.model, BrainConfig.effort  # Opus 5 · high


def _saved_defaults() -> tuple[str, str]:
    saved = load_config().brain
    return saved.model, saved.effort


def _save_defaults(cfg: Config, model: str, effort: str) -> str:
    """Persist model + effort as Vision's defaults and apply them to this session. Returns a note."""
    effort, note = coerce_effort(model, effort)
    save_brain_defaults(model, effort)
    cfg.brain.model, cfg.brain.effort = model, effort
    out = f"default saved → {model_label(model) or model} · {effort or 'effort off'}"
    return f"{out}; {note}" if note else out


ROUTE_COMMANDS = ("agent", "opus", "codex", "local", "search", "weather", "cancel")  # the front end's commands (vision/routing.py)

HELP_TEXT = (
    "[bold]/model[/bold] pick Claude, Codex or Grok for this session (/model sonnet; /model grok-4.6; /model default → saved model + effort; "
    "same provider: keeps the whole transcript in the existing session; "
    "crossing providers: Vision carries the previous transcript from disk) · "
    "[bold]/effort[/bold] pick a model-supported effort · "
    "[bold]/fast[/bold] toggle faster, higher-usage inference (/fast on or /fast off) · "
    "[bold]/default[/bold] choose and save the default model and effort (/default reset → Opus 5 · high)\n"
    "[bold]/speak[/bold] toggle spoken replies · [bold]/voice <name>[/bold] change voice\n"
    "[bold]/listen[/bold] say one turn with the mic · [bold]/talk[/bold] voice conversation on/off "
    "(hands-free, right here in the chat; type to answer instead; Esc, /talk or “goodbye” ends it) · "
    "[bold]/wake[/bold] wake word on/off: say “Vision” whenever the chat is idle and it warms up and listens "
    "(“Vision, what's the weather?” runs straight away; it dozes off again after a quiet moment) · "
    "[bold]/mic[/bold] pick the microphone (/mic <name> or /mic default; saved to config, and a running /talk or /wake moves over) · "
    "[bold]/say <text>[/bold] speak text\n"
    "[bold]/mode[/bold] (or Shift-Tab) switch between [bold]auto[/bold] (every tool runs, no approvals; denied_tools still apply) "
    "and [bold]plan[/bold] (read-only: Vision proposes a plan and carries it out once you approve it) · shown bottom left\n"
    "[bold]/cd <dir>[/bold] change the working directory for this session (/cd ~, /cd .., /cd - for the previous one; "
    "the brains work there from the next reply on; bare /cd says where you are) · "
    "[bold]/new[/bold] fresh conversation · [bold]/session[/bold] pick an earlier conversation "
    "(Claude, Codex and Grok tabs; same provider resumes that thread, another provider continues it here) "
    "(/session <id> resumes one; /session id shows the current id) · "
    "[bold]/usage[/bold] subscription usage (/usage all for every provider; claude, codex or grok for one; add 'full' for detail) · "
    "[bold]/version[/bold] Vision and the Claude Code, Codex and Grok CLI versions (/version codex for one) · "
    "[bold]/update[/bold] update those CLIs with their own updaters (/update all, or claude, codex, grok) · "
    "[bold]/memory[/bold] show Vision's own memory (shared by every brain; the brains add to it themselves) · "
    "[bold]/remember <fact>[/bold] add a line by hand · [bold]/forget <text>[/bold] drop the lines containing it · "
    "[bold]/clear[/bold] clear the screen · [bold]/quit[/bold]\n"
    "[bold]/agent opus|codex [--effort low|medium|high|…] [request][/bold] have that agent do the request (or the next message); "
    "in words works too: “use Opus high for this”, “have Codex handle this” · "
    "[bold]/local[/bold] answer with the conversation model alone (no agent) · [bold]/search[/bold] a search-only web lookup · "
    "[bold]/weather [place][/bold] · [bold]/cancel[/bold] stop the running or waiting agent · "
    "[bold]/router off|audit|on [save][/bold] the front end: off = the voice model decides delegation itself; audit = routes are logged "
    "and shown, nothing changes; on = basic questions stay with the conversation model, the weather and current facts come as data, "
    "everything else goes to the default agent (Opus 5 · medium, high for hard tasks) · "
    "[bold]/voicemodel [model] [save][/bold] the voice for Codex and Grok chats (a Claude, or a Local model on the llama-server; "
    "a Claude or Local chat talks through its own /model; /voicemodel qwen3.6 save keeps it as the default)\n"
    "[dim]Type / for the command menu (↑/↓ choose · Enter runs · Tab fills in · Esc hides) · "
    "Enter sends (mid-reply it queues) · Ctrl-X sends it into the running reply now (empty box: the queued ones) · "
    "↑ on an empty box picks a queued message: Enter edits, Del removes, Esc leaves (the queue waits meanwhile) · "
    "Ctrl-J newline · PgUp/PgDn or Shift-↑/↓ scroll (Alt-End follows again) · "
    "Esc cancels a reply · Ctrl-O unfolds the last reply's tool calls · Ctrl-D quits[/dim]"
)


def _mic_rows() -> list[tuple[str, str, str]]:
    """Input devices for the /mic menu; [] when audio can't be queried (the command itself says why)."""
    from vision.config import input_device_choices

    try:
        return input_device_choices()
    except Exception:  # noqa: BLE001
        return []


def _menu_commands(cfg: Config, brain_ref: Callable[[], object]) -> list[SlashCommand]:
    """The /commands offered by the chat screen's pop-up menu, with argument choices where useful."""
    from vision.routing import agent_model

    def models():
        rows = [("default", "saved model + effort")]
        for tab, entries, _ in MODEL_TABS:
            rows += [(v, f"{label} · {tab} · {desc}") for v, label, desc in entries]
        return rows

    def efforts():
        return [(v, d) for v, _, d in effort_choices(cfg.brain.model) if v]

    def sessions():
        from vision.sessions import PROVIDER_TITLES, list_all_sessions

        return [
            (s.short_id, f"{s.title} · {PROVIDER_TITLES[s.provider]} · {s.age()}")
            for s in list_all_sessions()
        ]

    def dirs():
        rows = [("~", "home"), ("-", "previous directory"), ("..", "up one level")]
        try:
            names = sorted(e.name for e in os.scandir(brain_ref().workdir) if e.is_dir() and not e.name.startswith("."))
        except OSError:
            names = []
        return rows + [(n, "") for n in names[:60]]

    return [
        SlashCommand("model", "switch model for this session", models),
        SlashCommand("effort", "reasoning effort for this session", efforts),
        SlashCommand("fast", "toggle faster, higher-usage inference", lambda: [("on", "use the fast service tier"), ("off", "use standard inference")]),
        SlashCommand("default", "choose and save the default model + effort", lambda: [("reset", "back to Opus 5 · high")]),
        SlashCommand("session", "resume an earlier conversation, or join one open on the phone", sessions, aliases=("sessions", "resume")),
        SlashCommand("cd", "change the working directory (~, .., - or a path)", dirs),
        SlashCommand("new", "start a fresh conversation"),
        SlashCommand("mode", "auto or plan mode (Shift-Tab toggles)", lambda: [("auto", "every tool runs, no approvals"), ("plan", "read-only until you approve the plan")], aliases=("auto", "plan")),
        SlashCommand("usage", "subscription usage", lambda: [("all", "every provider"), ("claude", "Claude subscription windows"), ("codex", "Codex subscription windows"), ("grok", "Grok weekly allowance"), ("full", "also show what has been contributing")]),
        SlashCommand("version", "Vision and the Claude Code, Codex and Grok versions", lambda: [("all", "Vision and every CLI"), ("claude", "Claude Code only"), ("codex", "Codex only"), ("grok", "Grok only"), ("vision", "Vision only")], aliases=("versions",)),
        SlashCommand("update", "update the provider CLIs with their own updaters", lambda: [("all", "Claude Code, Codex and Grok"), ("claude", "Claude Code only"), ("codex", "Codex only"), ("grok", "Grok only")]),
        SlashCommand("memory", "show Vision's own memory (shared by every brain)"),
        SlashCommand("remember", "add a fact to Vision's memory by hand"),
        SlashCommand("forget", "drop the memory lines containing some text"),
        SlashCommand("speak", "toggle spoken replies"),
        SlashCommand("voice", "change the voice", lambda: voice_choices(cfg.voice)),
        SlashCommand("say", "speak some text aloud"),
        SlashCommand("listen", "say one turn with the mic"),
        SlashCommand("talk", "voice conversation on/off (Esc or “goodbye” ends it)"),
        SlashCommand("mic", "choose the microphone (saved to config)", lambda: [(v or "default", d if v else "system default") for v, _, d in _mic_rows()]),
        SlashCommand("wake", "wake word on/off: say “Vision” to start talking", lambda: [("on", "listen for the name whenever idle"), ("off", "stop listening for it")]),
        SlashCommand("agent", "have an agent do the request: /agent opus --effort high <request>", lambda: [(n, f"→ {model_label(agent_model(cfg.router, n)) or '?'}") for n in cfg.router.agents], aliases=("opus", "codex")),
        SlashCommand("local", "answer with the conversation model alone, no agent"),
        SlashCommand("search", "a search-only web lookup (results as data, no page opened)"),
        SlashCommand("weather", "the weather from Apple WeatherKit (/weather <place>)"),
        SlashCommand("cancel", "stop the agent that is running or waiting for an answer"),
        SlashCommand("voicemodel", "the voice for Codex and Grok chats: a Claude or a Local model (add save to keep it)",
                     lambda: [(v, f"{l} · {d}") for _, rows, _ in CONVERSATION_TABS for v, l, d in rows]),
        SlashCommand("router", "the front end: off, audit or on (add save to keep it)", lambda: [("off", "the voice model decides delegation itself"), ("audit", "log and show routes, change nothing"), ("on", "enforce: basic → local, weather, search, else the default agent")]),
        SlashCommand("clear", "clear the screen"),
        SlashCommand("help", "show the command list"),
        SlashCommand("quit", "leave Vision", aliases=("exit", "q")),
    ]


SHOWN_HISTORY = 200  # most turns of a resumed conversation replayed into the transcript


def chat(speak: bool, model: Optional[str], effort: Optional[str], voice: Optional[str], cont: bool, wake: bool = False, join: Optional[str] = None):
    from vision.brain import PLAN_QUESTION
    from vision.config import STATE_DIR
    from vision.conversation import VoiceConversation
    from vision.sessions import LiveTitle

    cfg = _cfg(model, voice, effort, quiet=True)
    brain = _brain(cfg, voice_mode=False, cont=cont, new=False)
    conversation = VoiceConversation(cfg, agent=brain)
    state = {"speak": speak, "speaker": None, "mic": None, "stt": None, "ss": None}
    # Voice modes run on a worker thread inside the screen (see voice_loop): `talk` is on while a
    # conversation is running, `hearing` is the mic status for the status row, `typed` carries a
    # message typed while the mic was open, and `cancel` stops the current recording.
    state.update({"talk": False, "once": False, "hearing": "", "typed": None, "cancel": None, "thread": None})
    # The wake word (see wake_loop): `wake` is the switch, `wake_thread`/`wake_cancel` the listener
    # waiting for the name; the utterance that woke it is handed to voice_loop as `woken`.
    state.update({"wake": False, "wake_thread": None, "wake_cancel": None, "wake_listener": None, "quitting": False, "update_checks": set()})
    # Cutting into a spoken reply by voice (`barge_in` in [listen]) uses the same small listener as the
    # wake word; `cut_in` turns False after it fails once, so a dead mic is reported once, not every turn.
    state.update({"listener_lock": threading.Lock(), "cut_in": cfg.listen.barge_in != "off"})
    state["prev_dir"] = None  # where /cd came from, for /cd -
    # Typed turns use one serial worker.  The input remains live while a reply is running, but the
    # provider CLIs still see exactly one turn at a time and therefore keep session order intact.
    import queue

    from vision.turnqueue import QueuedTurn, TurnQueue

    state.update({"turn_queue": TurnQueue(), "turn_lock": threading.Lock(), "turn_active": False, "turn_thread": None})
    # The running turn's driver and its `steered(text)` (splits the reply where a Ctrl-X message went
    # in). Messages sent meanwhile wait in the queued strip above the input box (vision.turnqueue).
    state.update({"driver": None, "steered": None})
    # `vision serve` finds this chat through its socket (link.py) and shows it on the phone as a
    # terminal chat: messages from there run here, and every turn here streams there.
    state["link"] = None
    # Joined to a chat `vision serve` runs (remote.py): `home` is the local brain (and its model,
    # effort, directory) to return to when the chat closes or /new leaves it.
    state.update({"server": None, "home": None})
    if speak:
        state["speaker"] = _speaker(cfg)
        threading.Thread(target=state["speaker"]._load, daemon=True).start()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    def speaker():
        if state["speaker"] is None:
            state["speaker"] = _speaker(cfg)
        return state["speaker"]

    def update_status() -> str:
        """The current provider's CLI: 'Codex 0.156.0 available · /update', or nothing when it is current.
        Reads the cached check; the first look at a provider starts one in the background, so the
        status bar never blocks on the network (and says nothing until the answer is in)."""
        provider = brain.provider
        check = clis.peek_check(provider)
        if check is not None:
            return check.status()
        if provider not in state["update_checks"]:
            state["update_checks"].add(provider)

            def go():
                try:
                    clis.check_update(provider)
                finally:
                    state["update_checks"].discard(provider)
                    screen.app.invalidate()
            background(go)
        return ""

    def corner_label() -> str:
        if state["talk"] and not state["once"]:
            return _voice_summary(cfg, " ")
        return _corner_label(cfg, brain)

    def status_line() -> str:
        # The model that answers sits in the input box's corner (see corner_label): the brain, or the
        # conversation model while talking, which then answers typed turns too.
        parts = []
        if state["talk"]:
            parts.append(f"talk · {state['hearing'] or 'replying'}")
        elif state["speak"]:
            parts.append("speech on")  # off is the default and says nothing
        if cfg.router.mode != "off":
            parts.append(f"router {cfg.router.mode}")
        parts.append(short_path(brain.workdir))
        if (u := update_status()) and not state["talk"]:
            parts.append(u)
        return "  ·  ".join(parts)

    def wake_status() -> str:  # bottom-right corner: `34% ctx  ·  cache 58:12  ·  wake off` (the figures the brain has)
        parts = []
        ctx = getattr(brain, "context", None)
        if ctx and ctx[1]:
            parts.append(f"{min(100, round(100 * ctx[0] / ctx[1]))}% ctx")
        if brain.provider == "claude" and (cache := cachettl.label(cachettl.seconds_left(brain.session_id, brain.resolved_model(), time.time()))):
            parts.append(cache)  # repainted every refresh_interval, so it ticks
        parts.append("wake on" if state["wake"] else "wake off")
        return "  ·  ".join(parts)

    buddy = Buddy(cfg.buddy.name) if cfg.buddy.enabled else None
    screen = ChatScreen(str(STATE_DIR / "history"), status_line, buddy=buddy, mode=lambda: cfg.brain.mode, status_right=wake_status,
                        corner=corner_label)
    if buddy:
        buddy.speaking_fn = lambda: bool(state["ss"] and state["ss"].speaking)
    screen.commands = _menu_commands(cfg, lambda: brain)
    live_title = LiveTitle()
    screen.title_fn = lambda: live_title.get(brain.provider, brain.session_id)  # the terminal tab

    def load_arrow_history(provider: str | None = None, sid: str | None = None):
        """Up/Down follows this conversation's user turns, including a resumed session's."""
        from vision.sessions import user_turns

        p = provider or brain.provider
        s = brain.session_id if sid is None else sid
        screen.set_history(user_turns(p, s) if s else [])

    load_arrow_history()

    def note(msg: str, kind: str = "info"):
        """A system line in the transcript. A message wrapped whole in `[red]…[/red]` is an error,
        `[yellow]…[/yellow]` a warning: they get the `✗` / `⚠` mark instead of the info `·`."""
        for colour, k in (("red", "error"), ("yellow", "warn")):
            if msg.startswith(f"[{colour}]") and msg.endswith(f"[/{colour}]"):
                msg, kind = msg[len(colour) + 2 : -(len(colour) + 3)], k
        screen.add(notice_grid(msg, kind))

    def paint_open():
        """The session card: Pip + model / directory / voice. First block on open and after /clear."""
        screen.add(header_renderable(
            model=_active_summary(cfg, brain, state["talk"], " "),  # "Opus 5 high", like the status row
            directory=short_path(brain.workdir),
            voice=cfg.voice.voice if state["speak"] else "off",
            resumed=bool(brain.session_id),
        ))

    def show_history(provider: str, sid: str, limit: int = SHOWN_HISTORY):
        """A resumed conversation's earlier turns, in the transcript where they can be scrolled, not
        only in the model's memory. The transcript is cleared first so the chat reads as one thread,
        then the open card sits above those turns."""
        from vision.sessions import session_history

        paint_turns(session_history(provider, sid, limit=limit + 1), limit)

    def paint_turns(turns: list[dict], limit: int = SHOWN_HISTORY):
        with screen._lock:
            screen.entries.clear()
        paint_open()
        if len(turns) > limit:
            turns = turns[-limit:]
            note("earlier turns not shown…")
        for turn in turns:
            if turn["role"] == "user":
                screen.add(user_grid(turn["text"]), gap_before=True)
            else:
                marks = past_marks(turn)
                if marks:
                    screen.add_past_reply(turn["text"], marks)
                else:
                    screen.add(reply_grid(turn["text"]))

    def past_marks(turn: dict) -> list[tuple[int, object]]:
        """Rebuild the rows a past reply ran, in the order they appeared beside its text."""
        from vision.brain import ToolCall, run_from_frame
        from vision.remote import RemoteAgentRun

        marks = []
        for f in turn.get("agents") or []:
            run = run_from_frame({**f, "done": True}, RemoteAgentRun(f.get("id", ""), "agent", ""))
            run.failed, run.cut_off = bool(f.get("failed")), bool(f.get("cut_off"))
            run.status_text = f.get("status") or ""
            marks.append((int(f.get("at") or 0), run))
        for f in turn.get("tools") or []:
            call = ToolCall(f.get("id", ""), f.get("name", ""), f.get("detail", ""), done=True,
                            is_error=bool(f.get("is_error")), output=f.get("output") or "")
            marks.append((int(f.get("at") or 0), call))
        return marks

    if brain.session_id:  # --continue: the picked-up conversation is on screen from the start
        show_history(brain.provider, brain.session_id)
    else:
        paint_open()

    # ---- one reply, run on a worker thread so the UI keeps drawing
    def run_turn(text: str, from_voice: bool = False, timing=None, speak_remote: bool = False, follow: bool = False,
                 voice_remote: bool = False, talk_remote: bool = False):
        """`follow`: the turn is already running on the phone chat this screen is joined to; show it
        as it streams rather than sending the message (RemoteBrain.ask knows it is that turn).
        `voice_remote`: spoken on the phone; `talk_remote`: typed during a call on the phone. Both go
        to the conversation model, like talk mode here."""
        if from_voice and timing is None and cfg.conversation.timing:
            from vision.timing import VoiceTiming
            timing = VoiceTiming()
        link = state["link"]
        link_post = link.post if link else (lambda ev: None)
        driver = brain if follow else _turn_brain(brain, conversation, from_voice or voice_remote, text,
                                                        talk=(state["talk"] and not state["once"]) or talk_remote)
        state["driver"] = driver
        ss = None
        if state["speak"] or (state["talk"] and not state["once"]):  # a voice conversation always talks back
            from vision.tts import StreamingSpeaker

            ss = StreamingSpeaker(speaker(), timing=timing)
            if from_voice:
                _arm_filler(ss, speaker(), cfg.voice)  # "One sec." if the first words are late
        state["ss"] = ss
        entry = screen.start_reply(markdown=not from_voice)
        entry.tokens_fn = lambda: getattr(brain, "output_tokens", 0)  # the `↓ 1.2k` on the live line (Claude Code only)
        if ss:
            entry.gate = lambda: ss.spoken_position  # keep the sub-character timing for smooth frames
        started = time.time()
        link_post({"type": "start", "text": text, "speak": speak_remote})
        if link:
            link.post_summary(force=True)

        streamed = []  # the reply as it streamed (where steered messages split it, what agents keep)
        homes = {}  # agent id → the reply entry its row lives in (a steered message splits the reply)
        tool_homes = {}  # a call that finishes after a split still belongs to its starting reply
        agent_frames = {}
        tool_frames = {}
        steers = []

        def on_text(d):
            streamed.append(d)
            screen.update_reply(entry, delta=d)
            if ss:
                ss.feed(d)  # sentence by sentence; a voice reply streams in as the model writes it
            link_post({"type": "delta", "text": d})

        def steered(t):
            # A message went into the running turn (Ctrl-X here, or from the phone): it sits in the
            # transcript where it went in, and the rest of the reply follows it (as Claude Code shows it).
            nonlocal entry
            steers.append((len("".join(streamed)), t, len(agent_frames)))
            entry = screen.split_reply(entry, user_grid(t))
            if ss:
                ss.flush()
            link_post({"type": "steered", "text": t})

        state["steered"] = steered

        def on_status(tool):
            if ss and tool:
                ss.flush()  # the text block is over: say its last sentence while the tool runs
            screen.update_reply(entry, status=status_label(tool), force=True)
            link_post({"type": "status", "tool": tool, "label": status_label(tool)})

        def on_question(questions):
            nonlocal entry
            screen.update_reply(entry, status="waiting for your answer…", force=True)
            if buddy:
                buddy.listening()
            def opened():
                # Only once the form is up, so the summary says `waiting`; the phone drops its copy of the
                # form when a later summary says it no longer is (answered here).
                link_post({"type": "question", "questions": questions})
                if link:
                    link.post_summary(force=True)

            answers = screen.ask_questions(questions, on_open=opened)  # answered here or on the phone, whichever is first
            link_post({"type": "answered", "questions": questions, "answers": answers})  # the phone shows it in the reply
            if link:
                link.post_summary(force=True)
            # What was chosen goes into the transcript as a block of its own (as in Claude Code), and
            # what the answer leads to (the plan being carried out, the next step) is a fresh reply.
            blocks = [answered_grid(questions, answers)]
            if questions == [PLAN_QUESTION] and (answers or {}).get(PLAN_QUESTION["question"]) == "Yes":
                blocks.append(Text.from_markup("plan approved → [yellow]auto mode[/yellow] (Shift-Tab or /mode plan to plan again)"))
            entry = screen.split_reply(entry, *blocks)
            return answers

        def on_agent(run):
            if run.id in entry.agents:  # still at work when a question or steer split the reply: it moved down
                homes[run.id] = entry
            home = homes.setdefault(run.id, entry)
            if home is entry:
                screen.update_reply(entry, agent=run)
            else:  # it finished before a question or steered message split the reply: its row stays up there
                home.invalidate()
                screen.app.invalidate()
            frame = agent_frame(run)
            frame["at"] = agent_frames.get(run.id, {}).get("at", len("".join(streamed)))
            agent_frames[run.id] = frame
            link_post(dict(frame))

        def on_tool(call):
            home = tool_homes.setdefault(call.id, entry)
            if home is entry:
                screen.update_reply(entry, tool=call)
            else:
                home.place(call)
                home.invalidate()
                screen.app.invalidate()
            frame = tool_frame(call)
            frame["at"] = tool_frames.get(call.id, {}).get("at", len("".join(streamed)))
            tool_frames[call.id] = frame
            link_post(dict(frame))  # the phone draws it as a row too

        turn = None
        cancelled = False
        error = ""
        try:
            extra = {"timing": timing} if from_voice and timing else {}
            turn = driver.ask(text, on_text=on_text, on_status=on_status, on_question=on_question, on_agent=on_agent, on_tool=on_tool, **extra)
            cancelled = turn.error == "cancelled"
            if turn.is_error:
                error = turn.error
            if turn.is_error and not cancelled:
                extra = _failed_turn_delta(turn.text, turn.error)
                if extra:
                    screen.update_reply(entry, delta=extra, force=True)
                if buddy:
                    buddy.fail()
        except Exception as e:  # noqa: BLE001
            error = str(e)
            screen.update_reply(entry, delta=f"**Error:** {e}", force=True)
            if buddy:
                buddy.fail()
        finally:
            state["driver"] = state["steered"] = None
            if (agent_frames or tool_frames) and not joined():
                # The rows stay with the conversation (the server keeps a joined chat's itself).
                from vision import agentlog

                reply = (turn.text if turn is not None else "") or ""
                try:
                    agentlog.record_turn(brain.provider, (turn.session_id if turn is not None else None) or brain.session_id,
                                         agentlog.turn_entries(text, reply, "", "".join(streamed), list(agent_frames.values()), steers,
                                                               list(tool_frames.values())))
                except Exception:  # noqa: BLE001  (a history nicety must never break a turn)
                    pass
            link_post({
                "type": "done",
                "text": (turn.text if turn is not None else "") or "",
                "error": error,
                "busy": not state["turn_queue"].empty(),
                "session_id": (turn.session_id if turn is not None else None) or brain.session_id,
                "model": turn.model if turn is not None else None,
                "duration_ms": int((time.time() - started) * 1000),
            })
            # `cancelled` sits on the footer beside the time rather than in the reply text (the token
            # counts left the footer on 2026-09-18: the context figure is in the status row, /usage has the rest).
            if ss:
                try:
                    ss.finish()
                except Exception:
                    pass
            screen.end_reply(entry, cancelled=cancelled)
            state["ss"] = None
            if timing:
                timing.event("turn_finished")
                timing.write()
            audit_line(text, driver is conversation, "voice" if from_voice else "text")

    def audit_line(text: str, routed: bool, channel: str):
        """In audit mode, what the router would have done with this request: logged, and one dim line
        in the chat. A turn the front end handled has its route already; one that went straight to the
        brain is classified here so typed requests are reviewed too."""
        from vision import routing

        if cfg.router.mode != "audit":
            return
        route = conversation.last_route if routed else None
        if route is None:
            try:
                route = routing.route(text, cfg)
            except routing.OverrideError:
                return
            conversation.supervisor.log("route", route=route, channel=channel, mode="audit")
        note(f"router (audit): {route.describe(cfg.router)}")

    def cancel_reply():
        conversation.cancel()
        brain.cancel()
        if state["ss"]:
            state["ss"].stop()
        elif state["speaker"]:
            state["speaker"].stop()

    def background(fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def drain_turns():
        """Run typed follow-ups in submission order, never concurrently on the same brain. While a
        queued message is being picked or edited in the strip, the queue is held: nothing goes."""
        pending = state["turn_queue"]
        while not state["quitting"]:
            if not pending.wait_free(0.25):
                continue
            with state["turn_lock"]:
                item = pending.pop()
                if item is None:
                    state["turn_active"] = False
                    state["turn_thread"] = None
                    return
            if item.shown:
                screen.add(user_grid(item.text), gap_before=True)  # its turn now: into the transcript, above its reply
            screen.app.invalidate()
            run_turn(item.text, speak_remote=item.speak, follow=item.follow, voice_remote=item.voice, talk_remote=item.talk)
        with state["turn_lock"]:
            state["turn_active"] = False
            state["turn_thread"] = None

    def enqueue_turn(text: str, speak_remote: bool = False, follow: bool = False, voice: bool = False, talk: bool = False,
                     shown: bool = False):
        """`speak_remote`: the phone asked for this reply spoken; the server does that from the deltas.
        `follow`, `voice`, `talk`: see run_turn. `shown`: it waits in the queued strip (queue_or_add)."""
        pending = state["turn_queue"]
        with state["turn_lock"]:
            pending.put(QueuedTurn(text, speak_remote, follow, voice, talk, shown=shown))
            if state["turn_active"]:
                return
            state["turn_active"] = True
            thread = threading.Thread(target=drain_turns, daemon=True)
            state["turn_thread"] = thread
            thread.start()

    def switch(model: str, effort: str | None = None, full: bool = False, save: bool = False, prefix: str = "model → "):
        """Change model on a worker thread while preserving the conversation."""
        if screen.busy:
            note("still replying — switch models once it has finished (Esc cancels)")
            return
        if joined():
            # The server owns the session: it switches, then reports back with a note and a fresh summary.
            brain.switch_model(model, effort)
            if save:
                note(_save_defaults(cfg, model, effort if effort is not None else cfg.brain.effort))
            return
        screen.busy = True

        def go():
            nonlocal brain
            before = (cfg.brain.model, cfg.brain.effort)
            link = state["link"]
            try:
                brain, detail = _switch_model(cfg, brain, model, voice_mode=False, effort=effort, full=full)
                conversation.agent = brain
                conversation.follow()  # talk mode speaks through the new pick too
                msg = _save_defaults(cfg, cfg.brain.model, cfg.brain.effort) if save else f"{prefix}{_brain_summary(cfg, brain)}"
                note(f"{msg}; {detail}" if detail else msg)
                if link and (cfg.brain.model, cfg.brain.effort) != before:
                    # The phone mirrors this chat: give it the same one-line note a phone-side switch gets.
                    effort_word = _effort_word(cfg.brain.effort)
                    if cfg.brain.model != before[0]:
                        link.post({"type": "note", "text": f"Switched to {_model_label(brain)} ({effort_word})"})
                    else:
                        link.post({"type": "note", "text": f"Effort set to {effort_word}"})
            except Exception as e:  # noqa: BLE001
                note(f"[red]model switch failed: {e}[/red]")
                if link:
                    link.post({"type": "error", "text": f"model switch failed: {e}"})
            finally:
                screen.busy, screen.busy_label = False, ""
                if link:
                    link.post_summary(force=True)
        background(go)

    # ---- a chat open on the phone: this screen joins it and the server keeps running it (remote.py)
    def joined() -> bool:
        from vision.remote import RemoteBrain

        return isinstance(brain, RemoteBrain)

    def phone_chats() -> list[dict]:
        """The chats `vision serve` runs on this machine (none when it is not running)."""
        from vision.remote import list_remote_chats, running_server

        server = running_server()
        chats = []
        if server:
            try:
                chats = list_remote_chats(server)
            except Exception as e:  # noqa: BLE001  (the descriptor outlived the server, or it is starting)
                note(f"[yellow]vision serve is not answering: {e}[/yellow]")
                server = None
        state["server"] = server
        return chats

    def attach(info: dict):
        """Join a phone chat: what is typed here and on the phone reads as one conversation."""
        nonlocal brain
        from vision.remote import RemoteBrain, remote_history

        if screen.busy or state["turn_active"]:
            note("[yellow]wait for the current reply to finish (Esc cancels it), then /session again[/yellow]")
            return
        if joined() and brain.chat_id == info["chat"]:
            note("already in that conversation")
            return
        server = state["server"]
        before = (cfg.brain.model, cfg.brain.effort, cfg.brain.workdir)

        def foreign_turn(text: str):
            screen.add(user_grid(text), gap_before=True)
            note("from the phone")
            enqueue_turn(text, follow=True)

        def on_note(text: str, kind: str):
            note(f"[red]{text}[/red]" if kind == "error" else text)

        remote = RemoteBrain(
            server, info, cfg.brain,
            on_summary=lambda s: screen.app.invalidate(),  # the corner label, the ctx figure, the title
            on_foreign_turn=foreign_turn, on_note=on_note,
            on_answered=lambda: screen.answer_questions(None),
            on_closed=lambda reason: detach(reason),
            on_steered=lambda t: state["steered"](t) if state["steered"] else None,
        )
        try:
            remote.connect()
        except Exception as e:  # noqa: BLE001
            cfg.brain.model, cfg.brain.effort, cfg.brain.workdir = before
            note(f"[red]could not join that chat: {e}[/red]")
            return
        if joined():
            brain.close()
        else:
            state["home"] = (brain, *before)
        if state["link"]:
            state["link"].stop()  # not a chat of its own while it mirrors one the server runs
            state["link"] = None
        brain = remote
        conversation.agent = brain
        conversation.new_session()
        try:
            turns = remote_history(server, info["chat"])
        except Exception:  # noqa: BLE001
            turns = []
        paint_turns(turns)
        screen.set_history([t["text"] for t in turns if t.get("role") == "user" and t.get("text")])
        note(f"joined “{remote.title or 'new conversation'}” from the phone · typed here and there alike · /new leaves it")
        if (running := remote.pending_turn()) is not None:
            foreign_turn(running)  # joined mid-reply: follow it from here
        screen.app.invalidate()

    def detach(reason: str = ""):
        """Back to this terminal's own conversation (the phone chat carries on without us)."""
        nonlocal brain
        if not joined():
            return
        brain.close()
        home, state["home"] = state["home"], None
        if home:
            brain, cfg.brain.model, cfg.brain.effort, cfg.brain.workdir = home
        else:
            from vision.brain import create_brain

            brain = create_brain(cfg.brain, voice_mode=False)
        conversation.agent = brain
        conversation.new_session()
        load_arrow_history()
        if brain.session_id:
            show_history(brain.provider, brain.session_id)
        else:
            paint_turns([])
        start_link()
        note(f"left the phone chat{': ' + reason if reason else ''} · back to your own conversation")
        screen.app.invalidate()

    # ---- voice: the mic runs on a worker thread while the screen keeps drawing
    def ears():
        """The reusable mic plus the speech models loaded for this talk session."""
        conversation.warm_up()
        components = ("ears", "voice", "listener") if state["cut_in"] else ("ears", "voice")

        def report(progress):
            if state["talk"]:
                screen.busy_label = f"warming up ears and voice… {len(progress.ready)}/{len(progress.components)} components ready"
                screen.app.invalidate()

        progress = WarmupProgress(components, on_update=report)
        if buddy and state["talk"]:
            buddy.warming(progress)
        if state["mic"] is None:
            from vision.audio import Microphone
            from vision.stt import Transcriber

            state["mic"], state["stt"] = Microphone(cfg.listen), Transcriber(cfg.listen)
        mic, stt = state["mic"], state["stt"]
        listener_error = warm_voice(
            progress, stt.warm_up, lambda: _load_voice(speaker(), cfg.voice),
            small_listener if "listener" in components else None,
        )
        if listener_error is not None:
            state["cut_in"] = False
            note(f"[yellow]cutting in by voice is off: {listener_error}[/yellow]")
        return mic, stt

    def small_listener():
        """The tiny CPU listener that spots the name: shared by the wake word and cutting into a reply.
        Made (and its model loaded, unless only voice activity is needed) on first use."""
        from vision.wake import WakeListener

        with state["listener_lock"]:
            if state["wake_listener"] is None:
                state["wake_listener"] = WakeListener(cfg.listen, cfg.wake)
            if state["wake"] or cfg.listen.barge_in == "wake":
                state["wake_listener"].warm_up()
            return state["wake_listener"]

    def cut_in():
        """Start listening for a cut-in while the reply about to be spoken plays; None when that is off."""
        if not state["cut_in"]:
            return None
        from vision.wake import BargeIn

        try:
            return BargeIn(small_listener(), cfg.listen.barge_in, on_cut=cancel_reply).start()
        except Exception as e:  # noqa: BLE001  the model or the mic failed: say so once, keep talking
            state["cut_in"] = False
            note(f"[yellow]cutting in by voice is off: {e}[/yellow]")
            return None

    def cut_in_done(barge) -> tuple | None:
        """The reply is over: let go of the mic; what was said to cut in, if anything, is the next turn."""
        if barge is None:
            return None
        hit = barge.stop()
        if barge.error is not None:
            state["cut_in"] = False
            note(f"[yellow]cutting in by voice is off: {barge.error}[/yellow]")
        return hit

    def set_hearing(what: str):
        state["hearing"] = what
        screen.placeholder = f"{what}  speak, or type a message  (Esc ends listening)" if what else PLACEHOLDER
        screen.app.invalidate()

    def voice_loop(once: bool, woken=None, cut: bool = False):
        """Listen → transcribe → reply, until told to stop (/talk, Esc, “goodbye”) or, for /listen, once.

        `woken` is the (audio, command) the wake word heard: that utterance is the first turn (if it was
        only the name, a chime asks for the request), and a quiet follow-up window ends the conversation.
        `cut` says it was heard while cutting into a spoken reply, so it may open with the tail of that
        reply and the name is looked for anywhere in it.
        """
        from vision.config import resolve_device
        from vision.stt import LiveTranscript
        from vision.tts import chime
        from vision.wake import match_wake

        import queue

        typed: queue.Queue[str] = state["typed"]

        def settled():
            """Wait until nothing is being replied: the mic only opens once a reply is fully shown and spoken."""
            while screen.busy and state["talk"]:
                time.sleep(0.02)

        try:
            settled()
            if not state["talk"]:
                return
            screen.busy, screen.busy_label = True, "warming up ears and voice…"
            try:
                mic, stt = ears()
            finally:
                screen.busy, screen.busy_label = False, ""
                if buddy:
                    buddy.rest()
            if not state["talk"]:  # /talk was switched off while the models were warming up
                return
            if not once:
                if woken is None:
                    note(
                        f"voice conversation on · ears {stt.device} · voice {speaker().voice} — just speak; pause to send. "
                        "Type to answer instead. Esc, /talk or “goodbye” ends it"
                    )
            out_dev = resolve_device(cfg.voice.output_device, "output")
            # The wake utterance goes through the real ears too: the tiny model only had to spot the name.
            # `expect_name` says where to look for it in the transcript ("start" / "anywhere" / "").
            pending = woken[0] if woken else None
            expect_name = ("anywhere" if cut else "start") if woken else ""
            follow_up = cfg.wake.follow_up_s if woken else None

            def next_typed() -> str | None:
                try:
                    return typed.get_nowait()
                except queue.Empty:
                    return None

            preview: list = [None]  # the live `you ›` line of the utterance being heard, until the final text lands

            def drop_preview():
                if preview[0] is not None:
                    screen.remove_entry(preview[0])
                    preview[0] = None

            while state["talk"]:
                drop_preview()
                from vision.timing import VoiceTiming
                timing = VoiceTiming() if cfg.conversation.timing else None
                heard = next_typed()  # a message typed while the mic was open (or between turns) takes its turn
                from_voice = heard is None
                if heard is None and pending is not None:
                    audio, pending = pending, None
                elif heard is None:
                    cancel = threading.Event()
                    state["cancel"] = cancel
                    set_hearing("listening…")
                    if cfg.listen.chime:
                        chime("listen", out_dev)
                    # The words form on screen while you talk: the mic hands out the audio so far and
                    # LiveTranscript keeps re-reading it into a `you ›` line that the final text replaces.
                    # Re-transcribing a growing utterance on the CPU can take longer than the audio
                    # itself and used to make /talk appear frozen between listening and replying.
                    live = (
                        LiveTranscript(stt, lambda t: screen.update_entry(preview[0], hearing_grid(t)))
                        if cfg.listen.live_ms and stt.can_preview_live else None
                    )

                    def speech_started():
                        set_hearing("hearing you…")
                        if buddy:
                            buddy.listening()
                        if live is not None:
                            preview[0] = screen.add(hearing_grid(""), gap_before=True)

                    try:
                        if buddy:
                            buddy.listening()
                        audio = mic.record_utterance(
                            on_speech_start=speech_started, cancel=cancel, start_timeout_s=follow_up,
                            on_audio=live.feed if live is not None else None, every_ms=cfg.listen.live_ms,
                            on_level=buddy.hear if buddy else None,
                            **({"timing": timing} if timing else {}),
                        )
                    finally:
                        if timing:
                            timing.event("preview_stop_start")
                        preview_stopped = live is None or live.stop()
                        if timing:
                            timing.event("preview_stop_end")
                        set_hearing("")
                        if buddy:
                            buddy.listening(False)
                    if not preview_stopped:
                        raise RuntimeError("live transcription stopped responding; voice conversation reset")
                    heard = next_typed()
                    from_voice = heard is None
                    if heard is not None or not state["talk"]:
                        drop_preview()  # a typed message, or Esc, took over: the half-heard line goes
                if heard is None:
                    if not state["talk"]:
                        break
                    if audio is None or audio.size == 0:
                        if once:
                            note("didn't catch that")
                            break
                        if woken:
                            break  # nothing more said in the follow-up window: back to sleep
                        continue
                    screen.busy, screen.busy_label = True, "transcribing…"
                    if buddy:
                        buddy.chore("transcribing")
                    try:
                        if timing:
                            timing.event("transcription_start")
                        heard = stt.transcribe(audio).strip()
                        if timing:
                            timing.event("transcription_end")
                    finally:
                        screen.busy, screen.busy_label = False, ""
                        if buddy:
                            buddy.rest()
                    if expect_name:
                        command = match_wake(heard, cfg.wake.names, anywhere=expect_name == "anywhere") if heard else None
                        expect_name = ""
                        if command is not None:
                            heard = command
                        if not heard:
                            drop_preview()
                            continue  # only the name: the chime says "go ahead" and the mic opens
                    if not heard:
                        drop_preview()
                        note("didn't catch that")
                        if once:
                            break
                        continue
                    if preview[0] is not None:
                        screen.update_entry(preview[0], user_grid(heard))
                        preview[0] = None
                    else:
                        screen.add(user_grid(heard), gap_before=True)
                    screen.remember_user(heard)
                if not once and heard.lower().strip(" .!?,") in _EXIT_PHRASES:
                    from vision.tts import StreamingSpeaker

                    state["ss"] = ss = StreamingSpeaker(speaker())
                    try:
                        ss.feed("Goodbye.")
                        ss.finish()
                    finally:
                        state["ss"] = None
                    break
                barge = cut_in() if not once else None  # the mic is free now: hear a cut-in while it speaks
                run_turn(heard, from_voice=from_voice, timing=timing if from_voice else None)
                if once:
                    break
                settled()
                hit = cut_in_done(barge)
                if hit is not None:
                    pending, expect_name = hit[0], "anywhere" if cfg.listen.barge_in == "wake" else ""
        except (Exception, SystemExit) as e:  # noqa: BLE001  SystemExit: no such input device
            note(f"[red]listening failed: {e}[/red]")
            if buddy:
                buddy.fail()
        finally:
            conversation.model.close()
            state["talk"], state["once"], state["cancel"] = False, False, None
            set_hearing("")
            if buddy:
                buddy.rest()
            # Drop Whisper first, then the voice and its claim. When another Vision gets the claim it
            # also gets enough free VRAM to load both sides of a conversation. /speak keeps only the
            # voice because typed replies still use it.
            unload_errors = []
            for resource in (state["stt"], None if state["speak"] else state["speaker"]):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as e:  # noqa: BLE001
                        unload_errors.append(str(e))
            state["thread"] = None
            if unload_errors:
                note(f"[yellow]speech did not unload cleanly: {'; '.join(unload_errors)}[/yellow]")
            if not once and woken is None:
                note("voice conversation off")
            start_wake()  # back to waiting for the name, if the wake word is on

    def start_listening(once: bool, woken=None, cut: bool = False):
        if state["talk"]:
            note("already listening" if state["once"] else "voice conversation is already on (/talk again or Esc ends it)")
            return
        import queue

        stale = state["thread"]
        if stale is not None and stale.is_alive():
            stale.join(timeout=1)  # the last conversation is still letting go of the mic
            if stale.is_alive():  # stuck inside the audio library: a second stream on top would hang too
                note("[yellow]the mic from the last voice conversation hasn't closed yet; if this keeps up, restart Vision[/yellow]")
                return
        stop_wake(join=True)  # one mic stream at a time: the wake listener lets go before the ears open
        state["talk"], state["once"], state["typed"] = True, once, queue.Queue()
        state["thread"] = threading.Thread(target=voice_loop, args=(once, woken, cut), daemon=True)
        state["thread"].start()

    # ---- wake word: a tiny CPU model waits for the name whenever nothing else has the mic
    def wake_loop(cancel: threading.Event):
        """Wait for the name, then hand the utterance to a voice conversation (see voice_loop's `woken`)."""

        def hint(text: str):
            screen.placeholder = text
            screen.app.invalidate()

        try:
            if state["wake_listener"] is None:
                # Not `screen.busy`: typing must keep working while the small model loads (a few seconds).
                hint("loading the wake word…  (type a message meanwhile)")
                if buddy:
                    buddy.chore("loading")
                try:
                    small_listener()
                finally:
                    if buddy:
                        buddy.rest()
            if cancel.is_set():
                return
            hint(f"Say “{wake_name()}” to talk, or type a message…")
            # Drop audio while a reply is being written: the wake word is for an idle chat. While one is
            # being spoken, either drop it too (the speakers must not wake it) or, with barge_in on, keep
            # listening over it for the name / any voice, and cut the reply short when it comes.
            cut = {"yes": False}

            def speaking() -> bool:
                return state["ss"] is not None

            def spotted():
                if speaking():
                    cut["yes"] = True
                    cancel_reply()

            if state["cut_in"]:
                hit = state["wake_listener"].wait(
                    cancel,
                    paused=lambda: screen.busy and not speaking(),
                    replying=speaking,
                    on_spotted=spotted,
                    speech_cuts_in=cfg.listen.barge_in == "speech",
                )
            else:
                hit = state["wake_listener"].wait(cancel, paused=lambda: screen.busy or speaking())
            if hit is not None:
                hint(PLACEHOLDER)
                start_listening(once=False, woken=hit, cut=cut["yes"])  # no chime when a request came with the name
        except (Exception, SystemExit) as e:  # SystemExit: no such input device
            state["wake"] = False
            note(f"[red]wake word off: {e}[/red]")
            if buddy:
                buddy.fail()
        finally:
            if state["wake_thread"] is threading.current_thread():
                state["wake_thread"], state["wake_cancel"] = None, None
            if not state["talk"]:
                hint(PLACEHOLDER)

    def wake_name() -> str:
        return cfg.wake.names[0].capitalize()

    def wake_note():
        name = wake_name()
        note(f"wake word on · say “{name}” and I'll listen (“{name}, …” runs straight away) · /wake off ends it")

    def start_wake():
        """Listen for the name on a worker thread, unless a voice conversation has the mic (it resumes after)."""
        if not state["wake"] or state["quitting"] or state["talk"]:
            return
        old = state["wake_thread"]
        if old is not None and old.is_alive() and old is not threading.current_thread():
            old.join(timeout=1)  # the previous listener is on its way out after a hand-off
            if old.is_alive():
                return  # no: it is still waiting for the name
        state["wake_cancel"] = threading.Event()
        state["wake_thread"] = threading.Thread(target=wake_loop, args=(state["wake_cancel"],), daemon=True)
        state["wake_thread"].start()

    def stop_wake(join: bool = False):
        """Tell the wake listener to let go of the mic; `join` waits until it has (before opening it again)."""
        cancel, thread = state["wake_cancel"], state["wake_thread"]
        if cancel is not None:
            cancel.set()
        if join and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)

    def stop_listening() -> bool:
        """Ask the voice loop to stop; returns True if it was running. A reply in flight finishes first."""
        if not state["talk"]:
            return False
        state["talk"] = False
        if state["cancel"]:
            state["cancel"].set()
        # Clear the "listening…" prompt and face right away rather than when the loop winds down (it
        # does so again itself): if the mic ever hangs, the screen must not sit on listening.
        set_hearing("")
        if buddy:
            buddy.rest()
        return True

    # ---- slash commands
    def command(cmd: str, arg: str):
        nonlocal brain
        if cmd in ("quit", "exit", "q"):
            screen.exit(("quit",))
        elif cmd == "help":
            note(HELP_TEXT)
        elif cmd == "clear":
            with screen._lock:
                screen.entries.clear()
            paint_open()
        elif cmd == "model":
            choice, full = _split_model_arg(arg)
            if choice.lower() == "default":
                saved_model, saved_effort = _saved_defaults()
                switch(saved_model, effort=saved_effort, full=full, prefix="→ ")
            elif choice:
                switch(choice, full=full)
            else:
                screen.open_picker("Choose a model for this session", MODEL_TABS, cfg.brain.model, switch)
        elif cmd == "effort":
            def set_effort(choice):
                if not supports_effort(cfg.brain.model, choice):
                    allowed = ", ".join(v for v, _, _ in effort_choices(cfg.brain.model)) or "none"
                    note(f"{model_label(cfg.brain.model)} supports effort: {allowed}")
                    return
                if joined():
                    switch(cfg.brain.model, effort=choice)
                    return
                cfg.brain.effort = choice
                note(f"effort → {cfg.brain.effort or 'off'}")
            if arg:
                set_effort(arg.lower())
            else:
                screen.open_picker("Choose an effort level for this session", effort_choices(cfg.brain.model), cfg.brain.effort, set_effort)
        elif cmd == "fast":
            if screen.busy:
                note("still replying — change fast mode once it has finished (Esc cancels)")
                return
            try:
                enabled = _fast_value(arg, cfg.brain.fast)
            except ValueError as e:
                note(f"[yellow]{e}[/yellow]")
                return
            screen.busy = True

            def go_fast():
                nonlocal brain
                try:
                    brain, detail = _set_fast(cfg, brain, enabled, voice_mode=False)
                    msg = f"fast mode {'on' if enabled else 'off'} · {_brain_summary(cfg, brain)}"
                    note(f"{msg}{'; ' + detail if detail else ''}")
                    screen.app.invalidate()
                except Exception as e:  # noqa: BLE001
                    note(f"[red]fast mode failed: {e}[/red]")
                finally:
                    screen.busy = False

            background(go_fast)
        elif cmd == "default" and arg.strip().lower() == "reset":
            switch(FACTORY_MODEL, effort=FACTORY_EFFORT, save=True)
        elif cmd == "default":
            # Two pickers in a row (model, then effort), then persist both to config.toml.
            saved_model, saved_effort = _saved_defaults()

            def pick_effort(model_choice):
                selected_effort, _ = coerce_effort(model_choice, saved_effort)
                screen.open_picker(
                    "Default effort level", effort_choices(model_choice), selected_effort,
                    lambda effort_choice: switch(model_choice, effort=effort_choice, save=True),
                )

            screen.open_picker("Default model", MODEL_TABS, saved_model, pick_effort)
        elif cmd == "usage":
            try:
                which, full = _split_usage_arg(arg)
            except Exception as e:  # unknown word, or that provider's CLI is not installed
                note(f"[red]{e}[/red]")
                return
            label = "all provider" if which == "all" else provider_label((which or brain.provider), cli=True)
            screen.busy, screen.busy_label = True, f"reading {label} usage…"

            def go():
                try:
                    screen.add(_usage_selection(cfg, brain, which, full=full))
                except Exception as e:  # an unavailable provider selected by name
                    note(f"[red]{e}[/red]")
                finally:
                    screen.busy = False
            background(go)
        elif cmd in ("version", "versions"):
            try:
                show_vision, providers = clis.parse_version_arg(arg)
            except ValueError as e:
                note(f"[red]{e}[/red]")
                return
            screen.busy, screen.busy_label = True, "reading versions…"

            def go():
                try:
                    screen.add(_versions_renderable(providers=providers, vision=show_vision))
                finally:
                    screen.busy = False
            background(go)
        elif cmd == "update":
            try:
                which = clis.parse_update_arg(arg)
            except ValueError as e:
                note(f"[red]{e}[/red]")
                return
            if screen.busy:
                note("still busy — run /update once the reply has finished (Esc cancels)")
                return
            names = ", ".join(provider_label(p, cli=True) for p in which)
            screen.busy, screen.busy_label = True, f"updating {names}…"

            def go():
                try:
                    results = clis.update_clis(which, on_start=lambda p: setattr(screen, "busy_label", f"updating {provider_label(p, cli=True)}…"))
                    note(_update_summary(results))
                finally:
                    screen.busy = False
            background(go)
        elif cmd == "memory":
            from vision.memory import MEMORY_FILE, facts

            lines = facts()
            note(f"{MEMORY_FILE}\n" + ("\n".join(lines) if lines else "empty so far: the brain saves facts here, or /remember <fact> adds one"))
        elif cmd == "remember":
            from vision.memory import remember

            try:
                note(f"remembered: {remember(arg)}")
            except ValueError as e:
                note(f"[red]{e}[/red]")
        elif cmd == "forget":
            from vision.memory import forget

            try:
                gone = forget(arg)
                note("forgot:\n" + "\n".join(gone) if gone else "nothing in memory matches that")
            except ValueError as e:
                note(f"[red]{e}[/red]")
        elif cmd == "cd":
            if not arg.strip():
                note(f"working in {short_path(brain.workdir)} · /cd <dir>, /cd ~, /cd .. or /cd - moves")
                return
            if joined():
                note("[yellow]this chat runs under vision serve; its directory is set there[/yellow]")
                return
            try:
                target = _cd_target(arg, brain.workdir, state["prev_dir"])
            except ValueError as e:
                note(f"[red]{e}[/red]")
                return
            if target == brain.workdir:
                note(f"already in {short_path(target)}")
                return
            state["prev_dir"] = brain.workdir
            # The live brain and the config both move: brains built later (/model, the voice worker)
            # read cfg.workdir, and every provider spawns its CLI in brain.workdir on each turn, so
            # the change takes effect from the next reply on, even in a resumed conversation.
            brain.workdir = target
            cfg.brain.workdir = target
            note(f"directory → {short_path(target)}" + (" (from the next reply on)" if screen.busy else ""))
            screen.app.invalidate()  # the status row shows the directory
        elif cmd == "new":
            if screen.busy:
                note("still replying — start a new conversation once it has finished (Esc cancels)")
                return
            detach()  # a joined phone chat: leave it (it carries on there), then start afresh here
            brain.new_session()
            conversation.new_session()
            screen.set_history([])
            note("new conversation")
        elif cmd in ("mode", "auto", "plan"):
            word = cmd if cmd != "mode" else arg.strip().lower()
            if word and word not in MODES:
                note(f"[yellow]unknown mode “{word}”: auto or plan[/yellow]")
                return
            cfg.brain.mode = word or ("plan" if cfg.brain.mode == "auto" else "auto")  # bare /mode or Shift-Tab: toggle
            screen.app.invalidate()  # the status-row badge is the only feedback; nothing goes into the transcript
        elif cmd in ("session", "sessions", "resume"):
            from vision.remote import is_remote_key, live_sessions, mark_live_rows, remote_from_key, remote_rows
            from vision.sessions import find_any_session, list_all_sessions, session_from_key, session_key, session_tabs

            listed = list_all_sessions()
            phone = phone_chats()
            live = live_sessions(phone)

            def resume(key: str):
                if screen.busy:
                    note("[yellow]wait for the current reply to finish (Esc cancels it), then /session again[/yellow]")
                    return
                if is_remote_key(key):
                    hit = remote_from_key(key, phone)
                    if hit is None:
                        note("[red]that chat is gone from the phone[/red]")
                    else:
                        attach(hit)
                    return
                info = session_from_key(key, listed) or find_any_session(key, prefer=brain.provider)
                if info is None:
                    note(f"[red]no conversation starts with “{key}”[/red]")
                    return
                if session_key(info) in live:  # open on the phone: join it rather than fork a second copy
                    attach(live[session_key(info)])
                    return
                detach()  # a joined phone chat: leave it before this brain takes another conversation
                msg = _apply_session(brain, info)
                if msg != "already in that conversation":
                    conversation.new_session()
                    load_arrow_history(info.provider, info.id)
                    show_history(info.provider, info.id)
                note(msg)

            word = arg.strip().lower()
            if word in ("id", "info"):
                where = f" · joined to phone chat {brain.chat_id}" if joined() else ""
                note(f"current conversation: {brain.session_id or 'none yet (starts with the first message)'}{where}")
            elif word == "new":
                command("new", "")
            else:
                if word:
                    hit = find_any_session(word, prefer=brain.provider)
                    remote = remote_from_key(word, phone) if hit is None else None
                    if hit:
                        resume(session_key(hit))
                    elif remote:
                        attach(remote)
                    else:
                        note(f"[red]no conversation starts with “{arg.strip()}”[/red]")
                elif not listed and not phone:
                    note("no earlier conversations found")
                else:
                    tabs = mark_live_rows(session_tabs(), phone)
                    if state["server"]:
                        tabs.insert(0, ("Phone", remote_rows(phone), "" if phone else "no chats open on the phone"))
                    if joined():
                        current, prefer = f"remote:{brain.chat_id}", "Phone"
                    else:
                        current = f"{brain.provider}:{brain.session_id}" if brain.session_id else ""
                        prefer = provider_label(brain.provider)
                    screen.open_picker("Resume a conversation", tabs, current, resume, prefer_tab=prefer)
        elif cmd == "speak":
            state["speak"] = state["speaker"] is None or not state["speak"]
            note(f"speech {'on' if state['speak'] else 'off'}")
            if state["speak"] and not speaker()._loaded:
                # Load the voice now, on a worker thread (as --speak does at start-up), rather than
                # in the middle of the first spoken reply.
                def warm():
                    def report(progress):
                        screen.busy_label = f"loading voice… {len(progress.ready)}/{len(progress.components)} components ready"
                        screen.app.invalidate()

                    progress = WarmupProgress(("voice",), on_update=report)
                    try:
                        progress.run("voice", lambda: _load_voice(speaker(), cfg.voice))
                        note(f"voice ready · {speaker().voice} on {speaker().device_note}")
                    except Exception as e:  # noqa: BLE001
                        note(f"[red]voice failed to load: {e}[/red]")

                background(warm)
        elif cmd == "voice":
            def go():
                try:
                    speaker().set_voice(arg or cfg.voice.voice)
                    note(f"voice → {speaker().voice}")
                    if speaker()._loaded and cfg.voice.filler:
                        speaker().prepare_fillers(cfg.voice.filler_phrases + cfg.voice.filler_later_phrases)
                except Exception as e:
                    note(f"[red]{e}[/red]")
            background(go)
        elif cmd == "say":
            background(lambda: speaker().say(arg))
        elif cmd == "talk":
            if not stop_listening():
                start_listening(once=False)
        elif cmd == "listen":
            start_listening(once=True)
        elif cmd == "mic":
            def choices():
                from vision.config import input_device_choices

                try:
                    return input_device_choices()
                except Exception as e:  # noqa: BLE001  no PortAudio, no PipeWire
                    note(f"[red]can't list microphones: {e}[/red]")
                    return []

            def use(spec: str):
                """Switch every mic to `spec` ("" = system default) and reopen whatever is listening on it."""
                from vision.config import resolve_device

                spec = "" if spec.strip().lower() in ("", "default", "system") else spec.strip()
                try:
                    resolve_device(spec, "input")
                except SystemExit as e:
                    note(f"[red]{e}[/red]")
                    return
                cfg.listen.input_device = spec
                save_input_device(spec)
                for mic in (state["mic"], state["wake_listener"].mic if state["wake_listener"] else None):
                    if mic is not None:
                        mic.reconnect()
                talking, waking = state["talk"], state["wake"]
                if talking:
                    stop_listening()
                    thread = state["thread"]
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=5)  # a reply in flight finishes first
                elif waking:
                    stop_wake(join=True)
                mic_note = "the system default" if not spec else f"“{spec}”"
                note(f"mic → {mic_note}" + (" · reopening it" if talking or waking else ""))
                if talking:
                    if state["thread"] is not None and state["thread"].is_alive():
                        note("[yellow]the old voice conversation is still busy: /talk when it's done[/yellow]")
                    else:
                        start_listening(once=False)
                elif waking:
                    start_wake()

            if arg.strip():
                background(use, arg)
            else:
                def opened():
                    rows = choices()
                    if rows:
                        screen.open_picker("Choose a microphone", rows, cfg.listen.input_device, lambda v: background(use, v))
                        screen.app.invalidate()

                background(opened)  # pw-dump can take a moment: not on the UI thread
        elif cmd == "wake":
            word = arg.strip().lower()
            if word and word not in ("on", "off"):
                note("[yellow]/wake on or /wake off[/yellow]")
                return
            on = word == "on" if word else not state["wake"]
            state["wake"] = on
            cfg.wake.enabled = on
            save_wake_enabled(on)
            if on:
                wake_note()
                if state["talk"]:
                    note("(it starts listening for the name once this voice conversation ends)")
                start_wake()
            else:
                stop_wake()
                note("wake word off")
            screen.app.invalidate()
        elif cmd in ROUTE_COMMANDS:
            from vision.routing import OverrideError, parse_command

            text = f"/{cmd} {arg}".strip()
            try:
                override = parse_command(text, cfg.router)
            except OverrideError as e:
                note(f"[red]{e}[/red]")
                return
            if override is None:
                note(f"[red]unknown command /{cmd}[/red]")
                return
            if override.kind == "cancel" or override.text:
                send(text)  # the front end parses it again and acts on it
            else:
                conversation.pending_override = override
                what = f"{override.agent} {override.effort}".strip() if override.kind == "delegate" else override.kind
                note(f"next message → {what} (or put the request on the same line: /{cmd} {arg + ' ' if arg else ''}<request>)")
        elif cmd == "voicemodel":
            from vision.config import save_config_value

            words = arg.split()
            save = "save" in [w.lower() for w in words[1:]]

            def set_conversation(choice):
                if screen.busy:
                    note("still replying — change the voice model once it has finished (Esc cancels)")
                    return
                try:
                    conversation.set_model(choice)
                except BrainError as e:
                    note(f"[yellow]{e}[/yellow]")
                    return
                if save:
                    save_config_value("conversation", "model", f'"{choice}"')
                    if cfg.conversation.model == choice:
                        save_config_value("conversation", "effort", f'"{cfg.conversation.effort}"')
                note(_voicemodel_note(cfg, choice, save))
                screen.app.invalidate()

            if words:
                set_conversation(words[0])
            else:
                screen.open_picker("Choose the voice for Codex and Grok chats", CONVERSATION_TABS, conversation.fallback, set_conversation)
        elif cmd == "router":
            from vision.config import ROUTER_MODES, save_config_value

            words = arg.lower().split()
            mode = words[0] if words else ""
            if mode and mode not in ROUTER_MODES:
                note(f"[yellow]unknown router mode “{mode}”: off, audit or on[/yellow]")
                return
            if mode:
                cfg.router.mode = mode
                if "save" in words[1:]:
                    save_config_value("router", "mode", f'"{mode}"')
            from vision.routing import agent_model

            agents = ", ".join(f"{n} → {model_label(agent_model(cfg.router, n)) or '?'}" for n in cfg.router.agents)
            note(f"router {cfg.router.mode}{' (saved)' if mode and 'save' in words[1:] else ''} · default {cfg.router.default_agent} "
                 f"{cfg.router.default_effort}, hard tasks {cfg.router.high_effort} · agents: {agents} · log ~/.local/state/vision/routing.jsonl")
            screen.app.invalidate()
        else:
            note(f"[red]unknown command /{cmd}[/red]")

    def send(text: str):
        if state["talk"]:
            screen.add(user_grid(text), gap_before=True)
            state["typed"].put(text)
            if state["cancel"]:
                state["cancel"].set()
            return
        enqueue_turn(text, shown=queue_or_add(text))

    def queue_or_add(text: str) -> bool:
        """A message behind a running reply waits in the queued strip over the input box (True: it
        joins the transcript when its turn starts); otherwise it is in the transcript straight away."""
        if state["turn_active"]:
            screen.app.invalidate()
            return True
        screen.add(user_grid(text), gap_before=True)
        return False

    def take_queued(only: str | None = None) -> list[str]:
        """Queued typed messages out of the turn queue (all, or the one reading `only`), their rows
        out of the transcript: they are about to go into the running turn instead."""
        with state["turn_lock"]:
            taken = state["turn_queue"].take_where(lambda it: not it.follow and (only is None or it.text == only), first_only=only is not None)
        screen.app.invalidate()
        return [item.text for item in taken]

    def steer_now(text: str, from_phone: bool = False):
        """Ctrl-X, or Send now on the phone: into the running turn rather than after it; Claude takes
        it at its next step. "" = everything queued. A brain that can't take one mid-reply (Codex,
        Grok, a local model) leaves it queued. `from_phone`: the text may already sit in the queue
        (sent earlier, now pushed) or not (sent with `now`); either way it must get through."""
        running = screen.busy and state["turn_active"]
        if not running:
            if text and (not from_phone or not state["turn_queue"].has_text(text)):
                if from_phone:
                    enqueue_turn(text, shown=queue_or_add(text))
                else:
                    send(text)
            return
        if text:
            texts = [text]
            if from_phone:
                take_queued(text)
        else:
            texts = take_queued()
            if not texts:
                screen.notice("nothing queued", 2)
                return
        driver, split = state["driver"], state["steered"]
        for t in texts:
            if joined():
                brain.steer(t)  # the server sends it in or queues it; its `steered` splits the reply here
            elif getattr(driver, "steer", None) and driver.steer(t):
                if split:
                    split(t)
            else:
                note("this model can't take a message mid-reply, so it's queued")
                enqueue_turn(t, shown=queue_or_add(t))

    def submit(text: str):
        text = text.strip()
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            command(cmd.lower(), arg.strip())
            return
        send(text)

    screen.on_submit = submit
    screen.on_steer = steer_now
    screen.turns = state["turn_queue"]  # the queued strip reads and edits the same queue
    screen.can_steer_fn = lambda: joined() or bool(getattr(state["driver"], "steer", None))
    screen.on_cancel = cancel_reply
    screen.on_toggle_mode = lambda: command("mode", "")
    screen.on_interrupt = stop_listening  # Esc / Ctrl-C while idle: leave the voice conversation
    if wake or cfg.wake.enabled:
        state["wake"] = True
        start_wake()

    # ---- the phone: `vision serve` lists this chat and relays what is typed there (link.py)
    def link_summary() -> dict:
        return {
            "pid": os.getpid(),
            "title": live_title.get(brain.provider, brain.session_id),
            "provider": brain.provider,
            "model": _model_label(brain),
            "model_id": cfg.brain.model,
            "effort": cfg.brain.effort or "",
            "voice_model": model_label(cfg.conversation.model) or cfg.conversation.model,
            "voice_model_id": cfg.conversation.model,
            "voice_effort": cfg.conversation.effort or "",
            "session_id": brain.session_id,
            "context": context_figure(brain),
            "cache": cache_figure(brain),
            "workdir": brain.workdir,
            "busy": screen.busy or state["turn_active"],
            "waiting": screen.form_open,
            "questions": screen.pending_questions,
        }

    def link_frame(frame: dict):
        kind = frame.get("type")
        if kind == "message":
            text = (frame.get("text") or "").strip()
            if not text:
                return
            if state["talk"]:
                screen.add(user_grid(text), gap_before=True)
                note("from the phone")
                state["typed"].put(text)
                if state["cancel"]:
                    state["cancel"].set()
            else:
                shown = queue_or_add(text)
                note("from the phone")
                enqueue_turn(text, speak_remote=bool(frame.get("speak")), voice=bool(frame.get("voice")), talk=bool(frame.get("talk")), shown=shown)
        elif kind == "unqueue":
            # Removed (or taken back to edit) on the phone: out of the queue here too.
            text = (frame.get("text") or "").strip()
            if text:
                take_queued(text)
        elif kind == "steer":
            text = (frame.get("text") or "").strip()
            if text:
                steer_now(text, from_phone=True)
        elif kind == "answer":
            screen.answer_questions(frame.get("answers"))
        elif kind == "cancel":
            cancel_reply()
        elif kind == "model":
            if frame.get("model"):
                switch(frame["model"], effort=frame.get("effort") or None)
        elif kind == "quit":
            # Closed on the phone: leave like /quit. The link thread is not the UI loop, so hop over.
            note("closed from the phone")
            loop = screen.app.loop
            if loop is not None:
                loop.call_soon_threadsafe(screen.exit, ("quit",))
            else:
                screen.exit(("quit",))

    def start_link():
        try:
            from vision.link import LinkHost

            state["link"] = LinkHost(link_summary, link_frame, log=lambda m: screen.add(notice_grid(m, "warn")))
            state["link"].start()
        except Exception as e:  # noqa: BLE001  (a read-only state dir: the phone just does not see this chat)
            state["link"] = None
            note(f"[yellow]not visible to the phone: {e}[/yellow]")
    start_link()
    for n in RETIRED_NOTES:  # a saved model its provider no longer lists (see _cfg)
        note(f"[yellow]{n}[/yellow]")

    def join_on_start():
        """`vision --join <chat>` (the phone's "Open on laptop"): attach once the screen is up, so
        the joined history paints into a live app. A miss just leaves a note and starts as usual."""
        from vision.remote import remote_from_key

        phone = phone_chats()
        hit = remote_from_key(join, phone)
        if hit is None:
            note(f"[red]no phone chat starts with “{join}”[/red]" if state["server"] else "[red]vision serve is not running, so there is no phone chat to join[/red]")
            return
        attach(hit)

    try:
        screen.run(pre_run=join_on_start if join else None)
    except (EOFError, KeyboardInterrupt):
        pass
    state["quitting"] = True
    if state["link"]:
        state["link"].stop()
    if joined():
        brain.close()
    stop_wake()
    stop_listening()
    cancel_reply()
    for t in (state["thread"], state["wake_thread"], state["turn_thread"]):
        if t:
            t.join(timeout=3)  # let the mic stream close before the interpreter tears PortAudio down
    console.print(f"{NAME} [dim]signing off.[/dim]")


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
        from vision.stt import LiveTranscript

        with console.status(f"[{ACCENT}]listening…[/{ACCENT}]", spinner="dots") as st:
            live = LiveTranscript(stt, lambda t: st.update(f"[{ACCENT}]hearing you…[/{ACCENT}] [italic]{escape(t)}[/italic]")) if cfg.listen.live_ms else None
            try:
                audio = mic.record_utterance(
                    on_speech_start=lambda: st.update(f"[{ACCENT}]hearing you…[/{ACCENT}]"), cancel=cancel,
                    on_audio=live.feed if live is not None else None, every_ms=cfg.listen.live_ms,
                )
            finally:
                if live is not None:
                    live.stop()
    if audio is None or audio.size == 0:
        return ""
    with console.status("[dim]transcribing…[/dim]", spinner="dots"):
        return stt.transcribe(audio)


class _Keyboard:
    """Background stdin reader so you can type to Vision while it is listening.

    Polls the raw fd with select() instead of blocking in sys.stdin so close() can stop the
    thread cleanly: a reader left blocked on stdin keeps stealing keystrokes from the chat
    screen once talk mode returns to it.
    """

    def __init__(self):
        import queue

        self.q: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        if sys.platform == "win32":
            return self._reader_windows()
        import select

        buf = b""
        try:
            fd = sys.stdin.fileno()
            while not self._stop.is_set():
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                chunk = os.read(fd, 4096)
                if not chunk:
                    self.q.put(None)  # EOF (Ctrl-D)
                    return
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    self.q.put(line.decode(errors="replace").rstrip("\r"))
        except Exception:
            self.q.put(None)

    def _reader_windows(self):
        """select() takes only sockets on Windows, so poll the console with msvcrt instead (which
        also stops cleanly). getwch() does not echo, hence the writes; Ctrl-Z or Ctrl-D is EOF."""
        import msvcrt

        if not sys.stdin.isatty():  # piped input: nothing else will want stdin, so just block
            try:
                for text in sys.stdin:
                    self.q.put(text.rstrip("\r\n"))
            except Exception:
                pass
            self.q.put(None)
            return
        line: list[str] = []
        try:
            while not self._stop.is_set():
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two codes; ignore both
                    msvcrt.getwch()
                elif ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    self.q.put("".join(line))
                    line.clear()
                elif ch in ("\x1a", "\x04"):
                    self.q.put(None)
                    return
                elif ch == "\b":
                    if line:
                        line.pop()
                        sys.stdout.write("\b \b")
                elif ch.isprintable():
                    line.append(ch)
                    sys.stdout.write(ch)
                sys.stdout.flush()
        except Exception:
            self.q.put(None)

    def close(self):
        """Stop reading stdin; must run before anything else (the chat screen) takes the terminal."""
        self._stop.set()
        self._thread.join(timeout=1)

    def poll(self, timeout: float = 0.1):
        """Returns a typed line, None on EOF, or raises queue.Empty if nothing yet."""
        return self.q.get(timeout=timeout)


def _next_input(cfg: Config, mic, stt, kb: _Keyboard, ptt: bool, timing=None) -> tuple[str, bool] | None:
    """Return (text, from_voice), preserving its source even in talk mode; None on EOF."""
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
            return line.strip(), False
        stop = threading.Event()
        console.print(f"[{ACCENT}]recording… press Enter to stop[/{ACCENT}]")
        threading.Thread(target=lambda: (kb.q.get(), stop.set()), daemon=True).start()
        audio = mic.record_until_enter(stop)
        if timing:
            timing.event("endpoint")
    else:
        if cfg.listen.chime:
            chime("listen", out_dev)
        console.print(f"[{ACCENT}]listening…[/{ACCENT}] [dim](or type a message and press Enter)[/dim]")
        cancel = threading.Event()
        result: dict = {}

        def capture():
            try:
                result["audio"] = mic.record_utterance(
                    on_speech_start=lambda: console.print(f"[{ACCENT}]hearing you…[/{ACCENT}]"), cancel=cancel,
                    **({"timing": timing} if timing else {})
                )
            except Exception as e:  # noqa: BLE001  e.g. MicStalled: say so instead of dying on the worker thread
                result["error"] = e

        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        typed = None
        while worker.is_alive():
            try:
                typed = kb.poll(0.1)
            except queue.Empty:
                continue
            cancel.set()
            worker.join(timeout=3)  # a mic stuck in PortAudio must not take the keyboard down with it
            break
        if typed is not None or (not worker.is_alive() and cancel.is_set()):
            if typed is None:
                return None  # EOF (Ctrl-D)
            if typed.strip():
                return typed.strip(), False
            return "", False  # empty Enter: just restart listening
        if "error" in result:
            console.print(f"[red]listening failed: {result['error']}[/red] [dim]· trying again in a moment[/dim]")
            time.sleep(2)
            mic.reconnect()  # the device may have come back under a new index (USB mic replugged, PipeWire restarted)
            return "", True
        audio = result.get("audio")

    if audio is None or audio.size == 0:
        return "", True
    with console.status("[dim]transcribing…[/dim]", spinner="dots"):
        if timing:
            timing.event("transcription_start")
        text = stt.transcribe(audio)
        if timing:
            timing.event("transcription_end")
        return text, True


def _talk_loop(cfg: Config, brain, speaker, ptt: bool, echo: bool):
    from vision.audio import Microphone
    from vision.stt import Transcriber

    mic = Microphone(cfg.listen)
    stt = Transcriber(cfg.listen)
    listener = None  # hears you cut into a reply (barge_in in [listen]); shares nothing with the ears but the device
    if cfg.listen.barge_in != "off":
        from vision.wake import WakeListener

        listener = WakeListener(cfg.listen, cfg.wake)
    with console.status("[dim]warming up ears and voice…[/dim]"):
        loads = [threading.Thread(target=_load_voice, args=(speaker, cfg.voice), daemon=True)]
        if listener is not None and cfg.listen.barge_in == "wake":
            loads.append(threading.Thread(target=listener.warm_up, daemon=True))
        for t in loads:
            t.start()
        stt.warm_up()
        for t in loads:
            t.join()
    cut_in = {
        "wake": f"say “{cfg.wake.names[0].capitalize()}” to cut a reply short",
        "speech": "just speak to cut a reply short",
    }.get(cfg.listen.barge_in, "")
    show_header(
        console,
        model=_active_summary(cfg, brain, True),  # spoken and typed turns both go to the conversation model
        directory=short_path(brain.workdir),
        voice=f"{speaker.voice} on {speaker.device_note}",
        extra=(
            f"listening · ears {stt.device}",
            "Push-to-talk: press Enter to start and stop recording." if ptt else "Hands-free: just speak; pause to send.",
            "Type a message and press Enter at any time · Ctrl-C " + (f"or {cut_in} · " if cut_in else "interrupts a reply · ")
            + "say or type “goodbye” to exit",
        ),
    )
    kb = _Keyboard()
    try:
        return _talk_turns(cfg, brain, speaker, mic, stt, kb, ptt, listener)
    finally:
        kb.close()


def _talk_turns(cfg: Config, brain, speaker, mic, stt, kb: _Keyboard, ptt: bool, listener=None):
    """Voice/typed turns until goodbye, /quit, EOF or Ctrl-C. Returns the (possibly switched) brain."""
    from vision.wake import BargeIn, match_wake
    from vision.conversation import VoiceConversation

    conversation = VoiceConversation(cfg, agent=brain)
    try:
        conversation.warm_up()
        mode = cfg.listen.barge_in if listener is not None else "off"
        pending = None  # what you said while cutting into the last reply: it is the next turn
        while True:
            from vision.timing import VoiceTiming
            timing = VoiceTiming() if cfg.conversation.timing else None
            try:
                if pending is not None:
                    from_voice = True
                    audio, pending = pending, None
                    with console.status("[dim]transcribing…[/dim]", spinner="dots"):
                        if timing:
                            timing.event("transcription_start")
                        heard = stt.transcribe(audio).strip()
                        if timing:
                            timing.event("transcription_end")
                    if mode == "wake" and heard:  # it may open with the tail of the reply: keep what follows the name
                        command = match_wake(heard, cfg.wake.names, anywhere=True)
                        heard = command if command is not None else heard
                    if not heard:
                        continue  # only the name: the chime says "go ahead" and the mic opens
                else:
                    incoming = _next_input(cfg, mic, stt, kb, ptt, **({"timing": timing} if timing else {}))
                    heard, from_voice = incoming if incoming is not None else (None, False)
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
                    try:
                        which, full = _split_usage_arg(arg)
                        _show_usage(cfg, brain, which, full)
                    except Exception as e:
                        console.print(f"[red]{e}[/red]")
                elif cmd == "model" and _split_model_arg(arg)[0].lower() == "default":
                    saved_model, saved_effort = _saved_defaults()
                    brain, detail = _switch_spoken(cfg, brain, saved_model, effort=saved_effort, full=_split_model_arg(arg)[1])
                    console.print(f"[dim]→ {_brain_summary(cfg, brain)}{'; ' + detail if detail else ''}[/dim]")
                elif cmd == "model":
                    choice, full = _split_model_arg(arg)
                    choice = choice or pick("Choose a model for this session", MODEL_TABS, current=cfg.brain.model)
                    if choice is not None:
                        brain, detail = _switch_spoken(cfg, brain, choice, full=full)
                        console.print(f"[dim]model → {_brain_summary(cfg, brain)}{'; ' + detail if detail else ''}[/dim]")
                elif cmd == "effort":
                    choice = arg.strip().lower() or pick(
                        "Choose an effort level for this session", effort_choices(cfg.brain.model), current=cfg.brain.effort
                    )
                    if choice is None:
                        pass
                    elif supports_effort(cfg.brain.model, choice):
                        cfg.brain.effort = choice
                        console.print(f"[dim]effort → {choice or 'off'}[/dim]")
                    else:
                        allowed = ", ".join(v for v, _, _ in effort_choices(cfg.brain.model)) or "none"
                        console.print(f"[dim]{model_label(cfg.brain.model)} supports effort: {allowed}[/dim]")
                elif cmd == "fast":
                    try:
                        enabled = _fast_value(arg, cfg.brain.fast)
                        brain, detail = _set_fast(cfg, brain, enabled, voice_mode=False)
                        msg = f"fast mode {'on' if enabled else 'off'} · {_brain_summary(cfg, brain)}"
                        console.print(f"[dim]{msg}{'; ' + detail if detail else ''}[/dim]")
                    except ValueError as e:
                        console.print(f"[dim]{e}[/dim]")
                elif cmd == "default" and arg.strip().lower() == "reset":
                    brain, detail = _switch_spoken(cfg, brain, FACTORY_MODEL, effort=FACTORY_EFFORT)
                    msg = _save_defaults(cfg, cfg.brain.model, cfg.brain.effort)
                    console.print(f"[dim]{msg}{'; ' + detail if detail else ''}[/dim]")
                elif cmd == "default":
                    saved_model, saved_effort = _saved_defaults()
                    m = pick("Default model", MODEL_TABS, current=saved_model)
                    selected_effort, _ = coerce_effort(m, saved_effort) if m else ("", "")
                    e = pick("Default effort level", effort_choices(m), current=selected_effort) if m else None
                    if m and e is not None:
                        brain, detail = _switch_spoken(cfg, brain, m, effort=e)
                        msg = _save_defaults(cfg, cfg.brain.model, cfg.brain.effort)
                        console.print(f"[dim]{msg}{'; ' + detail if detail else ''}[/dim]")
                elif cmd in ("session", "sessions", "resume"):
                    from vision.sessions import find_any_session, list_all_sessions, session_from_key, session_tabs

                    listed = list_all_sessions()
                    word = arg.strip()
                    if word:
                        info = find_any_session(word, prefer=brain.provider)
                    elif not listed:
                        info = None
                        console.print("[dim]no earlier conversations found[/dim]")
                    else:
                        key = pick(
                            "Resume a conversation", session_tabs(),
                            current=f"{brain.provider}:{brain.session_id}" if brain.session_id else "",
                            prefer_tab=provider_label(brain.provider),
                        )
                        info = session_from_key(key or "", listed) if key else None
                    if info:
                        msg = _apply_session(brain, info)
                        if msg != "already in that conversation":
                            conversation.new_session()
                        console.print(f"[dim]{msg}[/dim]")
                    elif word:
                        console.print(f"[dim]no conversation starts with “{word}”[/dim]")
                elif cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "voicemodel":
                    from vision.config import save_config_value

                    words = arg.split()
                    choice = words[0] if words else pick("Choose the voice for Codex and Grok chats", CONVERSATION_TABS, current=conversation.fallback)
                    if choice is not None:
                        try:
                            conversation.set_model(choice)
                        except BrainError as e:
                            console.print(f"[dim]{e}[/dim]")
                        else:
                            save = "save" in [w.lower() for w in words[1:]]
                            if save:
                                save_config_value("conversation", "model", f'"{choice}"')
                                if cfg.conversation.model == choice:
                                    save_config_value("conversation", "effort", f'"{cfg.conversation.effort}"')
                            console.print(f"[dim]{_voicemodel_note(cfg, choice, save)}[/dim]")
                elif cmd == "new":
                    brain.new_session()
                    conversation.new_session()
                else:
                    console.print("[dim]in talk mode: /model, /effort, /fast, /default, /voicemodel, /session, /usage, /quit[/dim]")
                continue
            show_user(console, heard)
            if heard.lower().strip(" .!?,") in _EXIT_PHRASES:
                try:
                    speaker.say("Goodbye.")
                except KeyboardInterrupt:
                    speaker.stop()
                break
            # The mic is free now: hear you cut into the reply. speaker.stop() ends playback and the streaming
            # speaker inside _run_turn drops what is left.
            driver = _turn_brain(brain, conversation, from_voice, talk=True)
            barge = BargeIn(listener, mode, on_cut=lambda b=driver: (speaker.stop(), b.cancel())).start() if mode != "off" else None
            _run_turn(driver, heard, speaker, markdown=not from_voice, voice_cfg=cfg.voice if from_voice else None,
                      **({"timing": timing} if from_voice and timing else {}))
            if barge is not None:
                if barge.cut.is_set():
                    console.print(f"[{ACCENT}]hearing you…[/{ACCENT}]")  # the rest of what you said, up to the next pause
                hit = barge.stop()
                if barge.error is not None:
                    console.print(f"[yellow]cutting in by voice is off: {barge.error}[/yellow]")
                    mode = "off"
                elif hit is not None:
                    pending = hit[0]
        return brain

    finally:
        conversation.model.close()


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
    brain = _brain(cfg, voice_mode=False, cont=cont, new=False)
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
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice preset or name."),
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
        p = sp.save(msg, out)
        console.print(f"[dim]wrote[/dim] {p}")
        return
    try:
        sp.say(msg)
    except KeyboardInterrupt:
        sp.stop()


@app.command()
def weather(
    place: list[str] = typer.Argument(None, help="A place name. Omit for the home location in the config."),
    tomorrow: bool = typer.Option(False, "--tomorrow", help="Add tomorrow to the report."),
    week: bool = typer.Option(False, "--week", help="The whole outlook: tomorrow and the days after."),
    raw: bool = typer.Option(False, "--raw", help="Print the WeatherKit JSON instead of the report."),
):
    """Live weather from Apple WeatherKit (the report the voice hears; also for the typed brain).

    Covers now, the next twelve hours and today unless --tomorrow or --week asks for more."""
    from vision.weather import WeatherError, WeatherKit

    cfg = load_config()
    wk = WeatherKit(cfg.weather)
    name = " ".join(place).strip() if place else None
    try:
        if raw:
            spot = wk.geocode(name) if name else wk.default_place()
            print(json.dumps(wk.raw(spot), indent=1))
        else:
            print(wk.report(name, scope="week" if week else "tomorrow" if tomorrow else "today"))
    except WeatherError as e:
        raise SystemExit(f"weather: {e}")


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
    preview: bool = typer.Option(False, "--preview", "-p", help="Speak a sample line in each voice."),
    text: str = typer.Option("Good afternoon. Vision online, all systems nominal.", "--text", help="Sample line for --preview."),
):
    """List the voices of the configured engine (`[voice] engine` in the config)."""
    cfg = load_config()
    choices = voice_choices(cfg.voice)
    if cfg.voice.engine == "orpheus":
        t = Table(title="Orpheus voices", show_header=True, header_style=ACCENT)
        t.add_column("voice"), t.add_column("note")
        for n, note in choices:
            t.add_row(n, note or ("female" if n in ("tara", "leah", "jess", "mia", "zoe") else "male"))
        console.print(t)
        console.print("[dim]Inline sounds the model can act: <laugh> <chuckle> <sigh> <gasp> <groan> <yawn> <cough> <sniffle>[/dim]")
    else:
        t = Table(title=f"Qwen3-TTS voices in {VOICES_DIR}", show_header=True, header_style=ACCENT)
        t.add_column("voice"), t.add_column("kind"), t.add_column("description")
        for n, note in choices:
            design = voice_dir(n) / "design.txt"
            desc = design.read_text(encoding="utf-8").strip() if design.is_file() else ""
            t.add_row(f"[bold]{n}[/bold]" if n == cfg.voice.voice else n, note, desc[:90] + ("…" if len(desc) > 90 else ""))
        console.print(t)
        console.print("[dim]Make more: `vision voice design <name> \"<description>\"` or `vision voice add <name> --from clip.wav`[/dim]")
    playable = [n for n, note in choices if not note.startswith("built-in design, not made")]
    if preview and playable:
        sp = _speaker(cfg)
        for n in playable:
            console.print(f"  ▶ {n}")
            try:
                sp.set_voice(n)
                sp.say(text)
            except KeyboardInterrupt:
                sp.stop()
                break


voice_app = typer.Typer(help="Make and manage voices for the Qwen3-TTS engine.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")


def _save_voice(name: str, audio, transcript: str | None, design: str | None) -> Path:
    import soundfile as sf

    from vision.tts import SAMPLE_RATE

    d = voice_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "ref.wav"), audio, SAMPLE_RATE)
    if transcript:
        (d / "ref.txt").write_text(transcript.strip() + "\n", encoding="utf-8")
    elif (d / "ref.txt").exists():
        (d / "ref.txt").unlink()
    if design:
        (d / "design.txt").write_text(design.strip() + "\n", encoding="utf-8")
    elif (d / "design.txt").exists():
        (d / "design.txt").unlink()
    return d


def _check_voice_name(name: str) -> str:
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name):
        raise typer.BadParameter("voice names are lowercase letters, digits, - and _")
    return name


@voice_app.command("design")
def voice_design(
    name: str = typer.Argument(..., help="Name for the new voice (e.g. jarvis)."),
    description: Optional[str] = typer.Argument(None, help="How the voice should sound. Omit for a built-in design (jarvis, friday)."),
    text: Optional[str] = typer.Option(None, "--text", help="The line the model speaks to create the reference clip (8-12 s of speech is ideal)."),
    takes: int = typer.Option(3, "--takes", "-n", min=1, max=9, help="How many variations to generate and audition."),
    keep: Optional[int] = typer.Option(None, "--keep", "-k", help="Keep this take without listening (1-based)."),
    use: bool = typer.Option(False, "--use", help="Also make it the default voice in the config."),
):
    """Invent a voice from a text description (Qwen3-TTS VoiceDesign), audition takes and save the best one."""
    from vision.tts import SAMPLE_RATE, design_voice, qwen_present

    name = _check_voice_name(name)
    if description is None:
        if name not in VOICE_DESIGNS:
            raise typer.BadParameter(f"no built-in design for {name!r}; give a description (built-ins: {', '.join(VOICE_DESIGNS)})")
        description, default_text = VOICE_DESIGNS[name]
    else:
        default_text = VOICE_DESIGNS["jarvis"][1]
    text = text or default_text
    cfg = load_config()
    if not qwen_present(QWEN_TTS_DESIGN):
        _fetch_hf(QWEN_TTS_DESIGN)
    console.print(f"[bold]{name}[/bold]: {description}")
    console.print(f"[dim]says: {text}[/dim]")
    clips = []
    with console.status(f"[dim]designing the voice ({takes} take{'s' if takes > 1 else ''})…[/dim]") as st:
        for i, audio in enumerate(design_voice(description, text, cfg.voice.language, cfg.voice.device, takes), 1):
            clips.append(audio)
            st.update(f"[dim]designing the voice (take {i}/{takes} done)…[/dim]")
    if keep is None and takes == 1:
        keep = 1
    if keep is None:
        import sounddevice as sd

        from vision.config import resolve_device

        out = resolve_device(cfg.voice.output_device, "output")
        console.print("[dim]Playing each take. Answer with the number to keep, r to replay, q to give up.[/dim]")
        while True:
            for i, audio in enumerate(clips, 1):
                console.print(f"  ▶ take {i}  ({len(audio) / SAMPLE_RATE:.1f} s)")
                try:
                    with out.opening():
                        sd.play(audio, SAMPLE_RATE, device=out.index)
                    sd.wait()
                except KeyboardInterrupt:
                    sd.stop()
                    break
            ans = typer.prompt("keep", default="1").strip().lower()
            if ans == "r":
                continue
            if ans == "q":
                raise typer.Exit(1)
            if ans.isdigit() and 1 <= int(ans) <= len(clips):
                keep = int(ans)
                break
            console.print("[red]number, r or q[/red]")
    if not 1 <= keep <= len(clips):
        raise typer.BadParameter(f"--keep must be 1..{len(clips)}")
    d = _save_voice(name, clips[keep - 1], text, description)
    console.print(f"[green]✓[/green] voice [bold]{name}[/bold] saved in {d}")
    if use or cfg.voice.voice == name:
        save_voice_default(name)
        console.print(f"[green]✓[/green] default voice → {name}")
    else:
        console.print(f"[dim]use it: `vision voice use {name}` or `/voice {name}` in a chat[/dim]")


@voice_app.command("add")
def voice_add(
    name: str = typer.Argument(..., help="Name for the voice."),
    clip: Path = typer.Option(..., "--from", "-f", exists=True, dir_okay=False, help="A clean 5-15 s recording of the voice (WAV/FLAC/MP3)."),
    text: Optional[str] = typer.Option(None, "--text", help="Exact transcript of the clip. Omit to transcribe it with Whisper."),
    use: bool = typer.Option(False, "--use", help="Also make it the default voice in the config."),
):
    """Clone a voice from a recording. Only clone voices you have the right to use."""
    import numpy as np
    import soundfile as sf

    from vision.tts import SAMPLE_RATE

    name = _check_voice_name(name)
    audio, sr = sf.read(str(clip), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = np.arange(0, len(audio), sr / SAMPLE_RATE)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    secs = len(audio) / SAMPLE_RATE
    if secs < 2:
        raise typer.BadParameter("the clip is too short; give the model at least a few seconds of speech")
    if secs > 30:
        console.print("[yellow]![/yellow] clips over ~15 s do not clone better; using the first 20 s")
        audio = audio[: SAMPLE_RATE * 20]
    if text is None:
        from vision.stt import Transcriber

        cfg = load_config()
        with console.status("[dim]transcribing the clip with Whisper…[/dim]"):
            text = Transcriber(cfg.listen).transcribe(audio, SAMPLE_RATE).strip()
        console.print(f"[dim]heard: {text}[/dim]")
        if not text:
            console.print("[yellow]![/yellow] nothing transcribed; saving without a transcript (speaker-embedding only, a little less faithful)")
            text = None
    d = _save_voice(name, audio, text, None)
    console.print(f"[green]✓[/green] voice [bold]{name}[/bold] saved in {d}")
    if use:
        save_voice_default(name)
        console.print(f"[green]✓[/green] default voice → {name}")


@voice_app.command("use")
def voice_use(name: str = typer.Argument(..., help="A saved voice (qwen3) or an Orpheus voice/preset.")):
    """Make a voice the default in the config."""
    cfg = load_config()
    names = [n for n, _ in voice_choices(cfg.voice)]
    if name not in names:
        raise typer.BadParameter(f"unknown voice {name!r}; run `vision voices`")
    save_voice_default(name)
    console.print(f"[green]✓[/green] default voice → {name}")


@voice_app.command("rm")
def voice_rm(name: str = typer.Argument(..., help="A saved voice to delete.")):
    """Delete a saved voice."""
    import shutil as _shutil

    if name not in saved_voices():
        raise typer.BadParameter(f"no saved voice {name!r}")
    _shutil.rmtree(voice_dir(name))
    console.print(f"[green]✓[/green] removed {name}")


# ---------------------------------------------------------------- remote (iOS app)
@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (config: remote.host, default 0.0.0.0: the LAN, for the phone)."),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port (config: remote.port, default 8765)."),
    public_url: Optional[str] = typer.Option(None, "--public-url", help="The https:// address the phone uses (your tunnel)."),
    new_token: bool = typer.Option(False, "--new-token", help="Rotate the pairing token (old phones must re-pair)."),
    no_qr: bool = typer.Option(False, "--no-qr", help="Do not print the pairing QR code."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    effort: Optional[str] = typer.Option(None, "--effort"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v"),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the last Vision conversation."),
    new: bool = typer.Option(False, "--new", help="Open a fresh chat on startup."),
):
    """Serve Vision to the Vision Remote iOS app (chat + voice over HTTP/WebSocket). Pair by scanning the QR code."""
    from vision.server import TOKEN_FILE, pairing_info, qr_lines, serve as run_server

    cfg = _cfg(model, voice, effort)
    brain = _brain(cfg, voice_mode=False, cont=cont, new=new)
    info = pairing_info(cfg, host, port, public_url, regenerate_token=new_token)
    show_header(
        console,
        model=_brain_summary(cfg, brain),
        directory=short_path(brain.workdir),
        voice=cfg.voice.voice,
        resumed=bool(brain.session_id),
        extra=(
            f"listening on {info['local']}" + (f" · phone uses {info['url']}" if info["url"] != info["local"] else ""),
            f"token in {short_path(str(TOKEN_FILE))} · Ctrl-C stops",
        ),
    )
    if info["url"] == info["local"]:
        where = "on the laptop itself" if info["local"].startswith("http://127.") else "on this Wi-Fi"
        console.print(
            f"[yellow]no public URL:[/yellow] the phone can only reach this {where}. "
            "To reach it from anywhere, Tailscale Serve publishes it to your own devices only "
            "([dim]tailscale serve --bg %d[/dim]); Vision finds it and puts its https address in the QR. "
            "Tailscale Funnel or a Cloudflare Tunnel publish it to the whole internet instead; then set "
            "public_url under \\[remote] in config.toml or pass --public-url. See README → Remote." % info["port"]
        )
    if not no_qr:
        console.print()
        for line in qr_lines(info["payload"]):
            console.print("   " + line, markup=False, highlight=False)
        console.print("   [dim]scan with Vision Remote → Settings → Scan pairing code[/dim]")
    console.print(f"   [dim]or enter by hand:[/dim] {info['url']}  [dim]token[/dim] {info['token']}\n")

    def log(msg: str):
        console.print(Text("  " + msg, style="dim"))

    run_server(cfg, brain, info, log=log, follow_defaults=not (model or effort), open_initial=cont or new)


# ---------------------------------------------------------------- setup / doctor / config
def _download(url: str, dest: Path) -> None:
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    with console.status(f"[dim]downloading {dest.name}…[/dim]"):
        urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _install_llama() -> None:
    """Unpack the prebuilt llama.cpp CUDA build (binaries + bundled CUDA runtime) flat into LLAMA_DIR."""
    import tarfile
    import tempfile

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for url in LLAMA_URLS:
            archive = Path(tmp) / url.rsplit("/", 1)[1]
            _download(url, archive)
            if archive.suffix == ".zip":  # the Windows builds
                import zipfile

                with zipfile.ZipFile(archive) as z:
                    for info in z.infolist():
                        if not info.is_dir():
                            (LLAMA_DIR / Path(info.filename).name).write_bytes(z.read(info))
                continue
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = Path(member.name).name  # drop the versioned top-level folder
                        tar.extract(member, LLAMA_DIR)


def _fetch_hf(repo: str) -> None:
    from huggingface_hub import snapshot_download

    from vision.compat import check_hf_symlinks

    check_hf_symlinks(repo)
    with console.status(f"[dim]downloading {repo} (about 4.5 GB)…[/dim]"):
        snapshot_download(repo)


@app.command()
def setup(
    whisper: bool = typer.Option(True, help="Also pre-download the Whisper model."),
    orpheus: bool = typer.Option(False, help="Also fetch the Orpheus engine (needed only if [voice] engine = \"orpheus\")."),
):
    """Download voice and speech models (idempotent) and make the default voice if it does not exist yet."""
    from vision.tts import llama_server_bin, model_files, qwen_present

    cfg = load_config()
    if cfg.voice.engine == "qwen3" or not orpheus:
        if not qwen_present():
            _fetch_hf(QWEN_TTS_BASE)
        console.print(f"[green]✓[/green] Qwen3-TTS voice model ({QWEN_TTS_BASE})")
        if cfg.voice.voice in VOICE_DESIGNS and cfg.voice.voice not in saved_voices():
            console.print(f"[dim]no voice named {cfg.voice.voice!r} yet: designing it (first take, no audition)[/dim]")
            voice_design(cfg.voice.voice, None, None, takes=1, keep=1, use=False)
    if cfg.voice.engine == "orpheus" or orpheus:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        gguf, snac = model_files()
        if not gguf.exists():
            _download(ORPHEUS_MODEL_URL, gguf)  # 2.4 GB
        if not snac.exists():
            _download(SNAC_MODEL_URL, snac)
        console.print(f"[green]✓[/green] Orpheus voice model in {MODELS_DIR}")
        if not llama_server_bin().exists():
            _install_llama()
        console.print(f"[green]✓[/green] llama.cpp server in {LLAMA_DIR}")
    if whisper:
        from vision.stt import Transcriber

        stt = Transcriber(cfg.listen)
        with console.status("[dim]loading Whisper (downloads on first run)…[/dim]"):
            stt.load()
        console.print(f"[green]✓[/green] Whisper ready on {stt.device}")
        import dataclasses

        wake = Transcriber(dataclasses.replace(cfg.listen, whisper_model=cfg.wake.model, device="cpu"))
        with console.status("[dim]loading the wake-word model…[/dim]"):
            wake.load()
        console.print(f"[green]✓[/green] wake word (“{cfg.wake.names[0].capitalize()}”) ready on {wake.device}")
    console.print(f"[green]✓[/green] config: {CONFIG_PATH}")


@app.command()
def doctor(
    usage: bool = typer.Option(True, "--usage/--no-usage", help="Also show every provider's subscription usage."),
):
    """Check AI provider CLIs (versions, logins, usage), local models, GPU, and audio devices."""
    ok = "[green]✓[/green]"
    bad = "[red]✗[/red]"
    warn = "[yellow]![/yellow]"
    cfg = load_config()

    # provider CLIs: version, then a login check each
    infos = {i.provider: i for i in clis.cli_versions()}
    with console.status("[dim]checking for CLI updates…[/dim]"):
        checks = {p: clis.check_update(p) for p, i in infos.items() if i.ok}
    console.print(_versions_renderable(list(infos.values()), checks=checks))
    if any(c.available for c in checks.values()):
        console.print(f"{warn} newer release available: " + ", ".join(f"{c.label} {c.latest}" for c in checks.values() if c.available) + "  → run `vision update`")
    for i in infos.values():
        if not i.ok:
            console.print(f"{bad} {i.label} CLI: {i.error}\n   Install: {clis.install_hint(i.provider)}")

    claude_info = infos["claude"]
    if claude_info.ok:
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        r = subprocess.run(
            [claude_info.path, "-p", "Reply with the single word OK", "--output-format", "json", "--no-session-persistence", "--model", "haiku"],
            capture_output=True, text=True, env=env, cwd=str(WORKSPACE_DIR), timeout=120,
            encoding="utf-8",
        )
        if r.returncode == 0 and '"is_error":false' in r.stdout:
            console.print(f"{ok} Claude Code login works (headless round-trip succeeded)")
        else:
            console.print(f"{bad} Claude Code headless call failed: {(r.stderr or r.stdout).strip()[:300]}\n   Run `claude` once and log in.")

    codex_info = infos["codex"]
    if codex_info.ok:
        try:
            r = subprocess.run([codex_info.path, "login", "status"], capture_output=True, text=True, timeout=30, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired) as e:
            console.print(f"{bad} Codex login check failed: {e}")
        else:
            status = (r.stdout or r.stderr).strip()
            if r.returncode == 0:
                console.print(f"{ok} Codex login works ({status or 'authenticated'})")
            else:
                console.print(f"{bad} Codex is not logged in: {status}\n   Run `codex login` once.")

    grok_info = infos["grok"]
    if grok_info.ok:
        try:
            r = subprocess.run([grok_info.path, "models"], capture_output=True, text=True, timeout=30, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired) as e:
            console.print(f"{bad} Grok login check failed: {e}")
        else:
            if r.returncode == 0:
                console.print(f"{ok} Grok login works ({(r.stdout or '').strip().splitlines()[0] if r.stdout.strip() else 'authenticated'})")
            else:
                status = (r.stderr or r.stdout).strip()
                console.print(f"{bad} Grok is not logged in: {status[:300]}\n   Run `grok login` once.")

    # usage for every provider that is installed (one missing CLI is reported inline, not fatal)
    if usage and any(i.ok for i in infos.values()):
        with console.status("[dim]reading all provider usage…[/dim]"):
            report = _usage_selection(cfg, None, "all")
        console.print(report)

    # models, GPU and audio need the voice extra; a text-only install stops here, and that is fine
    import importlib.util

    missing = [m for m in ("numpy", "sounddevice", "soundfile", "faster_whisper") if importlib.util.find_spec(m) is None]
    if missing:
        console.print(f"{warn} voice not installed (optional; {', '.join(missing)} missing). For speech in and out:  pip install -e '.\\[voice]'")  # \\[: not rich markup
        console.print(f"[dim]config: {CONFIG_PATH}[/dim]")
        return

    # models
    from vision.tts import models_present

    have = models_present(cfg.voice)
    what = "Orpheus voice model + llama.cpp" if cfg.voice.engine == "orpheus" else f"Qwen3-TTS voice model ({QWEN_TTS_BASE})"
    console.print(f"{ok if have else bad} {what}" + ("" if have else "  → run `vision setup`"))
    if cfg.voice.engine == "qwen3":
        names = saved_voices()
        console.print(f"{ok if cfg.voice.voice in names else bad} voice {cfg.voice.voice!r} in {VOICES_DIR} (have: {', '.join(names) or 'none'})" + ("" if cfg.voice.voice in names else f"  → vision voice design {cfg.voice.voice}"))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8")
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
        if (d["max_input_channels"] or d["max_output_channels"]) and "JACK" not in sd.query_hostapis(d["hostapi"])["name"]:
            t.add_row(str(i), d["name"], str(d["max_input_channels"]), str(d["max_output_channels"]))
    console.print(t)
    from vision.config import pipewire_nodes

    nodes = [(kind, desc) for kind in ("input", "output") for _, desc in pipewire_nodes(kind)]
    if nodes:
        t = Table(title="PipeWire devices (use any part of the name as input_device / output_device)", header_style=ACCENT)
        t.add_column("kind"), t.add_column("name")
        for kind, desc in nodes:
            t.add_row(kind, desc)
        console.print(t)
    try:
        from vision.tts import Speaker

        sp = Speaker(cfg.voice)
        with console.status("[dim]loading the voice…[/dim]"):
            sp._load()
        console.print(f"{ok} voice '{cfg.voice.voice}' loads on {sp.device_note} ({sp.voice})" + (" — too slow for conversation on the CPU" if sp.device == "cpu" else ""))
    except Exception as e:
        console.print(f"{bad} voice failed: {e}")
    console.print(f"[dim]config: {CONFIG_PATH}[/dim]")


@app.command()
def default(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model alias or full name (e.g. opus)."),
    effort: Optional[str] = typer.Option(None, "--effort", "-e", help="Default reasoning effort (model-dependent)."),
    reset: bool = typer.Option(False, "--reset", help="Go back to Vision's factory default: Opus 5 · high."),
):
    """Choose and save Vision's default model and effort level (arrow-key pickers unless given as options)."""
    cfg = load_config()
    if reset:
        console.print(_save_defaults(cfg, FACTORY_MODEL, FACTORY_EFFORT))
        console.print(f"[dim]{CONFIG_PATH}[/dim]")
        return
    m = model or pick("Default model", MODEL_TABS, current=cfg.brain.model)
    if not m:
        raise typer.Exit(1)
    current_effort, _ = coerce_effort(m, cfg.brain.effort)
    e = effort.lower() if effort is not None else pick("Default effort level", effort_choices(m), current=current_effort)
    if e is None:
        raise typer.Exit(1)
    if not supports_effort(m, e):
        allowed = ", ".join(v for v, _, _ in effort_choices(m)) or "none"
        raise typer.BadParameter(f"{model_label(m)} supports effort: {allowed}")
    console.print(_save_defaults(cfg, m, e))
    console.print(f"[dim]{CONFIG_PATH}[/dim]")


@app.command()
def config(edit: bool = typer.Option(False, "--edit", "-e", help="Open the config in $EDITOR.")):
    """Show (or edit) Vision's configuration."""
    load_config()
    if edit:
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or ("notepad" if sys.platform == "win32" else "nano")
        subprocess.call([editor, str(CONFIG_PATH)])
        return
    console.print(f"[dim]{CONFIG_PATH}[/dim]")
    console.print(Text(CONFIG_PATH.read_text(encoding="utf-8")))


def main():
    from vision.proctitle import hide_cmdline

    hide_cmdline()  # the terminal tab reads `Vision`, not `Vision — /…/.venv/bin/python -m vision`
    code = 0
    try:
        app()
    except KeyboardInterrupt:
        console.print()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            console.print(str(e.code))
    # Skip the interpreter's teardown: with a voice model on the GPU and PortAudio streams open it can
    # take many seconds (or hang in Pa_Terminate on a stalled mic), and everything we keep is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()

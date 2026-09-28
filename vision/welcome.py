"""The guided setup: `vision provider setup`, and what `vision` runs by itself the first time it
starts with no brain ready (see first_run).

One question at a time in the plain console: which brain, then whatever that one needs. A CLI
provider (Claude Code, Codex, Grok) is installed on request (its install command, run here after a
yes) and logged in to by running its own login in the foreground; a server or API is asked for its
URL and checked; an ACP agent for its command. Each answer lands in config.toml at once, so a wizard
left halfway has still done something. It ends by saving the default model from what is ready.

Every question goes through an `IO` so tests script the answers (tests/test_welcome.py)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable

from vision import clis, providers
from vision.config import CONFIG_PATH, load_config, save_provider_table

SERVER, HOSTED, AGENT, DONE = "_server", "_hosted", "_agent", "_done"


class ConsoleIO:
    """The real console: rich for text, the arrow-key picker, typer's prompts, subprocess for the commands."""

    def __init__(self, console):
        self.console = console

    def say(self, text: str = "") -> None:
        self.console.print(text)

    def pick(self, title: str, options: list[tuple[str, str, str]], current: str = "") -> str | None:
        from vision.ui import pick

        return pick(title, options, current=current)

    def ask(self, question: str, default: str = "") -> str:
        import typer

        return typer.prompt(question, default=default, show_default=bool(default)).strip()

    def confirm(self, question: str, default: bool = True) -> bool:
        import typer

        return typer.confirm(question, default=default)

    def run(self, cmd: list[str] | str, shell: bool = False) -> int:
        """A command in the foreground with the terminal (installers, logins are interactive)."""
        try:
            return subprocess.run(cmd, shell=shell).returncode
        except (OSError, KeyboardInterrupt):
            return 1


class Wizard:
    def __init__(self, io, cfg=None):
        self.io = io
        self.cfg = cfg or load_config()
        self.added: list[str] = []

    # -- helpers
    def _ready(self, name: str) -> bool:
        providers.forget_ready()
        clis.forget_checks()
        return providers.ready(name, self.cfg)

    def _reload(self) -> None:
        self.cfg = load_config()
        providers.forget_ready()

    def _menu(self) -> list[tuple[str, str, str]]:
        rows = []
        for p in providers.REGISTRY.values():
            if not p.builtin or not p.cli:
                continue
            state = "ready" if self._ready(p.name) else "not installed yet"
            rows.append((p.name, p.label, f"{p.note.removeprefix('via ')} · {state}"))
        rows += [
            (SERVER, "A server on this machine", "Ollama, LM Studio, llama-server, vLLM… (free, private)"),
            (HOSTED, "A hosted API", "OpenRouter, or any OpenAI-compatible endpoint with a key"),
            (AGENT, "An agent that speaks ACP", "Gemini CLI and others: a command Vision drives"),
            (DONE, "Done", "finish setup"),
        ]
        return rows

    # -- steps
    def run(self) -> list[str]:
        """The whole flow; returns the providers that are ready at the end."""
        io = self.io
        io.say("[bold]Welcome to Vision.[/bold] It thinks with a brain you already have: a Claude, ChatGPT or Grok "
               "subscription through its own CLI, a model server on this machine, or an API.")
        io.say(f"[dim]Everything chosen here goes in {CONFIG_PATH}; /providers and /model change it later.[/dim]\n")
        while True:
            choice = io.pick("Which brain do you want to set up?", self._menu())
            if choice in (None, DONE):
                break
            if choice == SERVER:
                self.add_server()
            elif choice == HOSTED:
                self.add_hosted()
            elif choice == AGENT:
                self.add_agent()
            else:
                self.setup_cli(choice)
            io.say("")
        return self.finish()

    def setup_cli(self, name: str) -> bool:
        io, p = self.io, providers.REGISTRY[name]
        if self._ready(name):
            io.say(f"[green]✓[/green] {p.product} is installed.")
        else:
            io.say(f"{p.product} isn't installed yet.")
            if p.install_cmd and io.confirm(f"Run its installer now?  [dim]{p.install_cmd}[/dim]", default=True):
                io.say(f"[dim]$ {p.install_cmd}[/dim]")
                if io.run(p.install_cmd, shell=True) != 0:
                    io.say("[yellow]The installer did not finish cleanly.[/yellow]")
            else:
                where = p.install_url or p.install_hint
                io.say(f"Install it from {where}, then come back here.")
                io.confirm("Installed? (press Enter when it is)", default=True)
            if not self._ready(name):
                io.say(f"[yellow]Still can't find `{p.cli}` on PATH. Open a new terminal after installing, or add its folder to PATH; "
                       f"then `vision provider setup` again.[/yellow]")
                return False
            io.say(f"[green]✓[/green] {p.product} is installed.")
        if p.login_cmd and io.confirm(f"Log in to {p.product} now?  [dim]{' '.join(p.login_cmd)}"
                                      + (" (type /exit when it opens)" if p.login_cmd == ("claude",) else "") + "[/dim]", default=True):
            exe = shutil.which(p.login_cmd[0]) or p.login_cmd[0]
            io.run([exe, *p.login_cmd[1:]])
        io.say(f"[green]✓[/green] {p.label} models will show in /model.")
        self.added.append(name)
        return True

    def _name(self, default: str) -> str | None:
        import re

        for _ in range(3):
            name = self.io.ask("Short name for it (its models show as <name>/<model>)", default=default).lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) and not (name in providers.REGISTRY and providers.REGISTRY[name].source != "config"):
                return name
            self.io.say("[yellow]A plain lowercase word that isn't a built-in provider, please.[/yellow]")
        return None

    def _check_models(self, name: str) -> list[str]:
        """Register the table just written and ask the server for its models (kept for /model)."""
        from vision import models

        self._reload()
        try:
            ids = clis.local_model_ids(timeout=8, provider=name)
        except Exception as e:  # noqa: BLE001
            self.io.say(f"[yellow]The server didn't answer ({e}). It's saved anyway; start it and its models appear in /model.[/yellow]")
            return []
        models.set_local_models(ids, name)
        return ids

    def add_server(self) -> bool:
        io = self.io
        name = self._name("ollama")
        if not name:
            return False
        url = io.ask("Its OpenAI-compatible URL", default="http://localhost:11434/v1" if name == "ollama" else "http://localhost:8080/v1").rstrip("/")
        if not url.startswith(("http://", "https://")):
            io.say("[yellow]That's not an http(s) URL.[/yellow]")
            return False
        save_provider_table(name, {"base_url": url, "label": io.ask("Label in /model", default=name.capitalize())})
        ids = self._check_models(name)
        if ids:
            io.say(f"[green]✓[/green] {len(ids)} model{'s' if len(ids) != 1 else ''}: {', '.join(ids[:6])}{'…' if len(ids) > 6 else ''}")
        self.added.append(name)
        return True

    def add_hosted(self) -> bool:
        io = self.io
        name = self._name("openrouter")
        if not name:
            return False
        url = io.ask("Its OpenAI-compatible URL", default="https://openrouter.ai/api/v1" if name == "openrouter" else "").rstrip("/")
        if not url.startswith(("http://", "https://")):
            io.say("[yellow]That's not an http(s) URL.[/yellow]")
            return False
        key_env = io.ask("Environment variable holding its API key", default=f"{name.upper().replace('-', '_')}_API_KEY")
        if key_env and not os.environ.get(key_env):
            io.say(f"[yellow]{key_env} isn't set in this shell; export it (in your shell profile) before chatting.[/yellow]")
        ids = [m.strip() for m in io.ask("Model ids to offer, comma-separated (blank: ask the server)", default="").split(",") if m.strip()]
        save_provider_table(name, {"base_url": url, "label": io.ask("Label in /model", default=name.capitalize()), "api_key_env": key_env, "models": ids})
        found = self._check_models(name)
        if found:
            io.say(f"[green]✓[/green] {len(found)} model{'s' if len(found) != 1 else ''}: {', '.join(found[:6])}{'…' if len(found) > 6 else ''}")
        self.added.append(name)
        return True

    def add_agent(self) -> bool:
        io = self.io
        name = self._name("gemini")
        if not name:
            return False
        command = io.ask("The command that speaks ACP over stdio", default="gemini --experimental-acp" if name == "gemini" else "").split()
        if not command:
            return False
        if not shutil.which(command[0]):
            io.say(f"[yellow]`{command[0]}` isn't on PATH here; the provider shows as not set up until it is.[/yellow]")
        save_provider_table(name, {"type": "acp", "command": command, "label": io.ask("Label in /model", default=name.capitalize())})
        self._reload()
        io.say(f"[green]✓[/green] {name}/default in /model.")
        self.added.append(name)
        return True

    def finish(self) -> list[str]:
        io = self.io
        self._reload()
        ready = [p.name for p in providers.enabled(self.cfg) if providers.ready(p.name, self.cfg)]
        if not ready:
            io.say("[yellow]Nothing is ready yet.[/yellow] `vision provider setup` runs this again; `vision doctor` checks what's installed.")
            return ready
        from vision.models import coerce_effort, find, model_label, provider_default

        clis.refresh_models_soon(ready, max_age=0)  # their lists, for /model
        tabs = providers.model_tabs(self.cfg, setup_rows=False)
        rows = [(v, f"{label} · {tab}", desc) for tab, entries, _ in tabs for v, label, desc in entries]
        current = self.cfg.brain.model if providers.unavailable_reason(providers_of(self.cfg.brain.model), self.cfg) is None else provider_default(ready[0])
        chosen = io.pick("Which model should Vision start on?", rows, current=current) if len(rows) > 1 else (rows[0][0] if rows else current)
        if chosen:
            from vision.cli import _save_defaults

            m = find(chosen)
            effort = m.resting_effort() if m else coerce_effort(chosen, self.cfg.brain.effort)[0]  # a local model starts with thinking off
            io.say(_save_defaults(self.cfg, chosen, effort))
        io.say(f"\n[bold]All set.[/bold] `vision` to chat ({model_label(chosen or current) or chosen or current}), /model to switch, "
               "/providers to choose what's offered, `vision provider setup` to add more.")
        return ready


def providers_of(model: str) -> str:
    from vision.models import provider_for

    return provider_for(model)


def first_run(cfg, io=None, *, interactive: bool | None = None) -> bool:
    """Run the wizard when no enabled provider is ready and there is a person at the terminal.
    True when it ran (the caller reloads its config)."""
    if any(providers.ready(p.name, cfg) for p in providers.enabled(cfg)):
        return False
    if os.environ.get("VISION_NO_SETUP"):
        return False
    tty = sys.stdin.isatty() and sys.stdout.isatty() if interactive is None else interactive
    if not tty:
        return False
    if io is None:
        from vision.cli import console

        io = ConsoleIO(console)
    Wizard(io, cfg).run()
    return True

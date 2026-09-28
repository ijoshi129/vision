"""The providers Vision can think with, in one place: what each is called, whether it is set up on
this machine, how to set it up, and which brain class drives it.

`[providers] enabled` in config.toml says which ones /model offers (/providers picks them). One that
is enabled but not ready (its CLI is missing, its server does not answer) still shows in /model, as a
single row that says how to set it up, so a new install never fails on its first message.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass

from vision.config import PROVIDER_NAMES


@dataclass(frozen=True)
class Provider:
    name: str  # "claude", as in [providers] enabled and on the wire
    label: str  # "Claude", the /model tab
    note: str  # the tab's second line
    setup: str  # how to get it ready, shown when it is not
    brain: str  # "module:Class" of the driver

    def brain_class(self):
        module, cls = self.brain.split(":")
        return getattr(__import__(module, fromlist=[cls]), cls)


REGISTRY: dict[str, Provider] = {p.name: p for p in (
    Provider("claude", "Claude", "via Claude Code · Claude Subscription",
             "install Claude Code (curl -fsSL https://claude.ai/install.sh | bash), then run `claude` to log in",
             "vision.brain:Brain"),
    Provider("codex", "Codex", "via Codex CLI · ChatGPT Subscription",
             "install the Codex CLI (npm i -g @openai/codex), then run `codex login`", "vision.codex:CodexBrain"),
    Provider("grok", "Grok", "via Grok CLI · Grok Subscription",
             "install the Grok CLI (https://x.ai/cli), then run `grok login`", "vision.grok:GrokBrain"),
    Provider("local", "Local", "via your llama-server · free",
             "start llama-server and point [local] base_url in config.toml at it", "vision.local:LocalBrain"),
)}
assert tuple(REGISTRY) == PROVIDER_NAMES

SETUP_PREFIX = "setup:"  # a /model row value that means "tell me how to set this up", not a model
_READY_TTL = 30.0
_ready_cache: dict[tuple, tuple[float, bool]] = {}
_ready_lock = threading.Lock()


def get(name: str) -> Provider | None:
    return REGISTRY.get(name)


def enabled(cfg) -> list[Provider]:
    """The providers /model offers, in the order the config lists them."""
    names = getattr(getattr(cfg, "providers", None), "enabled", None)
    names = PROVIDER_NAMES if names is None else names
    return [REGISTRY[n] for n in names if n in REGISTRY]


def _server_answers(base_url: str) -> bool:
    """A TCP connect to the server's host and port, briefly: enough to tell "nothing there" from "up"."""
    parts = urllib.parse.urlsplit(base_url or "")
    if not parts.hostname:
        return False
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((parts.hostname, port), timeout=0.4):
            return True
    except OSError:
        return False


def ready(name: str, cfg=None) -> bool:
    """Whether this provider can take a turn here: its CLI is installed, or (local) its server answers.
    Cached for half a minute so /model opens instantly; a login problem still shows on the first turn."""
    key = (name, getattr(getattr(cfg, "local", None), "base_url", "") if name == "local" else "")
    now = time.monotonic()
    with _ready_lock:
        hit = _ready_cache.get(key)
        if hit and now - hit[0] < _READY_TTL:
            return hit[1]
    if name == "local":
        ok = _server_answers(key[1])
    elif name in REGISTRY:
        from vision import clis

        try:
            clis.find_cli(name)
            ok = True
        except Exception:  # noqa: BLE001  not installed, or a launcher Vision refuses (Windows .cmd)
            ok = False
    else:
        ok = False
    with _ready_lock:
        _ready_cache[key] = (now, ok)
    return ok


def forget_ready() -> None:
    """Drop the cached answers (after an install, a config change)."""
    with _ready_lock:
        _ready_cache.clear()


def model_tabs(cfg, *, setup_rows: bool = True) -> list[tuple[str, list, str]]:
    """/model's tabs: every enabled provider, its models when it is ready, else (setup_rows) one row that
    says how to set it up. Without setup_rows (the phone, which cannot install anything) a provider
    that is not ready is left out."""
    from vision.models import MODEL_TABS

    rows_by_label = {tab: (rows, note) for tab, rows, note in MODEL_TABS}
    tabs = []
    for p in enabled(cfg):
        rows, note = rows_by_label.get(p.label, ([], p.note))
        if ready(p.name, cfg) and rows:
            tabs.append((p.label, rows, note))
        elif setup_rows:
            tabs.append((p.label, [(SETUP_PREFIX + p.name, "Not set up", p.setup)], note))
    return tabs


def setup_note(value: str) -> str | None:
    """For a /model pick that is a setup row, what to tell the user; None for a real model."""
    if not value.startswith(SETUP_PREFIX):
        return None
    p = REGISTRY.get(value[len(SETUP_PREFIX):])
    return f"{p.label} isn't set up here yet: {p.setup}." if p else "That provider isn't set up here yet."


def first_ready_model(cfg) -> str | None:
    """The default model of the first enabled provider that is ready, for a start whose saved model can't run."""
    from vision.models import provider_default

    for p in enabled(cfg):
        if ready(p.name, cfg):
            model = provider_default(p.name)
            if model:
                return model
    return None


def unavailable_reason(provider: str, cfg) -> str | None:
    """Why a turn on this provider can't run here (switched off, not set up), or None when it can."""
    p = REGISTRY.get(provider)
    if p is None:
        return f"Vision has no provider called {provider!r}."
    if p not in enabled(cfg):
        return f"{p.label} is switched off in /providers."
    if not ready(provider, cfg):
        return f"{p.label} isn't set up here yet: {p.setup}."
    return None


def settle(cfg) -> str:
    """At the start of a chat: when the saved model's provider can't run here (switched off, not set up),
    move to the first ready provider's default for this run. Returns a note for the user ("" when the
    saved model stands); config.toml is left as it is."""
    from vision.models import coerce_effort, model_label, provider_for

    why = unavailable_reason(provider_for(cfg.brain.model), cfg)
    if not why:
        return ""
    other = first_ready_model(cfg)
    if not other:
        hints = "; ".join(f"{p.label}: {p.setup}" for p in enabled(cfg)) or "turn one on with /providers"
        return f"No provider is set up yet, so Vision can't answer. {hints}."
    cfg.brain.model = other
    cfg.brain.effort, _ = coerce_effort(other, cfg.brain.effort)
    return f"{why} Using {model_label(other)} for now (config.toml unchanged)."

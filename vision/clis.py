"""The provider CLIs Vision drives (Claude Code, Codex, Grok): where each one is, its version, and
how to update it. `vision --version`, `/version`, `vision doctor` and `vision update` all go through here."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass

from vision import models
from vision.models import provider_label

PROVIDERS = ("claude", "codex", "grok")

# Every CLI ships its own updater; `--version` output differs per tool, hence the parsing below.
_UPDATE_ARGS = {"claude": ["update"], "codex": ["update"], "grok": ["update"]}
_INSTALL_HINTS = {
    "claude": "curl -fsSL https://claude.ai/install.sh | bash, then run `claude` to log in",
    "codex": "npm i -g @openai/codex, then run `codex login`",
    "grok": "https://x.ai/cli, then run `grok login`",
}
_VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.]+)?")


@dataclass
class CliInfo:
    provider: str
    path: str | None = None
    version: str | None = None  # "2.1.278"
    raw: str = ""  # the whole `--version` line, e.g. "grok 1.0.34 (3736acbc8658) [stable]"
    error: str | None = None  # why there is no version: not installed, or `--version` failed

    @property
    def label(self) -> str:
        return provider_label(self.provider, cli=True)

    @property
    def ok(self) -> bool:
        return self.version is not None

    def line(self) -> str:
        """One plain-text line: 'Claude Code 2.1.278 (/home/me/.local/bin/claude)'."""
        if self.ok:
            return f"{self.label} {self.version} ({self.path})"
        return f"{self.label}: {self.error}"


def find_cli(provider: str) -> str:
    """The executable for a provider; raises that provider's error when it is not installed."""
    if provider == "claude":
        from vision.brain import find_claude

        return find_claude()
    if provider == "codex":
        from vision.codex import find_codex

        return find_codex()
    if provider == "grok":
        from vision.grok import find_grok

        return find_grok()
    raise ValueError(f"unknown provider {provider!r}")


def install_hint(provider: str) -> str:
    return _INSTALL_HINTS.get(provider, "")


def parse_version(text: str) -> str | None:
    """'codex-cli 0.155.1' → '0.155.1'; '2.1.278 (Claude Code)' → '2.1.278'."""
    m = _VERSION.search(text or "")
    return m.group(0) if m else None


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)  # `claude` refuses to nest inside a Claude Code session
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    return env


def cli_version(provider: str, timeout: float = 20) -> CliInfo:
    """Ask one CLI for its version. Never raises: a missing tool or a hung `--version` is reported in `error`."""
    info = CliInfo(provider)
    try:
        info.path = find_cli(provider)
    except Exception as e:  # noqa: BLE001 - each provider raises its own error class
        info.error = f"not installed ({e})"
        return info
    try:
        r = subprocess.run([info.path, "--version"], capture_output=True, text=True, timeout=timeout, env=_env(), encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as e:
        info.error = f"`{os.path.basename(info.path)} --version` failed: {e}"
        return info
    info.raw = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else ""
    info.version = parse_version(info.raw)
    if info.version is None:
        info.error = f"could not read the version from {info.raw!r}" if info.raw else f"`{os.path.basename(info.path)} --version` printed nothing"
    return info


def cli_versions(providers=PROVIDERS) -> list[CliInfo]:
    return [cli_version(p) for p in providers]


@dataclass
class UpdateResult:
    provider: str
    before: str | None
    after: str | None
    returncode: int
    output: str = ""  # captured stdout+stderr (empty when it streamed to the terminal)
    error: str | None = None  # the CLI is missing, or its updater could not be started

    @property
    def label(self) -> str:
        return provider_label(self.provider, cli=True)

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0

    @property
    def changed(self) -> bool:
        return self.ok and self.before != self.after

    def summary(self) -> str:
        """'Claude Code 2.1.278 → 2.1.280', 'Codex 0.155.1 (already up to date)', or what went wrong."""
        if self.error:
            return f"{self.label}: {self.error}"
        if not self.ok:
            tail = self.output.strip().splitlines()[-1] if self.output.strip() else f"exit status {self.returncode}"
            return f"{self.label} update failed: {tail}"
        if self.changed:
            return f"{self.label} {self.before or '?'} → {self.after or '?'}"
        return f"{self.label} {self.after or self.before or '?'} (already up to date)"


# ---------------------------------------------------------------- is an update available?
# Claude Code's installer resolves "latest" from this bucket (npm is the fallback); Codex is a standalone
# build whose numbers track the npm package; Grok has a check-only mode that honours its release channel.
_CLAUDE_LATEST_URL = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/latest"
_NPM_LATEST = {"claude": "https://registry.npmjs.org/@anthropic-ai/claude-code/latest", "codex": "https://registry.npmjs.org/@openai/codex/latest"}
CHECK_TTL = 6 * 3600  # a check per provider is good for this long (the status bar asks on every repaint)


@dataclass
class UpdateCheck:
    provider: str
    current: str | None = None
    latest: str | None = None
    error: str | None = None
    at: float = 0.0

    @property
    def label(self) -> str:
        return provider_label(self.provider, cli=True)

    @property
    def available(self) -> bool | None:
        """True when a newer version is out, False when current, None when the check could not tell."""
        if self.error or not self.current or not self.latest:
            return None
        return _newer(self.latest, self.current)

    def status(self) -> str:
        """Short status-bar copy: 'Claude Code 2.1.280 available · /update'; nothing when current or unknown."""
        return f"{self.label} {self.latest} available · /update" if self.available else ""


def _key(v: str) -> tuple:
    head, _, pre = v.partition("-")
    nums = tuple(int(x) for x in head.split(".") if x.isdigit())
    return nums, pre == "", pre  # a pre-release sorts below its release


def _newer(latest: str, current: str) -> bool:
    try:
        return _key(latest) > _key(current)
    except ValueError:
        return latest != current


def _http_text(url: str, timeout: float) -> str:
    import urllib.request

    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vision"}), timeout=timeout) as r:
        return r.read().decode()


def latest_version(provider: str, timeout: float = 15) -> str:
    """The newest released version of a CLI. Raises (OSError, ValueError, …) when it cannot be found."""
    if provider == "grok":
        r = subprocess.run([find_cli("grok"), "update", "--check", "--json"], capture_output=True, text=True, timeout=timeout, env=_env(), encoding="utf-8")
        data = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
        if data.get("error"):
            raise ValueError(data["error"])
        if not data.get("latestVersion"):
            raise ValueError((r.stderr or r.stdout).strip()[:200] or "no version in `grok update --check`")
        return data["latestVersion"]
    if provider == "claude":
        try:
            v = parse_version(_http_text(_CLAUDE_LATEST_URL, timeout))
            if v:
                return v
        except OSError:
            pass
    v = json.loads(_http_text(_NPM_LATEST[provider], timeout)).get("version")
    if not v:
        raise ValueError(f"no version from npm for {provider}")
    return v


_checks: dict[str, UpdateCheck] = {}
_checks_lock = threading.Lock()


def check_update(provider: str, *, max_age: float = CHECK_TTL) -> UpdateCheck:
    """Compare the installed CLI with the newest release. Cached per provider for `max_age` seconds;
    never raises (network or CLI trouble lands in `error`, and the status bar then says nothing)."""
    with _checks_lock:
        cached = _checks.get(provider)
    if cached and time.time() - cached.at < max_age:
        return cached
    info = cli_version(provider)
    check = UpdateCheck(provider, current=info.version, at=time.time())
    if not info.ok:
        check.error = info.error
    else:
        try:
            check.latest = latest_version(provider)
        except Exception as e:  # noqa: BLE001 - offline, endpoint changed, CLI hiccup: all mean "unknown"
            check.error = str(e) or type(e).__name__
    with _checks_lock:
        _checks[provider] = check
    return check


def peek_check(provider: str, *, max_age: float = CHECK_TTL) -> UpdateCheck | None:
    """The cached check when it is fresh, else None — never touches the network (for the status bar)."""
    with _checks_lock:
        cached = _checks.get(provider)
    return cached if cached and time.time() - cached.at < max_age else None


def forget_checks(providers=PROVIDERS) -> None:
    """Drop cached checks (after an update, so the status bar re-reads the installed version)."""
    with _checks_lock:
        for p in providers:
            _checks.pop(p, None)


# ---------------------------------------------------------------- each provider's model list
# Asked of the provider itself (see models.CATALOGUES_ON), all in about a second and none of it billed:
# Claude Code's `initialize` handshake, `codex debug models` and `grok models` (each rewrites its CLI's
# cache, which Vision then re-reads), and llama-server's /v1/models.
MODEL_PROVIDERS = ("claude", "codex", "grok", "local")
MODELS_TTL = 6 * 3600  # a long-running Vision re-asks this often
CLAUDE_MODELS_RECHECK = 600  # sooner when a Claude turn ran on a model the list lacks (at most this often)
_models_at: dict[str, float] = {}
_models_busy = {p: threading.Lock() for p in MODEL_PROVIDERS}
_REFRESH_ARGS = {"codex": ["debug", "models"], "grok": ["models"]}


def claude_catalogue(timeout: float = 30) -> list[dict]:
    """Claude Code's /model list (value, resolvedModel, supportedEffortLevels, …) from the `initialize`
    control request. No prompt is sent, so no model runs and nothing is billed. Raises when it cannot be read."""
    cmd = [find_cli("claude"), "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--tools", "", "--strict-mcp-config", "--no-session-persistence"]
    req = json.dumps({"type": "control_request", "request_id": "vision-models", "request": {"subtype": "initialize"}})
    r = subprocess.run(cmd, input=req + "\n", capture_output=True, text=True, timeout=timeout, env=_env(),
                       cwd=os.path.expanduser("~"), encoding="utf-8")
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        resp = ev.get("response") or {} if ev.get("type") == "control_response" else {}
        if resp.get("request_id") == "vision-models":
            models_list = (resp.get("response") or {}).get("models")
            if isinstance(models_list, list) and models_list:
                return models_list
            raise ValueError(resp.get("error") or "no models in Claude Code's initialize response")
    raise ValueError((r.stderr or r.stdout).strip()[-200:] or f"claude exited {r.returncode} without an initialize response")


def local_model_ids(timeout: float = 5) -> list[str]:
    """The model ids llama-server in [local] serves. Raises when it cannot be reached."""
    from vision.config import load_config

    data = json.loads(_http_text(load_config().local.base_url.rstrip("/") + "/models", timeout))
    return [str(m["id"]) for m in data.get("data") or [] if isinstance(m, dict) and m.get("id")]


def refresh_models(provider: str, *, max_age: float = MODELS_TTL) -> bool:
    """Ask one provider for its model list and swap it in (models.set_claude_catalogue and friends). At
    most one at a time per provider, and not again within `max_age` seconds. Never raises; True if replaced."""
    lock = _models_busy[provider]
    if not lock.acquire(blocking=False):
        return False
    try:
        if time.time() - _models_at.get(provider, 0.0) < max_age:
            return False
        _models_at[provider] = time.time()
        try:
            if provider == "claude":
                return models.set_claude_catalogue(claude_catalogue())
            if provider == "local":
                return models.set_local_models(local_model_ids())
            subprocess.run([find_cli(provider), *_REFRESH_ARGS[provider]], capture_output=True, timeout=60,
                           env=_env(), stdin=subprocess.DEVNULL, cwd=os.path.expanduser("~"))
            return models.reload_cli_cache(provider)
        except Exception:  # noqa: BLE001 - not installed, offline, Mac asleep, format changed: keep the list we have
            return False
    finally:
        lock.release()


def refresh_models_soon(providers=MODEL_PROVIDERS, *, max_age: float = MODELS_TTL) -> None:
    """refresh_models for each stale provider, on daemon threads (callers are on a turn or the UI). Off under tests."""
    if not models.CATALOGUES_ON:
        return
    for p in providers:
        if time.time() - _models_at.get(p, 0.0) >= max_age:
            threading.Thread(target=refresh_models, args=(p,), kwargs={"max_age": max_age}, daemon=True).start()


def update_cli(provider: str, *, stream: bool = False, timeout: float = 600) -> UpdateResult:
    """Run the CLI's own updater (`claude update`, `codex update`, `grok update`). With `stream` the
    updater's output goes straight to the terminal; otherwise it is captured into the result."""
    before = cli_version(provider)
    if not before.ok:
        return UpdateResult(provider, None, None, 1, error=before.error)
    cmd = [before.path, *_UPDATE_ARGS[provider]]
    try:
        r = subprocess.run(cmd, capture_output=not stream, text=True, timeout=timeout, env=_env(), stdin=subprocess.DEVNULL, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return UpdateResult(provider, before.version, before.version, 1, error=f"`{os.path.basename(cmd[0])} update` did not finish in {timeout:.0f}s")
    except OSError as e:
        return UpdateResult(provider, before.version, before.version, 1, error=f"could not run `{os.path.basename(cmd[0])} update`: {e}")
    output = "" if stream else ((r.stdout or "") + (r.stderr or ""))
    forget_checks((provider,))
    refresh_models_soon((provider,), max_age=0)  # a new CLI may list new models
    after = cli_version(provider)
    return UpdateResult(provider, before.version, after.version or before.version, r.returncode, output=output)


def update_clis(providers=PROVIDERS, *, stream: bool = False, on_start=None) -> list[UpdateResult]:
    """Update several CLIs in order; `on_start(provider)` is called before each one."""
    results = []
    for p in providers:
        if on_start:
            on_start(p)
        results.append(update_cli(p, stream=stream))
    return results


def _parse_providers(arg: str, command: str, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """'' or 'all' → every provider; 'claude codex' → those two (in PROVIDERS order). Unknown words raise."""
    words = [w.lower() for w in (arg or "").split()]
    if not words or "all" in words:
        return PROVIDERS
    bad = [w for w in words if w not in PROVIDERS and w not in extra]
    if bad:
        raise ValueError(f"{command} takes all, claude, codex or grok, not {bad[0]!r}")
    return tuple(p for p in PROVIDERS + extra if p in words)


def parse_update_arg(arg: str) -> tuple[str, ...]:
    return _parse_providers(arg, "update")


def parse_version_arg(arg: str) -> tuple[bool, tuple[str, ...]]:
    """'' or 'all' → (True, every provider); 'codex' → (False, ('codex',)); 'vision grok' → (True, ('grok',))."""
    picked = _parse_providers(arg, "version", extra=("vision",))
    if picked == PROVIDERS:
        return True, PROVIDERS
    return "vision" in picked, tuple(p for p in picked if p != "vision")

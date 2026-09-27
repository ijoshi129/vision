"""Grok brain: drives xAI's Grok models through the Grok CLI in headless mode.

Uses `grok --prompt-file … --output-format streaming-json` as documented for scripting, on the
normal grok.com login. Conversation continuity uses Grok's own session store via `--resume <id>`.

Verified against grok 1.0.34:
- Headless is triggered by `--prompt-file` (or `-p`); stdin is not the prompt.
- `--verbatim` sends the file exactly; `--rules` appends Vision's persona (native prompt stays).
- JSONL events: text{data}, thought, tool_call, tool_call_update, usage, plan, end{sessionId, usage},
  error{message}. `end` is last. Field names on the stream are camelCase (`sessionId`, `toolName`).
- Auto mode is `--always-approve` (headless cannot answer permission prompts). Plan mode is the
  read-only sandbox; Grok's own plan-approval UI is not used (`--no-plan`).
- `--deny` accepts Claude-style `Bash(prefix:*)` rules. `GROK_MEMORY=0` so Vision's MEMORY.md is
  the only long-term memory.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.text import Text

from vision import compat
from vision import usage as usage_ui
from vision.brain import context_read
from vision.config import STATE_DIR, BrainConfig, weather_ready
from vision.persona import system_prompt
from vision.reply import READING, THINKING, ReplyText, dedupe_status

if TYPE_CHECKING:
    from vision.brain import Turn

LAST_SESSION_FILE = STATE_DIR / "last_session.grok"  # JSON {"id": session_id, "model": slug, "at": ts}
USAGE_FILE = STATE_DIR / "usage.grok.json"
SANDBOXES = ("off", "workspace", "read-only", "strict")
GROK_HOME = os.path.expanduser(os.environ.get("GROK_HOME", "~/.grok"))
GROK_SESSIONS = os.path.join(GROK_HOME, "sessions")
GROK_AUTH_FILE = os.path.join(GROK_HOME, "auth.json")
GROK_SETTINGS_CACHE = os.path.join(GROK_HOME, "settings_cache.json")
GROK_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
_PRODUCT_LABELS = {
    "GrokBuild": "Build",
    "GrokImagine": "Imagine",
    "GrokChat": "Chat",
    "GrokVoice": "Voice",
    "GrokAPI": "API",
    "PRODUCT_GROK_BUILD": "Build",
    "PRODUCT_GROK_IMAGINE": "Imagine",
    "PRODUCT_GROK_CHAT": "Chat",
    "PRODUCT_GROK_VOICE": "Voice",
    "PRODUCT_GROK_API": "API",
}
# Status-row labels: Grok's internal tool ids → the names the rest of Vision already shows.
TOOL_LABELS = {
    "run_terminal_command": "Bash",
    "run_terminal_cmd": "Bash",
    "read_file": "Read",
    "search_replace": "Edit",
    "write": "Write",
    "list_dir": "Glob",
    "grep": "Grep",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
    "spawn_subagent": "Agent",
}
# Headless cannot answer these; leaving them on can stall a turn waiting for a TUI that is not there.
_HEADLESS_DISALLOWED = "ask_user_question,send_feedback,enter_plan_mode,exit_plan_mode"


class GrokError(RuntimeError):
    pass


def find_grok() -> str:
    exe = shutil.which("grok")
    if exe and compat.is_batch_file(exe):
        raise GrokError(f"Found Grok as {exe}, a .cmd launcher, which cannot pass Vision's multi-line instructions safely. Install the native grok.exe.")
    if exe:
        return exe
    home = os.path.expanduser(os.environ.get("GROK_HOME", "~/.grok"))
    for cand in (os.path.join(home, "bin", "grok"), os.path.expanduser("~/.local/bin/grok"), "/usr/local/bin/grok", "/usr/bin/grok"):
        if os.path.exists(cand):
            return cand
    raise GrokError("Grok CLI ('grok') not found on PATH. Install it (https://x.ai/cli) and run `grok login` once.")


def _auth_entry() -> dict | None:
    from vision.config import cli_logins_allowed

    if not cli_logins_allowed():
        return None
    try:
        data = json.loads(open(GROK_AUTH_FILE, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    entries = [v for v in data.values() if isinstance(v, dict) and v.get("key")]
    if not entries:
        return None
    return max(entries, key=lambda e: str(e.get("expires_at") or e.get("create_time") or ""))


def _cent(obj) -> int:
    if isinstance(obj, dict):
        try:
            return int(obj.get("val") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _parse_iso(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _period_label(period_type: str | None) -> str:
    raw = (period_type or "").replace("USAGE_PERIOD_TYPE_", "").strip()
    return raw.lower() if raw else "weekly"


def _product_label(name: str) -> str:
    return _PRODUCT_LABELS.get(name, name.replace("PRODUCT_GROK_", "").replace("Grok", "") or name)


def _cached_plan_name() -> str | None:
    try:
        raw = json.loads(open(GROK_SETTINGS_CACHE, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None
    payload = raw.get("payload") if isinstance(raw, dict) else None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
    name = settings.get("subscription_tier_display") or settings.get("subscription_tier")
    return str(name) if name else None


def _as_percent(value) -> float | None:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= pct <= 1:
        pct *= 100
    return pct


def fetch_subscription() -> dict | None:
    """Live grok.com allowance (weekly pool on unified billing). None when login/billing is unavailable."""
    entry = _auth_entry()
    if not entry:
        return None
    headers = {
        "Authorization": f"Bearer {entry['key']}",
        "Accept": "application/json",
        "X-XAI-Token-Auth": "xai-grok-cli",
    }
    if entry.get("user_id"):
        headers["x-userid"] = str(entry["user_id"])
    try:
        req = urllib.request.Request(GROK_BILLING_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    period = cfg.get("currentPeriod") if isinstance(cfg.get("currentPeriod"), dict) else {}
    used = _as_percent(cfg.get("creditUsagePercent"))
    if used is None:
        limit = _cent(cfg.get("monthlyLimit"))
        if limit:
            used = 100.0 * _cent(cfg.get("used")) / limit
    products = []
    for item in cfg.get("productUsage") or []:
        if not isinstance(item, dict):
            continue
        pct = _as_percent(item.get("usagePercent"))
        if pct is None:
            continue
        products.append({"label": _product_label(str(item.get("product") or "?")), "percent": pct})
    resets = _parse_iso(period.get("end") or cfg.get("billingPeriodEnd"))
    plan = payload.get("subscriptionTier") or payload.get("subscription_tier") or _cached_plan_name()
    if used is None and not products and not resets:
        return None
    return {
        "plan": str(plan) if plan else "",
        "used_percent": used,
        "period": _period_label(period.get("type")),
        "resets_at": resets,
        "products": products,
        "prepaid_cents": _cent(cfg.get("prepaidBalance")),
        "on_demand_used_cents": _cent(cfg.get("onDemandUsed")),
        "on_demand_cap_cents": _cent(cfg.get("onDemandCap")),
    }


def sandbox_for(cfg: BrainConfig) -> str:
    """Grok's kernel sandbox stands in for Claude's per-tool allow list.

    Plan mode → read-only. Auto mode → `[grok].sandbox` when one is named there, else `off`
    (unrestricted, matching Claude auto / Codex danger-full-access).
    """
    if getattr(cfg, "mode", "auto") == "plan":
        return "read-only"
    chosen = (getattr(cfg, "grok", None) and cfg.grok.sandbox) or "auto"
    return chosen if chosen in SANDBOXES else "off"


def _cli_model(alias: str) -> str:
    """`/model grok` is a nickname for grok-4.6; the CLI wants the real id."""
    from vision.models import find

    m = find(alias or "")
    return m.alias if m else (alias or "")


class GrokBrain:
    """Same interface as the Claude brain so the CLI does not care which one is thinking."""

    provider = "grok"

    def __init__(self, cfg: BrainConfig, voice_mode: bool = False, session_id: str | None = None):
        self.cfg = cfg
        self.voice_mode = voice_mode
        self.task_mode = False
        self.session_id = session_id
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.grok = find_grok()
        self.workdir = os.path.abspath(os.path.expanduser(cfg.workdir)) if cfg.workdir else os.getcwd()
        self.last_usage: dict | None = None
        self.model: str | None = None
        self._model_seen_for: str | None = None
        self.handoff: str | None = None
        # (tokens the model read on its last response, its window): the `% ctx` figure / phone gauge.
        self.context: tuple[int, int] | None = None

    # -- session helpers -------------------------------------------------
    @staticmethod
    def _read_last() -> dict:
        try:
            return json.loads(LAST_SESSION_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def last_session_id() -> str | None:
        return GrokBrain._read_last().get("id") or None

    def context_window(self) -> int:
        """The model's window from the CLI's models cache (~/.grok/models_cache.json), else 256k."""
        try:
            with open(os.path.join(GROK_HOME, "models_cache.json"), encoding="utf-8") as f:
                info = json.load(f).get("models", {}).get(_cli_model(self.cfg.model), {}).get("info", {})
            return int(info.get("context_window") or 0) or 256_000
        except (OSError, ValueError, AttributeError):
            return 256_000

    @staticmethod
    def last_session_model() -> str | None:
        return GrokBrain._read_last().get("model") or None

    def _remember_session(self) -> None:
        if self.task_mode:
            return
        if self.session_id:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            LAST_SESSION_FILE.write_text(json.dumps({"id": self.session_id, "model": self.cfg.model, "at": time.time()}), encoding="utf-8")

    def new_session(self) -> None:
        self.session_id = None
        self.last_usage = None
        self.handoff = None
        self.context = None

    def resume(self, session_id: str) -> None:
        """Continue an earlier session from the next turn on (and make it the `-c` target)."""
        self.session_id = session_id
        self.last_usage = None
        self.handoff = None
        self.context = None
        self._remember_session()

    def resolved_model(self) -> str | None:
        if self.model and self._model_seen_for == self.cfg.model:
            return self.model
        return _cli_model(self.cfg.model) or self.cfg.model or None

    # -- main entry point -------------------------------------------------
    def _command(self, prompt_path: str) -> list[str]:
        sandbox = sandbox_for(self.cfg)
        persona = system_prompt(
            self.voice_mode, self.cfg.address_user_as, self.workdir, self.cfg.allowed_tools,
            provider="grok", sandbox=sandbox, denied_tools=self.cfg.denied_tools, mode=self.cfg.mode,
            weather=weather_ready(self.cfg),
        )
        if self.task_mode:
            from vision.delegation import worker_prompt

            persona = worker_prompt(self.cfg, self.workdir, self.provider, sandbox)
        cmd = [
            self.grok,
            "--prompt-file", prompt_path,
            "--verbatim",
            "--output-format", "streaming-json",
            "--cwd", self.workdir,
            "--no-auto-update",
            "--no-plan",
            "--always-approve",
            "--sandbox", sandbox,
            "--rules", persona,
            "--disallowed-tools", _HEADLESS_DISALLOWED,
        ]
        model = _cli_model(self.cfg.model)
        if model:
            cmd += ["--model", model]
        if self.cfg.effort:
            cmd += ["--effort", self.cfg.effort]
        for rule in self.cfg.denied_tools or []:
            if not rule.startswith("PowerShell("):  # Claude Code's Windows shell; Grok has no such tool
                cmd += ["--deny", rule]
        if self.task_mode:
            from vision.delegation import RESULT_SCHEMA

            cmd += ["--json-schema", json.dumps(RESULT_SCHEMA)]
        for extra in getattr(getattr(self.cfg, "grok", None), "extra_args", []) or []:
            cmd.append(extra)
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def ask(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_question: Callable[[list[dict]], dict[str, str] | None] | None = None,  # Claude-only
        on_agent: Callable | None = None,  # sub-agent rows, from its spawn_subagent calls (vision.subagents)
        on_tool: Callable | None = None,  # Claude-only; grok tool_calls are status labels
    ) -> Turn:
        from vision.brain import Turn, brain_env, inject_handoff

        env = brain_env("grok")
        turn = Turn(session_id=self.session_id, model=_cli_model(self.cfg.model) or None)
        sandbox = sandbox_for(self.cfg)
        if compat.WINDOWS and sandbox != "off":
            # Grok always runs with --always-approve and leaves every limit to its kernel sandbox, which
            # is Linux/macOS machinery. Refuse rather than run "read-only" with nothing enforcing it.
            turn.is_error = True
            turn.error = (f"Grok's {sandbox} sandbox is not available on Windows, so Vision will not run Grok "
                          + ("in plan mode. Switch to auto (Shift-Tab) or use Claude for planning."
                             if self.cfg.mode == "plan" else "with it. Set [grok] sandbox = \"off\" to run Grok unrestricted."))
            return turn
        prompt = inject_handoff(self, prompt)
        reply = ReplyText(on_text)
        on_status = dedupe_status(on_status)
        pending: set[str] = set()
        last_error = ""
        from vision.subagents import AgentTracker, grok_tool_call, grok_tool_update

        subs = AgentTracker(turn, on_agent, model=_cli_model(self.cfg.model) or "", effort=self.cfg.effort or "")
        sub_calls: dict[str, str] = {}  # sub-agent tool calls by toolCallId
        fd, prompt_path = tempfile.mkstemp(prefix="vision-grok-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
            raise

        self._killed = False
        with self._lock:
            self._proc = subprocess.Popen(
                self._command(prompt_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.workdir,
                env=env,
                text=True,
                bufsize=1,
                encoding="utf-8",
            )
        proc = self._proc
        diagnostics: list[str] = []
        completed = False
        try:
            assert proc.stdout
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics.append(line)
                    continue
                t = ev.get("type")
                if t == "text" and ev.get("data"):
                    reply.add(ev["data"])
                elif t == "thought":
                    if on_status:
                        on_status(THINKING)  # real reasoning tokens (one line per chunk; deduped)
                elif t == "tool_call":
                    grok_tool_call(subs, ev, sub_calls)
                    name = ev.get("toolName") or ev.get("title") or "tool"
                    label = TOOL_LABELS.get(name, name)
                    turn.tools_used.append(label)
                    tid = ev.get("toolCallId") or name
                    pending.add(tid)
                    reply.tool()
                    if on_status:
                        on_status(label)
                elif t == "tool_call_update":
                    grok_tool_update(subs, ev, sub_calls)
                    status = ev.get("status") or ""
                    tid = ev.get("toolCallId")
                    if status in ("completed", "failed", "cancelled", "error") and tid:
                        pending.discard(tid)
                    if on_status and not pending:
                        on_status(READING)
                elif t == "usage" and isinstance(ev.get("usage"), dict):
                    # One line per model response: its full prompt (uncached + both cache buckets) is the context.
                    turn.usage = ev["usage"]
                    self.context = (context_read(ev["usage"]), self.context_window())
                elif t == "end":
                    turn.session_id = ev.get("sessionId") or ev.get("session_id") or turn.session_id
                    if isinstance(ev.get("usage"), dict):
                        turn.usage = ev["usage"]
                    cost = ev.get("total_cost_usd")
                    if isinstance(cost, (int, float)):
                        turn.cost_usd = float(cost)
                    if self.task_mode:
                        turn.data = ev.get("structured_output") or ev.get("structuredOutput")
                    stop = (ev.get("stopReason") or ev.get("stop_reason") or "").lower()
                    if stop in ("cancelled", "canceled"):
                        turn.is_error = True
                        turn.error = "cancelled"
                    completed = True
                    break
                elif t == "error":
                    last_error = ev.get("message") or last_error or "grok error"
                    turn.is_error = True
                    turn.error = last_error
                    sid = ev.get("sessionId") or ev.get("session_id")
                    if sid:
                        turn.session_id = sid
                    if isinstance(ev.get("usage"), dict):
                        turn.usage = ev["usage"]
            if completed:
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    compat.terminate(proc)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            else:
                proc.wait()
        except KeyboardInterrupt:
            self.cancel()
            turn.is_error = True
            turn.error = "cancelled"
            turn.text = reply.finish()
            return turn
        finally:
            with self._lock:
                self._proc = None
            subs.close("cancelled" if turn.error == "cancelled" else "cut off when the turn ended")
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

        turn.text = reply.finish()
        diagnostic_text = "\n".join(diagnostics)
        err_blob = f"{turn.error} {diagnostic_text}".lower()
        if not completed and proc.returncode not in (0, None):
            turn.is_error = True
            if proc.returncode in (-15, -9, 130, 143) or proc.returncode < 0 or (compat.WINDOWS and self._killed):
                turn.error = "cancelled"
            else:
                err_lines = [ln for ln in diagnostics if ln]
                turn.error = turn.error or last_error or (err_lines[-1] if err_lines else f"grok exited with code {proc.returncode}")
            if "resume" in err_blob or "session" in err_blob or "couldn't start session" in err_blob:
                self.session_id = None
                try:
                    LAST_SESSION_FILE.unlink()
                except FileNotFoundError:
                    pass
        else:
            if turn.session_id:
                self.session_id = turn.session_id
                self._remember_session()
            self.handoff = None
            self.model = turn.model or _cli_model(self.cfg.model) or self.model
            self._model_seen_for = self.cfg.model
            if turn.usage:
                self._record_usage(turn.usage, turn)
        return turn

    # -- usage --------------------------------------------------------------
    def _record_usage(self, usage: dict, turn) -> None:
        prev = self.last_usage or self.cached_usage() or {}
        same = bool(turn.session_id and prev.get("session_id") == turn.session_id)
        self.last_usage = {
            "provider": "grok",
            "model": self.cfg.model,
            "session_id": turn.session_id,
            "last": usage,
            "turns": int(prev.get("turns", 0)) + 1 if same else 1,
            "at": time.time(),
        }
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            USAGE_FILE.write_text(json.dumps(self.last_usage), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def cached_usage() -> dict | None:
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def usage_report(self) -> str | None:
        return None  # Grok has no equivalent of Claude Code's /usage text; see usage_renderable().

    def ping_usage(self) -> dict | None:
        return self.last_usage

    def usage_renderable(self, full: bool = False):
        """A table for /usage in the same shape as Claude's: the grok.com weekly pool, bars, resets."""
        sub = fetch_subscription()
        if not sub:
            return Text("Grok usage unavailable: needs a live grok.com login (run `grok login`).", style="dim")
        plan = sub.get("plan") or None
        period = sub.get("period") or "weekly"
        window = {"weekly": "Current week", "monthly": "Current month", "daily": "Current day"}.get(period, f"Current {period}")
        t = usage_ui.usage_table("Grok", plan)
        usage_ui.add_window(t, window, sub.get("used_percent"), sub.get("resets_at"))
        if full:
            for item in sub.get("products") or []:
                usage_ui.add_window(t, f"{window} ({item['label']})", item["percent"], sub.get("resets_at"))
        parts = [t]
        extras = []
        prepaid = sub.get("prepaid_cents") or 0
        cap = sub.get("on_demand_cap_cents") or 0
        od_used = sub.get("on_demand_used_cents") or 0
        if prepaid:
            extras.append(f"extra credits ${prepaid / 100:.2f}")
        if cap:
            extras.append(f"on-demand ${od_used / 100:.2f} / ${cap / 100:.2f}")
        if extras:
            parts.append(Text(" · ".join(extras), style="dim"))
        return usage_ui.usage_group(*parts)

    def cancel(self) -> None:
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            self._killed = True
            try:
                compat.terminate(proc)
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

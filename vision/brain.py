"""The brain: drives Claude through the Claude Code CLI in headless (print) mode.

This uses `claude -p` exactly as documented for scripting, so it runs on your normal
Claude Code login and subscription. No API keys, no token extraction.
Conversation continuity uses Claude Code's own session store via `--resume`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from vision.config import STATE_DIR, WORKSPACE_DIR, BrainConfig
from vision.persona import system_prompt

LAST_SESSION_FILE = STATE_DIR / "last_session"
USAGE_FILE = STATE_DIR / "usage.json"


class BrainError(RuntimeError):
    pass


def find_claude() -> str:
    exe = shutil.which("claude")
    if exe:
        return exe
    for cand in (os.path.expanduser("~/.local/bin/claude"), "/usr/local/bin/claude", "/usr/bin/claude"):
        if os.path.exists(cand):
            return cand
    raise BrainError("Claude Code CLI ('claude') not found on PATH. Install it and run `claude` once to log in.")


@dataclass
class Turn:
    text: str = ""
    session_id: str | None = None
    tools_used: list[str] = field(default_factory=list)
    is_error: bool = False
    error: str = ""
    cost_usd: float | None = None
    duration_ms: int | None = None


class Brain:
    def __init__(self, cfg: BrainConfig, voice_mode: bool = False, session_id: str | None = None):
        self.cfg = cfg
        self.voice_mode = voice_mode
        self.session_id = session_id
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.claude = find_claude()
        self.workdir = os.path.abspath(os.path.expanduser(cfg.workdir)) if cfg.workdir else os.getcwd()
        self.last_usage: dict | None = None
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # -- session helpers -------------------------------------------------
    @staticmethod
    def last_session_id() -> str | None:
        try:
            sid = LAST_SESSION_FILE.read_text().strip()
            return sid or None
        except FileNotFoundError:
            return None

    def _remember_session(self) -> None:
        if self.session_id:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            LAST_SESSION_FILE.write_text(self.session_id)

    def new_session(self) -> None:
        self.session_id = None

    # -- main entry point -------------------------------------------------
    def _command(self) -> list[str]:
        cmd = [
            self.claude,
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            system_prompt(self.voice_mode, self.cfg.address_user_as, self.workdir, self.cfg.allowed_tools),
        ]
        if self.cfg.model:
            cmd += ["--model", self.cfg.model]
        if self.cfg.effort:
            cmd += ["--effort", self.cfg.effort]
        # --tools limits which tools exist at all; --allowedTools pre-approves them so
        # headless mode never has to prompt (anything unapproved is denied automatically).
        tools = ",".join(self.cfg.allowed_tools)
        cmd += ["--tools", tools, "--allowedTools", tools] if tools else ["--tools", ""]
        if self.cfg.denied_tools:
            cmd += ["--disallowedTools", ",".join(self.cfg.denied_tools)]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def ask(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> Turn:
        """Send one user turn. Streams text deltas to on_text; returns the completed Turn."""
        env = dict(os.environ)
        # Never let a parent Claude Code session's nesting guard interfere.
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)

        turn = Turn(session_id=self.session_id)
        streamed = []
        final_text_from_message = []
        saw_delta = False

        with self._lock:
            self._proc = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workdir,
                env=env,
                text=True,
                bufsize=1,
            )
        proc = self._proc
        try:
            assert proc.stdin and proc.stdout
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "system" and ev.get("subtype") == "init":
                    turn.session_id = ev.get("session_id") or turn.session_id
                elif t == "stream_event":
                    e = ev.get("event", {})
                    et = e.get("type")
                    if et == "content_block_delta":
                        d = e.get("delta", {})
                        if d.get("type") == "text_delta":
                            saw_delta = True
                            streamed.append(d["text"])
                            if on_text:
                                on_text(d["text"])
                    elif et == "content_block_start":
                        cb = e.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            name = cb.get("name", "tool")
                            turn.tools_used.append(name)
                            if on_status:
                                on_status(name)
                        elif cb.get("type") == "text" and saw_delta and streamed and not streamed[-1].endswith("\n"):
                            # A new text block after a tool call: keep paragraphs separated.
                            streamed.append("\n\n")
                            if on_text:
                                on_text("\n\n")
                elif t == "rate_limit_event":
                    self._record_usage(ev.get("rate_limit_info"))
                elif t == "assistant":
                    # Fallback source of text when partial messages are unavailable.
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            final_text_from_message.append(block["text"])
                elif t == "result":
                    turn.session_id = ev.get("session_id") or turn.session_id
                    turn.is_error = bool(ev.get("is_error")) or ev.get("subtype", "").startswith("error")
                    turn.cost_usd = ev.get("total_cost_usd")
                    turn.duration_ms = ev.get("duration_ms")
                    if turn.is_error:
                        turn.error = ev.get("result") or ev.get("error") or ev.get("subtype", "error")
                    elif not saw_delta and ev.get("result"):
                        final_text_from_message = [ev["result"]]
            proc.wait()
            stderr = proc.stderr.read() if proc.stderr else ""
        except KeyboardInterrupt:
            self.cancel()
            turn.is_error = True
            turn.error = "cancelled"
            turn.text = "".join(streamed)
            return turn
        finally:
            with self._lock:
                self._proc = None

        if saw_delta:
            turn.text = "".join(streamed)
        else:
            turn.text = "\n\n".join(final_text_from_message)
            if turn.text and on_text:
                on_text(turn.text)

        if proc.returncode not in (0, None) and not turn.text:
            turn.is_error = True
            turn.error = turn.error or (stderr.strip().splitlines() or ["claude exited with code %s" % proc.returncode])[-1]
            if "resume" in turn.error.lower() or "session" in turn.error.lower():
                # Stale session id: drop it so the next turn starts clean.
                self.session_id = None
        else:
            self.session_id = turn.session_id
            self._remember_session()
        return turn

    # -- subscription usage ---------------------------------------------
    def _record_usage(self, info: dict | None) -> None:
        if not info:
            return
        import time

        self.last_usage = {"info": info, "at": time.time()}
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            USAGE_FILE.write_text(json.dumps(self.last_usage))
        except OSError:
            pass

    @staticmethod
    def cached_usage() -> dict | None:
        try:
            return json.loads(USAGE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def ping_usage(self) -> dict | None:
        """Make the cheapest possible Claude call just to read the current rate-limit windows."""
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        cmd = [self.claude, "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence",
               "--model", "haiku", "--tools", "", "--max-turns", "1"]
        try:
            r = subprocess.run(cmd, input="Reply with OK.", capture_output=True, text=True, cwd=self.workdir, env=env, timeout=120)
        except (subprocess.TimeoutExpired, OSError):
            return None
        for line in r.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "rate_limit_event":
                self._record_usage(ev.get("rate_limit_info"))
        return self.last_usage

    def cancel(self) -> None:
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

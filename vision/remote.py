"""Attach: a terminal `vision` chat joins a conversation that `vision serve` runs (one the phone opened).

The mirror of link.py. There a terminal chat announces itself so the server can list it on the phone;
here the terminal dials the server's own WebSocket, the way the phone does, and follows one of its
chats. The server stays the only owner of the agent session: what is typed here goes over as a
`message` frame, the reply streams back as the same start/delta/status/agent/tool/question/done
frames every client gets, and a turn the phone starts shows up here too (`on_foreign_turn`). So the
phone and the terminal read one conversation, not two copies of the same session id.

`vision serve` drops ~/.local/state/vision/serve.json (pid, url) while it runs; `running_server()`
reads it and the pairing token, so no address or token is typed.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from vision.brain import ToolCall, Turn
from vision.config import STATE_DIR

SERVE_FILE = STATE_DIR / "serve.json"
CONNECT_TIMEOUT = 5.0


# ---------------------------------------------------------------- finding the server
def write_serve_descriptor(host: str, port: int) -> None:
    """Called by `vision serve`: where a terminal on this machine can dial it."""
    reach = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "url": f"http://{reach}:{port}", "port": port, "started": time.time()}
    tmp = SERVE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, SERVE_FILE)


def remove_serve_descriptor() -> None:
    try:
        if json.loads(SERVE_FILE.read_text()).get("pid") == os.getpid():
            SERVE_FILE.unlink()
    except (OSError, ValueError):
        pass


def running_server() -> dict | None:
    """{pid, url, port, token} for the `vision serve` on this machine, or None (a dead pid's
    descriptor is swept away)."""
    from vision.link import pid_alive

    try:
        info = json.loads(SERVE_FILE.read_text())
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not pid_alive(pid):
        try:
            SERVE_FILE.unlink()
        except OSError:
            pass
        return None
    from vision.server import TOKEN_FILE

    try:
        token = TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    if not token:
        return None
    return {"pid": pid, "url": info["url"], "port": info.get("port"), "token": token}


def _get(server: dict, path: str, timeout: float = CONNECT_TIMEOUT):
    req = urllib.request.Request(server["url"] + path, headers={"Authorization": f"Bearer {server['token']}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  (our own loopback server)
        return json.loads(resp.read().decode())


def list_remote_chats(server: dict) -> list[dict]:
    """The server's own chats (the ones a terminal can join), newest first. Terminal chats it lists
    are other `vision` processes on this machine and are left out: they are joined by sitting at them."""
    chats = [c for c in _get(server, "/chats") if c.get("source", "server") == "server"]
    return sorted(chats, key=lambda c: c.get("updated") or 0, reverse=True)


def remote_history(server: dict, chat_id: str, limit: int = 200) -> list[dict]:
    """The chat's transcript as the phone sees it: typed and spoken turns, voice-only ones included."""
    return _get(server, f"/history?chat={chat_id}&limit={limit}")


def remote_rows(chats: list[dict], now: float | None = None) -> list[tuple[str, str, str]]:
    """(value, label, description) picker rows, `remote:<chat id>` values."""
    now = now or time.time()
    rows = []
    for c in chats:
        age = _age(now - float(c.get("updated") or c.get("created") or now))
        desc = " · ".join(p for p in (c.get("model", ""), age, "replying…" if c.get("busy") else "") if p)
        rows.append((remote_key(c), c.get("title") or "new conversation", desc))
    return rows


def live_sessions(chats: list[dict]) -> dict[str, dict]:
    """Phone chats by the saved session they are running, keyed like `session_key` (provider:id).
    That session is on disk too, so it also lists in its provider's tab; resuming it there would
    fork it into a second process instead of joining the one the server drives."""
    return {f"{c['provider']}:{c['session_id']}": c for c in chats if c.get("provider") and c.get("session_id")}


def mark_live_rows(tabs: list[tuple[str, list[tuple[str, str, str]], str]], chats: list[dict]):
    """Tag provider-tab rows whose session is open on the phone (picking one joins it)."""
    live = live_sessions(chats)
    return [
        (title, [(v, label, f"{desc} · live on phone" if v in live else desc) for v, label, desc in rows], empty)
        for title, rows, empty in tabs
    ]


def remote_key(chat: dict) -> str:
    return f"remote:{chat['chat']}"


def is_remote_key(key: str) -> bool:
    return (key or "").startswith("remote:")


def remote_from_key(key: str, chats: list[dict]) -> dict | None:
    """Resolve a picker value or a bare chat-id prefix against `chats`."""
    key = (key or "").strip()
    if key.startswith("remote:"):
        key = key[len("remote:"):]
    if not key:
        return None
    return next((c for c in chats if c["chat"] == key), None) or next((c for c in chats if c["chat"].startswith(key)), None)


def _age(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


# ---------------------------------------------------------------- the brain that isn't here
@dataclass
class RemoteAgentRun:
    """A subagent row rebuilt from the server's `agent` frames (what ui.agent_rows reads)."""

    id: str
    kind: str = "agent"
    label: str = ""
    done: bool = False
    failed: bool = False
    cut_off: bool = False
    model: str = ""
    effort: str = ""
    status: str = ""
    started: float = 0.0
    tokens_fn: Callable[[], int] | None = None
    steps: list[tuple[str, str]] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


class RemoteBrain:
    """The driver a chat screen uses while attached: the same face as Brain (provider, session_id,
    workdir, cfg, ask, cancel…) with the turns run by `vision serve`.

    One thread reads the socket. Frames about this chat during a turn are queued for `ask()`, which
    replays them into the screen's callbacks; between turns a `start` the phone caused is announced
    through `on_foreign_turn(text)` and the frames wait in the queue until the screen calls
    `ask(text)` for it (the message is not sent again: the turn is already running).
    """

    def __init__(self, server: dict, chat: dict, cfg, *, on_summary: Callable[[dict], None] | None = None,
                 on_foreign_turn: Callable[[str], None] | None = None, on_note: Callable[[str, str], None] | None = None,
                 on_answered: Callable[[], None] | None = None, on_closed: Callable[[str], None] | None = None):
        self.server = server
        self.chat_id = chat["chat"]
        self.state: dict = dict(chat)
        self.cfg = cfg  # the terminal's [brain] config: model/effort mirror the chat's (see _apply_summary)
        self.on_summary = on_summary or (lambda s: None)
        self.on_foreign_turn = on_foreign_turn or (lambda t: None)
        self.on_note = on_note or (lambda text, kind: None)
        self.on_answered = on_answered or (lambda: None)
        self.on_closed = on_closed or (lambda reason: None)
        self.output_tokens = 0
        self.last_usage = None
        self.handoff = None
        self.task_mode = False
        self.voice_mode = False
        self._ws = None
        self._frames: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._active = False  # ask() is running
        self._incoming: str | None = None  # a turn another client started, not yet picked up by ask()
        self._question_open = False
        self._answered_elsewhere = False
        self._hello = threading.Event()
        self._hello_error = ""
        self._closed = False  # the socket is gone (either side)
        self._left = False  # close() was called: the screen is detaching, no on_closed for it
        self._finished = False
        self._agents: dict[str, RemoteAgentRun] = {}
        self._tools: dict[str, ToolCall] = {}
        self._apply_summary(chat)

    # -- what the screen reads
    provider = property(lambda self: self.state.get("provider", ""))
    session_id = property(lambda self: self.state.get("session_id") or None)
    workdir = property(lambda self: self.state.get("workdir", ""))
    title = property(lambda self: self.state.get("title", ""))
    busy = property(lambda self: bool(self.state.get("busy")))

    @property
    def context(self) -> tuple[int, int] | None:
        ctx = self.state.get("context")
        if not ctx or not ctx.get("window"):
            return None
        return int(ctx.get("tokens") or 0), int(ctx["window"])

    def resolved_model(self) -> str | None:
        return self.state.get("model_id") or self.cfg.model

    # -- lifecycle
    def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Dial the server and wait for its hello; raises if the chat is not there any more."""
        threading.Thread(target=self._run, daemon=True, name="remote-chat").start()
        if not self._hello.wait(timeout):
            self.close()
            raise ConnectionError("no answer from vision serve")
        if self._hello_error:
            raise ConnectionError(self._hello_error)

    def pending_turn(self) -> str | None:
        """The turn already running when we joined (its frames are queued for ask()), if any."""
        with self._lock:
            return self._incoming

    def close(self) -> None:
        """Leave the chat (it keeps running on the server)."""
        self._left = True
        self._drop_socket()

    def _drop_socket(self) -> None:
        with self._lock:
            ws, self._ws = self._ws, None
            self._closed = True
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        from websockets.sync.client import connect

        reason = "the server closed the connection"
        url = self.server["url"].replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/ws?client=terminal"
        try:
            with connect(url, additional_headers={"Authorization": f"Bearer {self.server['token']}"}, open_timeout=CONNECT_TIMEOUT) as ws:
                with self._lock:
                    if self._closed:
                        return
                    self._ws = ws
                for raw in ws:
                    try:
                        ev = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(ev, dict):
                        self._dispatch(ev)
        except Exception as e:  # noqa: BLE001
            if not self._left:
                reason = f"lost vision serve: {e}"
        finally:
            self._finish(reason)

    def _finish(self, reason: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._closed = True
            self._ws = None
        if not self._hello.is_set():
            self._hello_error = self._hello_error or reason
            self._hello.set()
            return
        self._frames.put({"type": "closed", "text": reason})  # ends a running ask()
        if not self._left:
            self.on_closed(reason)

    def send(self, frame: dict) -> None:
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        frame.setdefault("chat", self.chat_id)
        try:
            ws.send(json.dumps(frame))
        except Exception:  # noqa: BLE001
            pass

    # -- incoming frames (socket thread)
    def _dispatch(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "hello":
            mine = next((c for c in ev.get("chats", []) if c.get("chat") == self.chat_id), None)
            if mine is None:
                self._hello_error = "that chat is gone"
                self._hello.set()
                self._drop_socket()
                return
            self._apply_summary(mine)
            if mine.get("busy") and mine.get("user_text") and not self._active:
                # Joined mid-reply: the turn waits as a pending one, with what has streamed so far,
                # for the screen to pick up once connect() returns (see pending_turn).
                self._incoming = mine["user_text"]
                self._frames.put({"type": "start", "text": mine["user_text"]})
                if mine.get("partial"):
                    self._frames.put({"type": "delta", "text": mine["partial"]})
            self._hello.set()
            return
        if kind == "pong":
            return
        if kind == "chat_closed":
            if ev.get("chat") == self.chat_id:
                self._finish("closed on the phone")
                self._drop_socket()
            return
        if kind == "error" and "chat" not in ev:
            # Server replies to this socket's own frame (that chat is gone, still replying…).
            if self._active:
                self._frames.put(ev)
            else:
                self.on_note(ev.get("text", "error"), "error")
            return
        if ev.get("chat") != self.chat_id:
            return
        if kind == "chat":
            self._apply_summary(ev)
            self.on_summary(ev)
            if self._question_open and not ev.get("waiting"):
                self._question_open = False
                self._answered_elsewhere = True
                self.on_answered()  # the phone answered: drop the form here
            return
        if kind in ("audio", "audio_end"):
            return  # the phone's speech; the terminal has its own
        if self._active or self._incoming is not None:
            self._frames.put(ev)
            return
        if kind == "start":
            self._incoming = ev.get("text", "")
            self._frames.put(ev)
            self.on_foreign_turn(self._incoming)
        elif kind in ("note", "error"):
            self.on_note(ev.get("text", ""), "error" if kind == "error" else "info")
        # a stray delta/done with no turn to attach to: nothing to show it under

    def _apply_summary(self, s: dict) -> None:
        self.state.update({k: v for k, v in s.items() if k not in ("type", "opened")})
        if s.get("model_id"):
            self.cfg.model = s["model_id"]
        if "effort" in s:
            self.cfg.effort = s.get("effort") or ""
        if s.get("workdir"):
            self.cfg.workdir = s["workdir"]

    # -- the turn (screen's worker thread)
    def ask(self, text: str, *, on_text=None, on_status=None, on_question=None, on_agent=None, on_tool=None, **_) -> Turn:
        with self._lock:
            self._active = True
            following = self._incoming == text
            self._incoming = None
            closed = self._closed
        self.output_tokens = 0
        self._agents.clear()
        self._tools.clear()
        turn = Turn(session_id=self.session_id)
        if closed:
            self._active = False
            turn.is_error, turn.error = True, "not attached to vision serve any more"
            return turn
        if not following:
            self.send({"type": "message", "text": text, "speak": False, "voice": False})
        started = time.time()
        try:
            while True:
                ev = self._frames.get()
                kind = ev.get("type")
                if kind == "start":
                    continue
                if kind == "delta":
                    d = ev.get("text", "")
                    turn.text += d
                    self.output_tokens += max(1, len(d) // 4)
                    if on_text:
                        on_text(d)
                elif kind == "status":
                    if on_status:
                        on_status(ev.get("tool", ""))
                elif kind == "agent":
                    run = self._agent(ev)
                    if on_agent:
                        on_agent(run)
                elif kind == "tool":
                    call = self._tool(ev)
                    if on_tool:
                        on_tool(call)
                elif kind == "question":
                    self._answer(ev.get("questions") or [], on_question)
                elif kind == "note":
                    self.on_note(ev.get("text", ""), "info")
                elif kind == "error":
                    turn.is_error, turn.error = True, ev.get("text", "error")
                    break
                elif kind == "closed":
                    turn.is_error, turn.error = True, ev.get("text", "lost vision serve")
                    break
                elif kind == "done":
                    turn.text = ev.get("text") or turn.text
                    turn.error = ev.get("error") or ""
                    turn.is_error = bool(turn.error)
                    turn.session_id = ev.get("session_id") or turn.session_id
                    turn.model = ev.get("model")
                    turn.duration_ms = ev.get("duration_ms") or int((time.time() - started) * 1000)
                    break
        finally:
            with self._lock:
                self._active = False
            turn.agents = list(self._agents.values())
            turn.tools = list(self._tools.values())
        return turn

    def _answer(self, questions: list[dict], on_question) -> None:
        self._question_open, self._answered_elsewhere = True, False
        answers = None
        try:
            answers = on_question(questions) if on_question else None
        finally:
            was_open, self._question_open = self._question_open, False
        if was_open and not self._answered_elsewhere:
            self.send({"type": "answer", "answers": answers if isinstance(answers, dict) and answers else None})

    def _agent(self, ev: dict) -> RemoteAgentRun:
        run = self._agents.get(ev.get("id", ""))
        if run is None:
            run = RemoteAgentRun(id=ev.get("id", ""), started=time.monotonic())
            self._agents[run.id] = run
        run.kind = ev.get("kind") or run.kind
        run.label = ev.get("label") or run.label
        run.model = ev.get("model") or run.model
        run.effort = ev.get("effort") or run.effort
        step = ev.get("step")
        if step and (not run.steps or run.steps[-1] != (step.get("tool", ""), step.get("detail", ""))):
            run.steps.append((step.get("tool", ""), step.get("detail", "")))
        if ev.get("done"):
            run.done = True
            run.failed = bool(ev.get("failed"))
            run.cut_off = bool(ev.get("cut_off"))
            run.status = ev.get("status") or run.status
            run.details = list(ev.get("details") or [])
        return run

    def _tool(self, ev: dict) -> ToolCall:
        call = self._tools.get(ev.get("id", ""))
        if call is None:
            call = ToolCall(id=ev.get("id", ""), name=ev.get("name", ""))
            self._tools[call.id] = call
        call.name = ev.get("name") or call.name
        call.detail = ev.get("detail") or call.detail
        call.done = bool(ev.get("done"))
        call.is_error = bool(ev.get("is_error"))
        if "output" in ev:
            call.output = ev.get("output") or ""
        return call

    # -- controls
    def cancel(self) -> None:
        if self._active or self._question_open:
            self.send({"type": "cancel"})

    def switch_model(self, model: str, effort: str | None) -> None:
        self.send({"type": "model", "model": model, "effort": effort or None})

    def new_session(self) -> None:  # the screen detaches before it starts a new conversation
        pass

    def resume(self, session_id: str) -> None:
        pass

    # -- /usage: the chat's spending is the server's; the provider's own report still works here
    def usage_report(self):
        return None

    @staticmethod
    def cached_usage():
        from vision.brain import Brain

        return Brain.cached_usage()

    def usage_renderable(self, full: bool = False):
        from rich.text import Text

        return Text("Usage isn't tracked for a phone chat here; try /usage <provider>.", style="yellow")

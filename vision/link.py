"""Live link: a terminal chat announces itself so `vision serve` can list it on the phone and drive it.

Each `vision` chat process (LinkHost) listens on a Unix socket under ~/.local/state/vision/live/ and
drops a small JSON descriptor beside it. The server (Hub._watch_links in server.py) scans that
directory, connects to every socket it finds and shows each one as a chat marked `terminal`. The
terminal process stays the only owner of the agent session: a message from the phone is typed into
it, its reply streams back over the socket, and whatever is typed at the keyboard streams to the
phone too. When the terminal quits, its socket goes and the chat leaves the phone's list.

Wire format: one JSON object per line, both ways.
  server → terminal   hello · message {text, speak} · answer {answers} · cancel · model {model, effort} · quit
  terminal → server   chat {…summary…} on hello and whenever the state changes, then the same
                      frames a server-owned chat posts (start, delta, status, agent, question, done, note)
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable

from vision import compat
from vision.config import STATE_DIR

LIVE_DIR = STATE_DIR / "live"
# Unix sockets: CPython has no AF_UNIX on Windows, so there a chat is simply not linked to `vision serve`.
SUPPORTED = hasattr(socket, "AF_UNIX")


def descriptor_path(pid: int) -> Path:
    return LIVE_DIR / f"{pid}.json"


def socket_path(pid: int) -> Path:
    return LIVE_DIR / f"{pid}.sock"


def pid_alive(pid: int) -> bool:
    return compat.pid_alive(pid)


def list_links() -> list[dict]:
    """Every terminal chat that is announcing itself, stale descriptors (dead pids) swept away."""
    try:
        files = sorted(LIVE_DIR.glob("*.json"))
    except OSError:
        return []
    out = []
    for f in files:
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
            pid = int(info["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            _sweep(f)
            continue
        if not pid_alive(pid):
            _sweep(f)
            continue
        info["sock"] = str(socket_path(pid))
        out.append(info)
    return out


def _sweep(desc: Path) -> None:
    for p in (desc, desc.with_suffix(".sock")):
        try:
            p.unlink()
        except OSError:
            pass


class LinkHost:
    """The terminal side. `summary()` describes the chat as the phone should see it; `on_frame`
    gets every frame the server sends (message, answer, cancel, model). `post()` fans an event
    out to every connected server. The watch thread posts a fresh summary whenever it changes,
    so /model, /new, /resume and the title settling all reach the phone without extra plumbing."""

    WATCH = 1.0

    def __init__(self, summary: Callable[[], dict], on_frame: Callable[[dict], None], log: Callable[[str], None] | None = None):
        self.summary = summary
        self.on_frame = on_frame
        self.log = log or (lambda _msg: None)
        self.pid = os.getpid()
        self._server: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last: dict | None = None

    # -- lifecycle
    def start(self) -> None:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(LIVE_DIR, 0o700)
        except OSError:
            pass
        sock_path = socket_path(self.pid)
        try:
            sock_path.unlink()
        except OSError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        os.chmod(sock_path, 0o600)
        srv.listen(4)
        self._server = srv
        self._write_descriptor()
        threading.Thread(target=self._accept, daemon=True, name="link-accept").start()
        threading.Thread(target=self._watch, daemon=True, name="link-watch").start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            clients, self._clients = self._clients, []
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        _sweep(descriptor_path(self.pid))

    @property
    def connected(self) -> bool:
        with self._lock:
            return bool(self._clients)

    def _write_descriptor(self) -> None:
        info = {"pid": self.pid, "started": time.time(), "cwd": os.getcwd()}
        tmp = descriptor_path(self.pid).with_suffix(".tmp")
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, descriptor_path(self.pid))

    # -- outgoing
    def post(self, ev: dict) -> None:
        with self._lock:
            if not self._clients:
                return
            data = (json.dumps(ev) + "\n").encode()
            dead = []
            for c in self._clients:
                try:
                    c.sendall(data)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def post_summary(self, force: bool = False) -> None:
        try:
            s = self.summary()
        except Exception:  # noqa: BLE001
            return
        if force or s != self._last:
            self._last = s
            self.post({"type": "chat", **s})

    # -- threads
    def _accept(self) -> None:
        assert self._server
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with self._lock:
                self._clients.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="link-client").start()

    def _serve(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except ValueError:
                        continue
                    if frame.get("type") == "hello":
                        with self._lock:
                            try:
                                conn.sendall((json.dumps({"type": "chat", **self.summary()}) + "\n").encode())
                            except OSError:
                                return
                        continue
                    try:
                        self.on_frame(frame)
                    except Exception as e:  # noqa: BLE001
                        self.log(f"link: {e}")
        except OSError:
            pass
        finally:
            with self._lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _watch(self) -> None:
        while not self._stop.wait(self.WATCH):
            if self.connected:
                self.post_summary()


class LinkClient:
    """The server side of one terminal chat: a blocking socket read on a thread, frames handed to
    `on_event`; `send()` from any thread. `on_close` fires once when the terminal goes away."""

    def __init__(self, info: dict, on_event: Callable[[dict], None], on_close: Callable[[], None]):
        self.info = info
        self.pid = int(info["pid"])
        self.on_event = on_event
        self.on_close = on_close
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._closed = False

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(self.info["sock"])
        s.settimeout(None)
        self._sock = s
        threading.Thread(target=self._read, daemon=True, name=f"link-{self.pid}").start()
        self.send({"type": "hello"})

    def send(self, frame: dict) -> None:
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall((json.dumps(frame) + "\n").encode())
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
        self._finish()

    def _finish(self) -> None:
        if not self._closed:
            self._closed = True
            self.on_close()

    def _read(self) -> None:
        s = self._sock
        buf = b""
        try:
            while s is not None:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    self.on_event(ev)
        except OSError:
            pass
        finally:
            self.close()

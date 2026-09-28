"""Remote access: `vision serve` puts the brain, ears and voice behind HTTP + WebSocket for the iOS app.

One user, several chats. Each chat is an independent conversation with its own agent session, model
and transcript, and chats run at the same time (like Claude Code sessions in the Claude app); the
ears and the voice are shared. The server listens on the LAN by default so the phone can reach it
over the same Wi-Fi; Tailscale Serve takes it to the user's other devices anywhere (a public tunnel,
Tailscale Funnel or Cloudflare Tunnel, to the whole internet). Every
request carries the bearer token from ~/.config/vision/remote_token; the QR code printed at start-up
carries the URL and the token so the app can pair with one scan.

WebSocket protocol (JSON, one object per frame). Every frame about a chat carries `chat` (its id);
a phone → laptop frame without one goes to the most recently active chat.
  phone → laptop   message {chat, text, speak, voice, images?: [path…]} (photos, videos and files from /upload) · answer {chat, answers} · cancel {chat}
                   new · close {chat} · resume {session_id, provider?} · model {chat, model, effort} · ping
                   open_terminal {chat} (a terminal window here running `vision --join <chat>`; one per chat:
                     a chat that already has one gets it brought forward, on Hyprland, and a toast)
  laptop → phone   hello {version, workdir, chats: [chat…]}
                   chat {chat, title, model, effort, session_id, busy, waiting, questions, partial, user_text, …}
                     (sent whenever a chat's state changes; `opened: true` on the reply to new/resume)
                   chat_closed {chat, moved_to?} · start {text, speak} · delta {text} · status {tool}
                   agent {id, kind, label, done, tools, step?, status?}
                   audio {seq, wav} (base64 WAV per sentence) · audio_end
                   question {questions} · done {text, error, session_id, duration_ms} · note {text}
                   error {text} · toast {text} (to the asker only; a scheduled run toasts everyone) · pong
                   scheduled (the task list changed or a task ran: refetch GET /scheduled)
Events are broadcast to every connected client, so a phone that reconnects mid-reply picks it up.
Message frames received while a chat is busy are queued on that chat and run in arrival order.

Chats carry `source`: "server" for one this process runs, "terminal" for a `vision` chat open in a
terminal on this machine (link.py). Terminal chats are found on disk, driven over their socket
and leave the list when the terminal quits; `close` quits that terminal (as /quit typed there would).
Closing the last chat leaves none open; the phone's next message (or +) opens one.

The other way round, a terminal `vision` can join one of this process's chats (remote.py): it dials
the same WebSocket (`/ws?client=terminal&chat=<id>&pid=<its pid>`) and follows the frames like a
phone; the chat and pid say which chat already has a window here. While running, this process leaves ~/.local/state/vision/serve.json so a terminal on this machine finds it.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path

# Module-level on purpose: with `from __future__ import annotations` FastAPI resolves the endpoint
# type hints by name, so Request/WebSocket/UploadFile have to be importable from this module's globals.
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from vision import __version__
from vision import agentlog
from vision.brain import agent_frame, context_figure, tool_frame
from vision.config import CONFIG_DIR, STATE_DIR, Config, save_brain_defaults, saved_brain_defaults
from vision.sessions import claude_history
from vision.reply import status_label
from vision.tts import SAMPLE_RATE, Speaker, StreamingSpeaker

UPLOAD_DIR = Path.home() / ".local" / "share" / "vision" / "uploads"


def open_terminal(chat_id: str, workdir: str) -> str:
    """Open a terminal window on this machine with `vision --join <chat>` in it (the phone's
    "Open on laptop"). Needs a desktop session: the server must run inside one for a window to
    appear. Returns the terminal used; raises RuntimeError when none is found."""
    vision = shutil.which("vision") or "vision"
    cmd = f"{vision} --join {chat_id}"
    cwd = workdir if workdir and os.path.isdir(workdir) else os.path.expanduser("~")
    env = os.environ
    if sys.platform == "win32":
        return _open_terminal_windows(vision, chat_id, cwd)
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise RuntimeError("vision serve is not running in a desktop session, so it cannot open a window")
    candidates = [
        (env.get("TERMINAL"), lambda t: [t, "-e", "sh", "-c", cmd]),
        ("ptyxis", lambda t: [t, "--new-window", "-d", cwd, "-x", cmd]),
        ("gnome-terminal", lambda t: [t, f"--working-directory={cwd}", "--", "sh", "-c", cmd]),
        ("kgx", lambda t: [t, f"--working-directory={cwd}", "-e", cmd]),
        ("konsole", lambda t: [t, "--workdir", cwd, "-e", "sh", "-c", cmd]),
        ("kitty", lambda t: [t, "-d", cwd, "sh", "-c", cmd]),
        ("alacritty", lambda t: [t, "--working-directory", cwd, "-e", "sh", "-c", cmd]),
        ("foot", lambda t: [t, "-D", cwd, "sh", "-c", cmd]),
        ("xterm", lambda t: [t, "-e", "sh", "-c", cmd]),
    ]
    for name, argv in candidates:
        if not name or not (path := shutil.which(name)):
            continue
        subprocess.Popen(argv(path), cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return os.path.basename(name)
    raise RuntimeError("no terminal emulator found (set $TERMINAL)")


def _parent_pid(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(stat.rsplit(")", 1)[1].split()[1])  # the name in (…) can hold spaces; ppid follows the state
    except (OSError, ValueError, IndexError):
        return 0


def focus_terminal(pid: int) -> bool:
    """Bring forward the window a `vision` process (`pid`) runs in: "Open on laptop" for a chat that
    already has one. Hyprland only, whose hyprctl gives each window's pid: the terminal emulator's, an
    ancestor of `vision` (through `sh -c`). Elsewhere, or when the window can't be told apart, nothing."""
    exe = shutil.which("hyprctl")
    if not exe or not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return False
    try:
        r = subprocess.run([exe, "clients", "-j"], capture_output=True, text=True, timeout=3, encoding="utf-8")
        clients = json.loads(r.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    by_pid: dict[int, list[dict]] = {}
    for c in clients if isinstance(clients, list) else []:
        by_pid.setdefault(c.get("pid"), []).append(c)
    while pid > 1 and pid not in by_pid:
        pid = _parent_pid(pid)
    windows = by_pid.get(pid, [])
    if len(windows) > 1:  # one process drawing several windows (a single-instance terminal): Vision's own title
        from vision.ui import TITLE_SPINNER

        titles = [(c.get("title") or "").lstrip(TITLE_SPINNER + " ") for c in windows]
        windows = [c for c, t in zip(windows, titles) if t == "Vision" or t.startswith("Vision - ")]
    if len(windows) != 1:
        return False
    try:
        r = subprocess.run([exe, "dispatch", "focuswindow", f"address:{windows[0]['address']}"],
                           capture_output=True, text=True, timeout=3, encoding="utf-8")
    except (OSError, KeyError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() == "ok"


def _open_terminal_windows(vision: str, chat_id: str, cwd: str) -> str:
    """Windows Terminal when it is installed, else a console window of its own. The chat id is the
    server's own hex, so nothing from the phone reaches the command line."""
    wt = shutil.which("wt")
    if wt:
        subprocess.Popen([wt, "-d", cwd, vision, "--join", chat_id], cwd=cwd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "wt"
    subprocess.Popen([vision, "--join", chat_id], cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return "console"

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".heic": "image/heic", ".webp": "image/webp", ".gif": "image/gif"}
VIDEO_TYPES = {".mp4", ".mov", ".m4v"}
MAX_FILE_BYTES = 100 * 1024 * 1024   # any other file from the phone's Files picker; brains open it with their tools
MAX_VIDEO_FRAMES = 24     # what a brain is handed for one clip, however long
SCENE_THRESHOLD = 0.04    # ffmpeg scene score that counts as "the screen changed" (screen recordings change little)


def with_attachments(text: str, images) -> str:
    """Append the uploaded photos' paths so the brain can open them with its file tools. A video is
    handed over as its frames (brains can read images, not video) plus what was said in it."""
    paths = [p for p in (images or []) if isinstance(p, str) and p.startswith(str(UPLOAD_DIR)) and os.path.isfile(p)]
    if not paths:
        return text
    lines: list[str] = []
    videos = images = files = False
    for p in paths:
        suffix = Path(p).suffix.lower()
        if suffix not in IMAGE_TYPES and suffix not in VIDEO_TYPES:
            files = True
            lines.append(f"[Attached file: {p}]")
            continue
        digest = video_digest(p)
        if digest is None:
            images = True
            lines.append(f"[Attached image: {p}]")
            continue
        videos = True
        frames = digest.get("frames") or []
        lines.append(f"[Attached video: {p}]")
        lines.append(f"[Video length {clock(digest.get('duration', 0))}; {len(frames)} frames in time order, open them to watch it]")
        lines.extend(f"[Frame {clock(t)}: {f}]" for t, f in frames)
        if said := (digest.get("transcript") or "").strip():
            lines.append(f"[Video audio, transcribed: {' '.join(said.split())}]")
    note = "\n".join(lines)
    nudge = "\n".join(line for wanted, line in [
        (images, "Look at the attached image(s)."),
        (videos, "Watch the attached video(s) through their frames."),
        (files, "Read the attached file(s)."),
    ] if wanted)
    return f"{text}\n\n{note}" if text else f"{note}\n\n{nudge}"


def image_path_allowed(path: Path) -> bool:
    """Reply pictures come from the home folder or /tmp (Windows: the temp folder), never the rest of the system."""
    tmp = Path(tempfile.gettempdir()) if sys.platform == "win32" else Path("/tmp")
    return any(path.is_relative_to(root) for root in (Path.home().resolve(), tmp.resolve()))


def clock(seconds: float) -> str:
    s = int(round(seconds or 0))
    return f"{s // 60}:{s % 60:02d}"


def video_digest(path: str | Path) -> dict | None:
    """What digest_video stored for an uploaded clip, or None when the path isn't a digested video."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_TYPES:
        return None
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def digest_video(path: Path, transcribe=None) -> dict:
    """Turn an uploaded clip into what a brain can take in: JPEG frames where the picture changes
    (plus evenly spaced ones so a static stretch isn't one frame), capped at MAX_VIDEO_FRAMES, and the
    soundtrack through Whisper. Stored beside the clip as <clip>.json; frames in <clip>-frames/."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg is not installed on the laptop")
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, encoding="utf-8").stdout)
    duration = float(probe.get("format", {}).get("duration") or 0)
    kinds = {s.get("codec_type") for s in probe.get("streams", [])}
    if "video" not in kinds:
        raise RuntimeError("no video track in that file")
    frames_dir = path.with_name(path.stem + "-frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    fit = "scale='if(gt(iw,ih),min(1568,iw),-2)':'if(gt(iw,ih),-2,min(1568,ih))'"  # long edge ≤ 1568 px, as photos

    # Scene changes (and the first frame); showinfo logs each kept frame's time.
    run = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(path),
         "-vf", f"select='eq(n,0)+gt(scene,{SCENE_THRESHOLD})',showinfo,{fit}", "-fps_mode", "vfr", "-q:v", "3",
         str(frames_dir / "s%04d.jpg")],
        capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read the video: {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else run.returncode}")
    times = [float(line.split("pts_time:")[1].split()[0]) for line in run.stderr.splitlines() if "pts_time:" in line]
    scene = sorted(zip(times, sorted(frames_dir.glob("s*.jpg"))))

    # Evenly spaced frames fill the gaps a scene pass leaves (one every ~4 s, up to 8).
    even = []
    count = min(8, int(duration // 4))
    for k in range(1, count):
        t = duration * k / count
        if any(abs(t - s) < 1.5 for s, _ in scene):
            continue
        out = frames_dir / f"e{k:04d}.jpg"
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                        "-frames:v", "1", "-vf", fit, "-q:v", "3", str(out)], capture_output=True)
        if out.is_file():
            even.append((t, out))

    frames = sorted(scene + even)
    if len(frames) > MAX_VIDEO_FRAMES:  # keep the first and last, spread the rest evenly
        step = (len(frames) - 1) / (MAX_VIDEO_FRAMES - 1)
        keep = {round(i * step) for i in range(MAX_VIDEO_FRAMES)}
        frames = [f for i, f in enumerate(frames) if i in keep]
    kept = set()
    named = []
    for i, (t, f) in enumerate(frames):
        final = frames_dir / f"{i:02d}.jpg"
        f.rename(final)
        kept.add(final)
        named.append((round(t, 2), str(final)))
    for stray in frames_dir.glob("*.jpg"):
        if stray not in kept:
            stray.unlink()

    transcript = ""
    if "audio" in kinds and transcribe is not None:
        wav = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "wav", "pipe:1"], capture_output=True).stdout
        if wav:
            transcript = (transcribe(wav) or "").strip()
    digest = {"duration": round(duration, 2), "frames": named, "transcript": transcript}
    path.with_suffix(".json").write_text(json.dumps(digest), encoding="utf-8")
    return digest

TOKEN_FILE = CONFIG_DIR / "remote_token"
QUESTION_TIMEOUT_S = 900  # how long a turn waits for the phone to answer an AskUserQuestion form


# ---------------------------------------------------------------- token
def load_token(regenerate: bool = False) -> str:
    """The shared secret the app must present. Created on first use, 0600, rotated with --new-token."""
    if not regenerate:
        try:
            tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if tok:
                return tok
        except OSError:
            pass
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(tok + "\n", encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    return tok


# ---------------------------------------------------------------- audio helpers
def wav_bytes(audio, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class WireSpeaker(StreamingSpeaker):
    """StreamingSpeaker that ships synthesised audio to a sink instead of the sound card."""

    MIN_PIECE_S = 1.0  # fewer, fatter WAV messages over the wire

    def __init__(self, speaker: Speaker, sink):
        self.sink = sink
        self.seq = 0
        super().__init__(speaker)

    def _play_loop(self) -> None:  # runs on its own thread, in sentence order
        while True:
            item = self._audio_q.get()
            if item is None:
                return
            audio, _raw, _last = item
            if audio is None or self.speaker._stop.is_set():
                continue
            self.seq += 1
            self.sink(self.seq, wav_bytes(audio))


# ---------------------------------------------------------------- chats
class Chat:
    """One independent conversation: its own brain (agent session), model, transcript and turn state.

    Several chats run at once, like Claude Code sessions in the Claude app: each owns a copy of the
    config (so /model on one never touches another) and drives its own agent process.
    """

    def __init__(self, hub: "Hub", brain, cfg: Config, title: str = "", chat_id: str | None = None,
                 restored: dict | None = None):
        from vision.conversation import VoiceConversation

        self.hub = hub
        self.id = chat_id or secrets.token_hex(6)
        self.cfg = cfg
        self.brain = brain
        self.conversation = VoiceConversation(cfg, agent=brain)
        self.title = title
        self.created = time.time()
        self.updated = self.created
        self.busy = False
        self._pending: deque[tuple[str, bool, bool, bool]] = deque()
        self._turn_task: asyncio.Task | None = None
        self.partial = ""
        self.user_text = ""
        self._question: Future | None = None
        self._questions: list[dict] = []
        self._wire: WireSpeaker | None = None
        self._lock = threading.Lock()
        # A remote transcript includes typed and spoken turns, never the private worker's JSON.
        # Kept for the lifetime of the chat so reconnecting phones recover voice-only turns too.
        self._transcript: list[dict] | None = None
        self.history_id = secrets.token_hex(12)
        # This turn's subagent rows as agent frames (with `at`, where in the reply each started): in the
        # summary while busy, so a phone that (re)connects mid-turn rebuilds them with their real timers.
        self.agents: dict[str, dict] = {}
        self.tools: dict[str, dict] = {}  # …and its tool calls as tool frames, the same way
        self._steers: list[tuple] = []  # (reply chars so far, text, agent rows so far) of messages sent into this turn
        self._driver = None  # what answers this turn (the brain, or the conversation model on a call)
        self.journal = None
        self._journal_has_base = bool(restored and restored["has_base"])
        if hub.journal_enabled:
            from vision.turnjournal import ChatJournal

            self.journal = ChatJournal(self.id)
            if not self.journal.claim():
                raise RuntimeError(f"chat {self.id} is already open in another Vision process")
            if restored is None:
                self.journal.append("chat", title=title, model=cfg.brain.model, effort=cfg.brain.effort,
                                    session_id=brain.session_id)
        if restored is not None:
            self._transcript = restored["history"] if restored["has_base"] or restored["history"] else None
            self.created, self.updated = restored["created"], restored["updated"]
            if not self.title:
                self.title = next((m["text"][:48] for m in self._transcript or []
                                   if m.get("role") == "user" and m.get("text")), "")
            # A first turn can crash before the provider gives Vision a session id. Carry the
            # recovered transcript into the next fresh provider thread.
            if not brain.session_id and self._transcript:
                from vision.turnjournal import recovery_context

                brain.handoff = recovery_context(self._transcript, restored["recovery_rows"])

    # -- the bits of the brain the hub reads (LinkedChat has the same, from the terminal's summary)
    source = "server"

    @property
    def provider(self) -> str:
        return self.brain.provider

    @property
    def session_id(self) -> str | None:
        return self.brain.session_id

    @property
    def workdir(self) -> str:
        return self.brain.workdir

    @property
    def model_id(self) -> str:
        return self.brain.cfg.model

    @property
    def effort(self) -> str:
        return self.brain.cfg.effort or ""

    # -- snapshot for the phone
    def summary(self) -> dict:
        from vision.models import model_label

        return {
            "type": "chat",
            "chat": self.id,
            "source": self.source,
            "title": self.title,
            "provider": self.brain.provider,
            "model": model_label(self.brain.resolved_model()) or self.brain.cfg.model,
            "model_id": self.brain.cfg.model,
            "effort": self.brain.cfg.effort or "",
            # spoken turns are answered by the conversation model, not the brain above
            "voice_model": model_label(self.cfg.conversation.model) or self.cfg.conversation.model,
            "voice_model_id": self.cfg.conversation.model,
            "voice_effort": self.cfg.conversation.effort or "",
            "session_id": self.brain.session_id,
            "context": context_figure(self.brain),
            "history_id": self.history_id,
            "busy": self.busy,
            "waiting": self._question is not None and not self._question.done(),
            "questions": self._questions if self._question is not None and not self._question.done() else [],
            "partial": self.partial if self.busy else "",
            "user_text": self.user_text if self.busy else "",
            "queued": [item[0] for item in self._pending],  # a phone opened cold gets its queued bubbles back
            "agents": list(self.agents.values()) if self.busy else [],
            "tools": list(self.tools.values()) if self.busy else [],
            "created": self.created,
            "updated": self.updated,
        }

    def post(self, ev: dict) -> None:
        ev["chat"] = self.id
        self.hub.post(ev)

    def log(self, msg: str) -> None:
        self.hub.log(f"[{self.id[:4]}] {msg}" if len(self.hub.chats) > 1 else msg)

    # -- transcript
    def reset_history(self) -> None:
        self._transcript = None
        self.history_id = secrets.token_hex(12)

    def history(self, limit: int = 60) -> list[dict]:
        from vision.sessions import session_history

        with self._lock:
            if self._transcript is None:
                self._transcript = session_history(self.brain.provider, self.brain.session_id or "")
            if not self.title:
                first = next((m["text"] for m in self._transcript if m.get("role") == "user" and m.get("text")), "")
                self.title = _title_from(first)
            return list(self._transcript[-max(1, min(limit, 200)):])

    # -- turn control
    def queue(self, text: str, speak: bool, voice: bool, talk: bool = False) -> None:
        """`voice`: spoken; `talk`: typed during a call. Both go to the conversation model, so one
        model answers the whole call."""
        if self.journal:
            self.journal.append("queued", text=text)
        if self.busy or self.hub.restarting:
            # A restart already under way: the journal carries it over and the new process runs it.
            self._pending.append((text, speak, voice, talk))
            return
        self.busy = True  # reserve the turn before yielding to a scheduled task
        self._turn_task = asyncio.create_task(self.run_turn(text, speak=speak, voice=voice, talk=talk))

    def resume_queued(self, texts: list[str]) -> None:
        """Run messages that were queued when a graceful restart took the last process down (their
        `queued` records are already in the journal, so they are not journalled again)."""
        self._pending.extend((text, False, False, False) for text in texts if text)
        if not self.busy and self._pending:
            text, speak, voice, talk = self._pending.popleft()
            self.busy = True
            self._turn_task = asyncio.create_task(self.run_turn(text, speak, voice, talk))

    def answer(self, answers) -> None:
        fut = self._question
        if fut is not None and not fut.done():
            fut.set_result(answers if isinstance(answers, dict) and answers else None)

    def unqueue(self, text: str) -> bool:
        """Take a queued message back out (the first one reading `text`); False if it already ran."""
        for i, item in enumerate(self._pending):
            if item[0] == text:
                if self.journal:
                    self.journal.append("unqueued", text=text)
                del self._pending[i]
                return True
        return False

    def steer(self, text: str) -> bool:
        """Send `text` into the running turn now instead of after it (Claude takes it at its next step).
        A copy still waiting in the queue is taken out. False when this turn's brain can't take one
        (Codex, Grok, a local model, nothing running): the message stays queued."""
        steer = getattr(self._driver, "steer", None)
        if not self.busy or steer is None:
            return False
        if self.journal:
            self.journal.append("steer_attempt", text=text)
        if not steer(text):
            if self.journal:
                self.journal.append("steer_failed", text=text)
            return False
        if self.journal:
            self.journal.append("steered", text=text)
            self.journal.append("unqueued", text=text)
        for i, item in enumerate(self._pending):
            if item[0] == text:
                del self._pending[i]
                break
        self._steers.append((len(self.partial), text, len(self.agents)))
        self.log(f"» {text[:80]}{'…' if len(text) > 80 else ''}")
        self.post({"type": "steered", "text": text})
        return True

    def cancel(self) -> None:
        fut = self._question
        if fut is not None and not fut.done():
            fut.set_result(None)
        if self._wire:
            self._wire.stop()
        self.conversation.cancel()
        self.brain.cancel()

    async def switch_model(self, model: str, effort: str | None) -> None:
        from vision.cli import _effort_word, _model_label, _switch_model

        if not model:
            return
        from vision.models import provider_for
        from vision.providers import unavailable_reason

        why = unavailable_reason(provider_for(model), self.cfg)
        if why:
            self.post({"type": "error", "text": why})
            return
        model_changed = model != self.cfg.brain.model
        self.busy = True
        loop = asyncio.get_running_loop()
        with self._lock:
            history = list(self._transcript or [])
        try:
            self.brain, _detail = await loop.run_in_executor(
                None, lambda: _switch_model(self.cfg, self.brain, model, voice_mode=False, effort=effort or None,
                                            history=history)
            )
            self.conversation.agent = self.brain
            if self.journal:
                self.journal.append("model", model=self.cfg.brain.model, effort=self.cfg.brain.effort,
                                    session_id=self.brain.session_id)
            self.conversation.follow()  # a call on this chat talks through the new pick
            # One short line; the transcript-handoff detail is terminal chatter the phone doesn't need.
            effort_word = _effort_word(self.cfg.brain.effort)
            if model_changed:
                text = f"Switched to {_model_label(self.brain)} ({effort_word})"
            else:
                text = f"Effort set to {effort_word}"
            self.post({"type": "note", "text": text})
        except Exception as e:  # noqa: BLE001
            self.post({"type": "error", "text": f"model switch failed: {e}"})
        finally:
            self.busy = False
            self.hub.post(self.summary())
            # A message may have arrived during the handoff. It belongs after the switch, on the new brain.
            if self._pending and not self.hub.restarting:
                text, speak, voice, talk = self._pending.popleft()
                self.busy = True
                self._turn_task = asyncio.create_task(self.run_turn(text, speak, voice, talk))
            self.hub.maybe_restart()

    async def run_turn(self, text: str, speak: bool, voice: bool, talk: bool = False) -> None:
        """Run this turn and every follow-up queued while it is active, in arrival order."""
        try:
            while True:
                await self._run_one_turn(text, speak, voice, talk)
                if not self._pending or self.hub.restarting:
                    return
                text, speak, voice, talk = self._pending.popleft()
        finally:
            self.busy = False
            if self._turn_task is asyncio.current_task():
                self._turn_task = None
            self.hub.post(self.summary())
            self.hub.maybe_restart()

    async def _run_one_turn(self, text: str, speak: bool, voice: bool, talk: bool = False) -> None:
        loop = asyncio.get_running_loop()
        self.busy, self.partial, self.user_text = True, "", text
        self.agents, self.tools, self._steers = {}, {}, []
        self.updated = time.time()
        if not self.title:
            self.title = _title_from(text)
        await loop.run_in_executor(None, self.history)  # seed from the typed session before it advances
        if self.journal:
            if not self._journal_has_base:
                self.journal.append("base", history=self._transcript or [])
                self._journal_has_base = True
            self.journal.append("start", text=text)
        self.log(f"› {text[:80]}{'…' if len(text) > 80 else ''}")
        self.post({"type": "start", "text": text, "speak": speak})
        self.hub.post(self.summary())
        started = time.time()

        wire = None
        if speak:
            try:
                sp = await loop.run_in_executor(None, self.hub.speaker)
                wire = WireSpeaker(sp, sink=lambda seq, wav: self.post({"type": "audio", "seq": seq, "wav": base64.b64encode(wav).decode()}))
                if voice:
                    from vision.cli import _arm_filler

                    _arm_filler(wire, sp, self.hub.cfg.voice)  # "One sec." if the first words are late
            except Exception as e:  # noqa: BLE001
                self.post({"type": "note", "text": f"voice unavailable: {e}"})
        self._wire = wire

        def on_text(delta: str) -> None:
            if self.journal:
                self.journal.append("delta", text=delta)
            self.partial += delta
            self.post({"type": "delta", "text": delta})
            if wire:
                wire.feed(delta)

        def on_status(tool: str) -> None:
            if wire and tool:
                wire.flush()  # the text block is over: say its last sentence while the tool runs
            self.post({"type": "status", "tool": tool, "label": status_label(tool)})

        def on_agent(run) -> None:
            # One frame per change; `step` is the newest tool call (absent when it just started or finished).
            frame = agent_frame(run)
            frame["at"] = self.agents.get(run.id, {}).get("at", len(self.partial))
            if self.journal:
                from vision.turnjournal import compact_agent_frame

                self.journal.append("agent", frame=compact_agent_frame(frame, self.agents.get(run.id)))
            self.agents[run.id] = frame
            self.post(dict(frame))

        def on_tool(call) -> None:
            # The main conversation's tool calls, drawn as rows by the phone and a terminal following this chat.
            frame = tool_frame(call)
            frame["at"] = self.tools.get(call.id, {}).get("at", len(self.partial))
            if self.journal:
                self.journal.append("tool", frame=frame)
            self.tools[call.id] = frame
            self.post(dict(frame))

        def on_question(questions: list[dict]) -> dict[str, str] | None:
            fut: Future = Future()
            if self.journal:
                self.journal.append("question", questions=questions)
            self._questions = questions
            self._question = fut
            self.post({"type": "question", "questions": questions})
            self.hub.post(self.summary())
            answers = None
            try:
                answers = fut.result(timeout=QUESTION_TIMEOUT_S)
                return answers
            except Exception:  # noqa: BLE001  (timeout → the model carries on without answers)
                return None
            finally:
                if self.journal:
                    self.journal.append("answered", questions=questions, answers=answers)
                self._question = None
                self._questions = []
                self.post({"type": "answered", "questions": questions, "answers": answers})
                self.hub.post(self.summary())

        from vision.cli import _turn_brain

        driver = self._driver = _turn_brain(self.brain, self.conversation, voice, text, talk=talk)
        try:
            turn = await loop.run_in_executor(
                None, lambda: driver.ask(text, on_text=on_text, on_status=on_status, on_question=on_question, on_agent=on_agent, on_tool=on_tool)
            )
            error = turn.error if turn.is_error else ""
            reply, session_id, model = turn.text, turn.session_id, turn.model
        except Exception as e:  # noqa: BLE001
            error, reply, session_id, model = str(e), self.partial, self.brain.session_id, None
        if wire:
            try:
                await loop.run_in_executor(None, wire.finish)  # flush the last sentence
            except Exception:  # noqa: BLE001
                pass
            self.post({"type": "audio_end"})
        self._wire = None
        self._driver = None
        if self.journal:
            self.journal.append("done", text=reply, error=error,
                                session_id=session_id or self.brain.session_id)
        entries = _turn_entries(text, reply, error, self.partial, list(self.agents.values()), self._steers, list(self.tools.values()))
        with self._lock:
            if self._transcript is None:
                self._transcript = []
            self._transcript.extend(entries)
            self._transcript = self._transcript[-200:]
        _record_agents(self.brain.provider, session_id or self.brain.session_id, entries)
        self.updated = time.time()
        self.post({
            "type": "done",
            "text": reply,
            "error": error,
            "busy": bool(self._pending),
            "session_id": session_id or self.brain.session_id,
            "history_id": self.history_id,
            "model": model,
            "duration_ms": int((time.time() - started) * 1000),
        })
        self.log(f"  {'✗ ' + error if error else '✓'} {time.time() - started:.1f} s")


_assistant_entry = agentlog.assistant_entry
_turn_entries = agentlog.turn_entries
_record_agents = agentlog.record_turn


def _title_from(text: str, limit: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------- linked (terminal) chats
class LinkedChat:
    """A chat that lives in a terminal `vision` process (see link.py), driven over its Unix socket.

    Same face as Chat for the hub and the phone; the terminal runs the turns and streams every
    frame back, this end adds the chat id, keeps the transcript and does the speaking when the
    phone asked for a voice reply (the terminal need not have its voice loaded).
    """

    source = "terminal"

    def __init__(self, hub: "Hub", info: dict):
        self.hub = hub
        self.id = secrets.token_hex(6)
        self.pid = int(info["pid"])
        self.created = float(info.get("started") or time.time())
        self.updated = self.created
        self.state: dict = {}  # the terminal's latest summary
        self.title = ""
        self.busy = False
        self.partial = ""
        self.user_text = ""
        self._questions: list[dict] = []
        self.history_id = secrets.token_hex(12)
        self._transcript: list[dict] | None = None
        self._lock = threading.Lock()
        self._wire: WireSpeaker | None = None
        self._speak: deque[bool] = deque()  # speak flag per message this end sent, in order
        self._started = 0.0
        self.link = None
        self.agents: dict[str, dict] = {}  # this turn's agent rows, as Chat.agents (from the terminal's frames)
        self.tools: dict[str, dict] = {}  # …and its tool calls, as Chat.tools
        self._steers: list[tuple] = []

    # -- what the hub reads
    @property
    def provider(self) -> str:
        return self.state.get("provider", "")

    @property
    def session_id(self) -> str | None:
        return self.state.get("session_id") or None

    @property
    def workdir(self) -> str:
        return self.state.get("workdir", "")

    @property
    def model_id(self) -> str:
        return self.state.get("model_id", "")

    @property
    def effort(self) -> str:
        return self.state.get("effort", "")

    @property
    def waiting(self) -> bool:
        return bool(self.state.get("waiting"))

    def summary(self) -> dict:
        s = {k: v for k, v in self.state.items() if k not in ("type", "chat")}
        s.update({
            "type": "chat",
            "chat": self.id,
            "source": self.source,
            "title": self.title or s.get("title", ""),
            "history_id": self.history_id,
            "busy": self.busy,
            "agents": list(self.agents.values()) if self.busy else [],
            "tools": list(self.tools.values()) if self.busy else [],
            "waiting": self.waiting,
            "questions": self._questions if self.waiting else [],
            "partial": self.partial if self.busy else "",
            "user_text": self.user_text if self.busy else "",
            "created": self.created,
            "updated": self.updated,
        })
        return s

    def post(self, ev: dict) -> None:
        ev["chat"] = self.id
        self.hub.post(ev)

    def log(self, msg: str) -> None:
        self.hub.log(f"[{self.id[:4]} terminal] {msg}")

    # -- transcript (seeded from the session on disk, then kept up from the terminal's done frames)
    def reset_history(self) -> None:
        with self._lock:
            self._transcript = None
        self.history_id = secrets.token_hex(12)

    def history(self, limit: int = 60) -> list[dict]:
        from vision.sessions import session_history

        with self._lock:
            if self._transcript is None:
                self._transcript = session_history(self.provider, self.session_id or "") if self.session_id else []
            if not self.title:
                first = next((m["text"] for m in self._transcript if m.get("role") == "user" and m.get("text")), "")
                self.title = _title_from(first)
            return list(self._transcript[-max(1, min(limit, 200)):])

    # -- phone → terminal
    def queue(self, text: str, speak: bool, voice: bool, talk: bool = False) -> None:
        self._speak.append(speak)
        self.link.send({"type": "message", "text": text, "speak": speak, "voice": voice, "talk": talk})

    def answer(self, answers) -> None:
        self.link.send({"type": "answer", "answers": answers if isinstance(answers, dict) and answers else None})

    def unqueue(self, text: str) -> bool:
        if not self.link:
            return False
        self.link.send({"type": "unqueue", "text": text})
        return True

    def steer(self, text: str) -> bool:
        """The terminal runs the turn: it sends the message in (and says `steered`) or leaves it queued."""
        if not self.link:
            return False
        self.link.send({"type": "steer", "text": text})
        return True

    def cancel(self) -> None:
        if self._wire:
            self._wire.stop()
        if self.link:
            self.link.send({"type": "cancel"})

    async def switch_model(self, model: str, effort: str | None) -> None:
        if model:
            self.link.send({"type": "model", "model": model, "effort": effort or None})

    def quit(self) -> None:
        """Close from the phone: the terminal exits; its link dropping removes the chat from the list."""
        if self.link:
            self.link.send({"type": "quit"})

    def detach(self) -> None:
        if self._wire:
            self._wire.stop()
        if self.link:
            self.link.close()

    # -- terminal → phone
    def on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "chat":
            before = (self.provider, self.session_id)
            self.state = ev
            if (self.provider, self.session_id) != before and self._transcript is not None:
                self.reset_history()  # /new or /resume in the terminal: another conversation now
            if ev.get("title"):
                self.title = ev["title"]
            self.busy = bool(ev.get("busy"))
            self._questions = ev.get("questions") or []
            self.hub.post(self.summary())
            self.hub.dedupe()
            return
        if kind == "question":
            self._questions = ev.get("questions") or []
        elif kind == "answered":
            self._questions = []
        if kind == "start":
            self.busy, self.partial, self.user_text = True, "", ev.get("text", "")
            self.agents, self.tools, self._steers = {}, {}, []
            self.updated = self._started = time.time()
            if not self.title and self.user_text:
                self.title = _title_from(self.user_text)
            threading.Thread(target=self._seed, daemon=True).start()
            self.log(f"› {self.user_text[:80]}{'…' if len(self.user_text) > 80 else ''}")
            speak = bool(ev.get("speak"))
            if speak:
                try:
                    sp = self.hub.speaker()
                    self._wire = WireSpeaker(sp, sink=lambda seq, wav: self.post({"type": "audio", "seq": seq, "wav": base64.b64encode(wav).decode()}))
                    if ev.get("voice"):
                        from vision.cli import _arm_filler

                        _arm_filler(self._wire, sp, self.hub.cfg.voice)
                except Exception as e:  # noqa: BLE001
                    self.post({"type": "note", "text": f"voice unavailable: {e}"})
            self.post(ev)
            self.hub.post(self.summary())
            return
        if kind == "delta":
            self.partial += ev.get("text", "")
            self.post(ev)
            if self._wire:
                self._wire.feed(ev.get("text", ""))
            return
        if kind == "status":
            if self._wire and ev.get("tool"):
                self._wire.flush()
            ev.setdefault("label", status_label(ev.get("tool", "")))
            self.post(ev)
            return
        if kind == "done":
            wire, self._wire = self._wire, None
            if wire:
                try:
                    wire.finish()
                except Exception:  # noqa: BLE001
                    pass
                self.post({"type": "audio_end"})
            reply = ev.get("text") or self.partial
            # (the terminal keeps the agent rows with the conversation itself: vision.agentlog)
            entries = _turn_entries(self.user_text, reply, ev.get("error") or "", self.partial, list(self.agents.values()), self._steers,
                                    list(self.tools.values()))
            with self._lock:
                if self._transcript is None:
                    self._transcript = []
                self._transcript.extend(entries)
                self._transcript = self._transcript[-200:]
            self.updated = time.time()
            self.busy = bool(ev.get("busy"))
            ev.setdefault("history_id", self.history_id)
            ev.setdefault("duration_ms", int((time.time() - self._started) * 1000))
            self.post(ev)
            self.log(f"  {'✗ ' + ev['error'] if ev.get('error') else '✓'} {(time.time() - self._started):.1f} s")
            self.hub.post(self.summary())
            return
        if kind == "agent" and ev.get("id"):
            frame = {k: v for k, v in ev.items() if k != "chat"}
            frame["at"] = self.agents.get(ev["id"], {}).get("at", frame.get("at", len(self.partial)))
            self.agents[ev["id"]] = frame
        elif kind == "tool" and ev.get("id"):
            frame = {k: v for k, v in ev.items() if k != "chat"}
            frame["at"] = self.tools.get(ev["id"], {}).get("at", frame.get("at", len(self.partial)))
            self.tools[ev["id"]] = frame
            ev = {**ev, "at": frame["at"]}
        elif kind == "steered":
            self._steers.append((len(self.partial), ev.get("text") or "", len(self.agents)))
        self.post(ev)  # agent, tool, steered, question, note, error: straight through

    def _seed(self) -> None:
        try:
            self.history()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- hub
class Hub:
    """Owns the chats, the speech models and the connected sockets."""

    def __init__(self, cfg: Config, brain, token: str, log=print, follow_defaults: bool = False,
                 open_initial: bool = True, journal_enabled: bool = False):
        self.cfg = cfg
        # New chats re-read the saved default model/effort, so /default elsewhere applies without a
        # restart. Off when `vision serve --model/--effort` pinned them for this run.
        self.follow_defaults = follow_defaults
        self.journal_enabled = journal_enabled
        self.token = token
        self.log = log
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.events: asyncio.Queue | None = None
        self._speaker: Speaker | None = None
        self._stt = None
        self._lock = threading.Lock()
        self.chats: dict[str, Chat | LinkedChat] = {}
        self.links: dict[int, LinkedChat] = {}  # terminal chats by pid (see link.py)
        self.restart_wanted = ""  # who asked for a graceful restart (see request_restart); "" = none
        self.restarting = False  # every chat went quiet and the new process is on its way
        self.exec_restart = None  # what replaces this process (serve sets it); None = just log
        self._resume: list[tuple[Chat, list[str]]] = []  # messages a graceful restart carried over, run at startup
        # The other way round: terminal windows following one of this process's chats (remote.py), by
        # chat id, each socket with its process's pid, and when a window was launched for a chat that
        # has not dialled in yet. "Open on laptop" opens one window per chat.
        self.mirrors: dict[str, dict] = {}
        self._opening: dict[str, float] = {}
        self._dedupe_lock = threading.Lock()
        from vision.scheduled import Schedule

        self.schedule = Schedule()
        if journal_enabled:
            self.restore_chats()
        # Only an explicit --new or --continue opens a chat at startup.
        if open_initial:
            self.open_chat(brain, copy.deepcopy(cfg))

    # -- chats
    def open_chat(self, brain=None, cfg: Config | None = None, title: str = "",
                  chat_id: str | None = None, restored: dict | None = None) -> Chat:
        from vision.brain import create_brain

        if cfg is None:
            cfg = copy.deepcopy(self.cfg)
            if brain is None and self.follow_defaults:
                saved = saved_brain_defaults()
                if saved is None:
                    self.log("config.toml unreadable; new chat uses the defaults from start-up")
                else:
                    cfg.brain.model, cfg.brain.effort = saved
        retired = ""
        if brain is None:
            from vision.models import coerce_effort, usable_model

            cfg.brain.model, retired = usable_model(cfg.brain.model)  # a saved model since retired
            if retired:
                cfg.brain.effort, _ = coerce_effort(cfg.brain.model, cfg.brain.effort)
                self.log(retired)
            from vision.providers import settle

            moved = settle(cfg)  # its provider isn't set up here, or is switched off
            if moved:
                retired = f"{retired} {moved}".strip()
                self.log(moved)
            brain = create_brain(cfg.brain, voice_mode=False, continue_session=False)
        chat = Chat(self, brain, cfg, title=title, chat_id=chat_id, restored=restored)
        self.chats[chat.id] = chat
        if retired:
            chat.post({"type": "note", "text": retired})
        return chat

    def restore_chats(self) -> None:
        from vision.brain import create_brain
        from vision.turnjournal import recover, saved_chats

        graceful = take_restart_marker()  # the last process restarted on purpose: its queued messages run
        for chat_id, records in saved_chats():
            try:
                state = recover(records, requeue=graceful)
                cfg = copy.deepcopy(self.cfg)
                cfg.brain.model = state["model"] or cfg.brain.model
                cfg.brain.effort = state["effort"] or cfg.brain.effort
                from vision.models import provider_for

                sid = state["session_id"]
                from vision.providers import cap

                if state["interrupted"] and not cap(provider_for(cfg.brain.model), "keeps_partial_turns", True):
                    sid = None  # its saved model context ends at the last completed turn
                brain = create_brain(cfg.brain, voice_mode=False, session_id=sid)
                chat = self.open_chat(brain, cfg, title=state["title"], chat_id=chat_id, restored=state)
                if graceful and state["pending"]:
                    self._resume.append((chat, list(state["pending"])))
                elif state["pending"] and chat.journal:
                    # Recovery showed these as lost. Retire them durably so a later
                    # graceful restart cannot silently run them after the user retries.
                    for text in state["pending"]:
                        chat.journal.append("unqueued", text=text)
                        chat.journal.append("steer_failed", text=text)
            except Exception as e:  # noqa: BLE001
                self.log(f"could not restore chat {chat_id}: {e}")

    # -- graceful restart
    def busy_chats(self) -> list:
        """The server's own chats still replying, or with messages waiting to run."""
        return [c for c in self.chats.values() if isinstance(c, Chat) and (c.busy or c._pending)]

    def request_restart(self, by: str) -> int:
        """Restart once every chat is quiet: nothing is cut off mid-reply. Returns how many chats it
        is waiting on (0: it restarts now). Asking again while one is pending changes nothing."""
        waiting = len(self.busy_chats())
        if not self.restart_wanted:
            self.restart_wanted = by or "someone"
            self.log(f"restart asked for ({self.restart_wanted})" + (f"; waiting on {waiting} chat{'s' * (waiting != 1)} to finish" if waiting else ""))
            if waiting:
                self.post({"type": "toast", "text": "Vision restarts once the current replies finish."})
        self.maybe_restart()
        return waiting

    def maybe_restart(self) -> None:
        """Restart now if one was asked for and nothing is running. From here on a new message is
        held (Chat.queue) and journalled, and the next process runs it (restore_chats)."""
        if not self.restart_wanted or self.restarting or self.busy_chats():
            return
        if self.exec_restart is None:
            return
        self.restarting = True
        self.log("restarting…")
        self.post({"type": "toast", "text": "Restarting Vision…"})
        write_restart_marker()
        if self.loop is not None:
            self.loop.call_later(0.4, self.exec_restart)  # let the toast reach the phone first
        else:
            self.exec_restart()

    def close_chat(self, chat: Chat, already_cancelled: bool = False) -> None:
        if isinstance(chat, Chat) and chat.journal:
            chat.journal.append("close")
        if not already_cancelled:
            chat.cancel()
        if isinstance(chat, Chat) and chat.journal:
            chat.journal.release()
        self.chats.pop(chat.id, None)
        # No replacement: the phone shows an empty screen and opens a chat with its next message.

    # -- terminal chats (link.py): found on disk, listed while their process lives
    LINK_POLL = 2.0

    def _attach_link(self, info: dict) -> None:
        from vision.link import LinkClient

        pid = int(info["pid"])
        if pid in self.links:
            return
        chat = LinkedChat(self, info)
        client = LinkClient(info, on_event=chat.on_event, on_close=lambda: self._drop_link(pid))
        try:
            client.connect()
        except OSError:
            return  # not listening yet, or gone; next sweep decides
        chat.link = client
        self.links[pid] = chat
        self.chats[chat.id] = chat
        self.log(f"terminal chat joined (pid {pid})")

    def _drop_link(self, pid: int) -> None:
        chat = self.links.pop(pid, None)
        if chat is None:
            return
        self.chats.pop(chat.id, None)
        self.post({"type": "chat_closed", "chat": chat.id})
        self.log(f"terminal chat left (pid {pid})")

    def sweep_links(self) -> None:
        from vision.link import list_links

        seen = set()
        for info in list_links():
            seen.add(int(info["pid"]))
            self._attach_link(info)
        for pid in list(self.links):
            if pid not in seen:
                self.links[pid].detach()  # descriptor gone: the terminal is closing
        self.dedupe()  # a copy that was mid-turn when its terminal showed up closes once idle

    def dedupe(self) -> None:
        """One conversation, one chat. When a terminal holds the same session as a chat this server
        runs (the one `vision serve` continued at start-up, or one resumed on both sides), the terminal
        keeps it and the server's copy closes as soon as it is idle; phones follow to the terminal row."""
        owners = {(c.provider, c.session_id): c for c in list(self.links.values()) if c.session_id}
        if not owners:
            return
        with self._dedupe_lock:  # runs on link threads and the sweep
            for chat in list(self.chats.values()):
                if chat.source != "server":
                    continue
                owner = owners.get((chat.provider, chat.session_id))
                asking = chat._question is not None and not chat._question.done()
                if owner is None or chat.busy or chat._pending or asking:
                    continue
                self.close_chat(chat)
                self.post({"type": "chat_closed", "chat": chat.id, "moved_to": owner.id})
                self.log(f"[{chat.id[:4]}] closed: its conversation is open in a terminal (pid {owner.pid})")

    async def _watch_links(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                await loop.run_in_executor(None, self.sweep_links)
            except Exception as e:  # noqa: BLE001
                self.log(f"link sweep failed: {e}")
            await asyncio.sleep(self.LINK_POLL)

    # -- terminal windows following a chat (remote.py); only touched on the event loop
    OPEN_GRACE = 15.0  # seconds a launched window has to dial in before another tap may open a second

    def mirror_joined(self, chat_id: str, ws, pid: int | None) -> None:
        self.mirrors.setdefault(chat_id, {})[ws] = pid
        self._opening.pop(chat_id, None)

    def mirror_left(self, chat_id: str, ws) -> None:
        windows = self.mirrors.get(chat_id, {})
        windows.pop(ws, None)
        if not windows:
            self.mirrors.pop(chat_id, None)

    def opening(self, chat_id: str) -> bool:
        """A window was launched for this chat moments ago and is still starting up."""
        return time.monotonic() - self._opening.get(chat_id, float("-inf")) < self.OPEN_GRACE

    # -- scheduled tasks (scheduled.py): each run opens a fresh chat and sends the task's prompt
    SCHEDULE_POLL = 20.0

    def run_task(self, task: dict) -> Chat:
        chat = self.open_chat(title=task.get("title") or "")
        if chat.journal:
            chat.journal.append("scheduled", task_id=task["id"])
        chat.updated = time.time()
        self.schedule.ran(task["id"], chat.id)
        self.post(chat.summary())
        chat.queue(task["prompt"], speak=False, voice=False)
        self.post({"type": "toast", "text": f"Scheduled: {chat.title or 'task'} started"})
        self.post({"type": "scheduled"})
        self.log(f"scheduled task {task['id']} → chat {chat.id[:4]}")
        return chat

    async def _watch_schedule(self) -> None:
        while True:
            try:
                for task in self.schedule.due():
                    self.run_task(task)
            except Exception as e:  # noqa: BLE001
                self.log(f"scheduled tasks failed: {e}")
            await asyncio.sleep(self.SCHEDULE_POLL)

    def chat(self, cid: str | None) -> Chat | None:
        if cid:
            return self.chats.get(cid)
        # No id (an older app build): the most recently active chat.
        return max(self.chats.values(), key=lambda c: c.updated, default=None)

    # -- auth
    def authorized(self, token: str | None) -> bool:
        return bool(token) and secrets.compare_digest(token, self.token)

    # -- models (lazy, thread-safe)
    def speaker(self) -> Speaker:
        with self._lock:
            fresh = self._speaker is None
            if fresh:
                self._speaker = Speaker(self.cfg.voice)
        self._speaker._load()
        if fresh and self.cfg.voice.filler:
            # Its "One sec." clips, made once and cached on disk; a turn before they are ready just stays quiet.
            sp, phrases = self._speaker, self.cfg.voice.filler_phrases
            threading.Thread(target=lambda: sp.prepare_fillers(phrases), daemon=True).start()
        return self._speaker

    def stt(self):
        with self._lock:
            if self._stt is None:
                from vision.stt import Transcriber

                self._stt = Transcriber(self.cfg.listen)
        self._stt.warm_up()
        return self._stt

    def warm_up(self) -> None:
        """Load the ears and the voice in the background so the first spoken turn is quick."""
        def ears():
            try:
                stt = self.stt()
                self.log(f"ears ready: {stt.device}")
            except Exception as e:  # noqa: BLE001
                self.log(f"ears unavailable: {e}")

        def voice():
            try:
                sp = self.speaker()
                self.log(f"voice ready: {sp.voice} on {sp.device}")
            except Exception as e:  # noqa: BLE001
                self.log(f"voice unavailable: {e}")

        def titles():
            # The continued session's title comes from its transcript on disk.
            for chat in list(self.chats.values()):
                try:
                    chat.history(1)
                except Exception:  # noqa: BLE001
                    continue
                if chat.title:
                    self.post(chat.summary())

        threading.Thread(target=ears, daemon=True).start()
        threading.Thread(target=voice, daemon=True).start()
        threading.Thread(target=titles, daemon=True).start()

    # -- snapshot for a (re)connecting client
    def hello(self) -> dict:
        chats = sorted(self.chats.values(), key=lambda c: c.created)
        return {
            "type": "hello",
            "version": __version__,
            "workdir": chats[0].workdir if chats else "",
            "chats": [c.summary() for c in chats],
        }

    # -- event fan-out (events queued from any thread, sent in order from the loop)
    def post(self, ev: dict) -> None:
        if self.loop and self.events:
            self.loop.call_soon_threadsafe(self.events.put_nowait, ev)

    async def broadcast(self, ev: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(ev)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def pump(self) -> None:
        assert self.events
        while True:
            ev = await self.events.get()
            await self.broadcast(ev)

    # -- incoming frames
    async def handle(self, ws, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "ping":
            await ws.send_json({"type": "pong"})
            return
        if kind == "new":
            chat = self.open_chat()
            await ws.send_json({**chat.summary(), "opened": True})  # the asker switches to it
            self.post(chat.summary())                                # everyone else just lists it
            return
        if kind == "resume":
            await self.resume(ws, msg)
            return
        if kind == "model" and not self.chats and not msg.get("chat"):
            # The phone's model picker is usable before its first chat. Its choice becomes the
            # default for the chat it opens when the first message arrives.
            from vision.models import coerce_effort, find

            model = str(msg.get("model") or "")
            if not find(model):
                await ws.send_json({"type": "error", "text": f"unknown model {model!r}"})
                return
            effort, _ = coerce_effort(model, str(msg.get("effort") or self.cfg.brain.effort or ""))
            try:
                save_brain_defaults(model, effort)
            except Exception as e:  # noqa: BLE001
                await ws.send_json({"type": "error", "text": f"could not save the model: {e}"})
                return
            self.cfg.brain.model, self.cfg.brain.effort = model, effort
            return
        if kind == "restart" or (kind == "message" and (msg.get("text") or "").strip().lower() == "/restart"):
            self.request_restart("phone")  # it tells every phone what happens next
            return
        chat = self.chat(msg.get("chat"))
        if chat is None:
            await ws.send_json({"type": "error", "text": "that chat is gone"})
            return
        if kind == "message":
            text = with_attachments((msg.get("text") or "").strip(), msg.get("images"))
            # `now`: send it into the running turn (Claude takes it at its next step); else it queues.
            if text and not (msg.get("now") and chat.busy and await asyncio.to_thread(chat.steer, text)):
                chat.queue(text, bool(msg.get("speak")), bool(msg.get("voice")), bool(msg.get("talk")))
        elif kind == "unqueue":
            # A queued message removed (or taken back to edit) on the phone: it never runs.
            text = with_attachments((msg.get("text") or "").strip(), msg.get("images"))
            if text:
                chat.unqueue(text)
        elif kind == "steer":
            # A queued message the phone wants sent into the running turn now.
            text = with_attachments((msg.get("text") or "").strip(), msg.get("images"))  # as it was queued
            if text and not await asyncio.to_thread(chat.steer, text):
                await ws.send_json({"type": "toast", "text": "This model can't take a message mid-reply, so it stays queued."})
        elif kind == "answer":
            chat.answer(msg.get("answers"))
        elif kind == "cancel":
            await asyncio.to_thread(chat.cancel)
        elif kind == "close":
            if chat.source == "terminal":
                chat.quit()  # the terminal exits as if /quit were typed; its link going away removes it from the list
                return
            await asyncio.to_thread(chat.cancel)
            self.close_chat(chat, already_cancelled=True)
            self.post({"type": "chat_closed", "chat": chat.id})
        elif kind == "model":
            if chat.busy:
                await ws.send_json({"type": "error", "text": "Vision is still replying; switch models once it has finished."})
                return
            await chat.switch_model(msg.get("model") or "", msg.get("effort"))
        elif kind == "open_terminal":
            # The phone asks for this chat in a terminal window here: `vision --join <chat>` (remote.py
            # attaches it; this process keeps running the conversation, the window mirrors it).
            if chat.source == "terminal":
                await ws.send_json({"type": "toast", "text": "That chat is already open in a terminal."})
                return
            windows = self.mirrors.get(chat.id)
            if windows:
                pids = [pid for pid in windows.values() if pid]
                shown = bool(pids) and await asyncio.to_thread(focus_terminal, pids[-1])
                text = "Already open on the laptop; brought it to the front." if shown else "That chat is already open on the laptop."
                await ws.send_json({"type": "toast", "text": text})
                return
            if self.opening(chat.id):
                await ws.send_json({"type": "toast", "text": "That chat is already opening on the laptop."})
                return
            self._opening[chat.id] = time.monotonic()  # before the await: a second tap lands while it runs
            try:
                term = await asyncio.to_thread(open_terminal, chat.id, chat.workdir)
            except Exception as e:  # noqa: BLE001
                self._opening.pop(chat.id, None)
                await ws.send_json({"type": "error", "text": f"could not open a terminal: {e}"})
                return
            chat.log(f"opened in a terminal ({term}) from the phone")
            await ws.send_json({"type": "toast", "text": "Opened in a terminal on the laptop."})
        else:
            await ws.send_json({"type": "error", "text": f"unknown frame type {kind!r}"})

    async def resume(self, ws, msg: dict) -> None:
        """Open an earlier conversation as its own chat (or switch to the chat that already has it)."""
        from vision.brain import create_brain
        from vision.sessions import find_any_session, find_session, session_history

        sid = msg.get("session_id") or ""
        if not sid:
            return
        provider = (msg.get("provider") or "").strip().lower() or None
        info = find_session(provider, sid) if provider else None
        if info is None:
            info = find_any_session(sid, prefer=self.cfg.brain.provider if hasattr(self.cfg.brain, "provider") else None)
        if info is None:
            await ws.send_json({"type": "error", "text": f"no conversation starts with “{sid}”"})
            return
        for chat in self.chats.values():
            if chat.provider == info.provider and chat.session_id == info.id:
                await ws.send_json({**chat.summary(), "opened": True})
                return
        loop = asyncio.get_running_loop()

        def build():
            from vision.cli import _apply_session

            cfg = copy.deepcopy(self.cfg)
            brain = create_brain(cfg.brain, voice_mode=False, continue_session=False)
            detail = _apply_session(brain, info)
            return cfg, brain, detail

        try:
            cfg, brain, detail = await loop.run_in_executor(None, build)
        except Exception as e:  # noqa: BLE001
            await ws.send_json({"type": "error", "text": f"could not open that conversation: {e}"})
            return
        chat = self.open_chat(brain, cfg, title=info.title)
        if info.provider != brain.provider:
            chat._transcript = session_history(info.provider, info.id)
        chat.updated = time.time()
        await ws.send_json({**chat.summary(), "opened": True})
        self.post(chat.summary())
        chat.post({"type": "note", "text": detail})

    def cancel(self) -> None:
        for chat in list(self.chats.values()):
            if chat.source == "terminal":
                chat.detach()
            else:
                chat.cancel()

    # -- speech helpers (thread pool)
    def transcribe_bytes(self, data: bytes) -> str:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.stt().transcribe(np.ascontiguousarray(audio), sr)

    def speak_bytes(self, text: str) -> bytes:
        from vision.tts import speechify

        sp = self.speaker()
        return wav_bytes(sp.synth(speechify(text)))


# ---------------------------------------------------------------- app
def create_app(hub: Hub) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.events = asyncio.Queue()
        asyncio.create_task(hub.pump())
        asyncio.create_task(hub._watch_links())
        asyncio.create_task(hub._watch_schedule())
        for chat, texts in hub._resume:
            chat.resume_queued(texts)
        hub._resume.clear()
        hub.warm_up()
        yield
        # The voice transport stays alive between requests; reap it when the server stops.
        await asyncio.to_thread(hub.cancel)

    app = FastAPI(title="Vision Remote", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)

    def bearer(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.query_params.get("token")
        if not hub.authorized(token):
            raise HTTPException(status_code=401, detail="bad token")

    @app.post("/restart", dependencies=[Depends(bearer)])
    async def restart() -> dict:
        """`vision restart`: restart once every chat is quiet (see Hub.request_restart)."""
        return {"waiting": hub.request_restart("vision restart")}

    @app.get("/health", dependencies=[Depends(bearer)])
    async def health() -> dict:
        h = hub.hello()
        h["type"] = "health"
        return h

    def chat_or_404(cid: str | None) -> Chat:
        chat = hub.chat(cid)
        if chat is None:
            raise HTTPException(status_code=404, detail="no such chat")
        return chat

    @app.get("/chats", dependencies=[Depends(bearer)])
    async def chats() -> list[dict]:
        return hub.hello()["chats"]

    @app.get("/history", dependencies=[Depends(bearer)])
    async def history(limit: int = 60, chat: str | None = None) -> list[dict]:
        c = chat_or_404(chat)
        return await asyncio.get_running_loop().run_in_executor(None, c.history, limit)

    @app.get("/models", dependencies=[Depends(bearer)])
    async def models(chat: str | None = None) -> dict:
        from vision.models import effort_choices
        from vision.providers import model_tabs

        c = hub.chat(chat)
        if c is None:
            if chat:
                raise HTTPException(status_code=404, detail="no such chat")
            model, effort = (saved_brain_defaults() or (hub.cfg.brain.model, hub.cfg.brain.effort)) if hub.follow_defaults else (hub.cfg.brain.model, hub.cfg.brain.effort)
            return defaults_payload(model, effort)
        tabs = [
            {"tab": tab, "models": [{"id": v, "label": label, "description": desc} for v, label, desc in entries]}
            for tab, entries, _ in model_tabs(hub.cfg, setup_rows=False)  # enabled + set up: the phone can't install
        ]
        efforts = [{"id": v, "label": label, "description": desc} for v, label, desc in effort_choices(c.model_id)]
        return {"tabs": tabs, "efforts": efforts, "current": c.model_id, "effort": c.effort}

    def defaults_payload(model: str, effort: str) -> dict:
        from vision.models import effort_choices
        from vision.providers import model_tabs

        tabs = [
            {"tab": tab, "models": [{"id": v, "label": label, "description": desc} for v, label, desc in entries]}
            for tab, entries, _ in model_tabs(hub.cfg, setup_rows=False)  # enabled + set up: the phone can't install
        ]
        efforts = [{"id": v, "label": label, "description": desc} for v, label, desc in effort_choices(model)]
        return {"tabs": tabs, "efforts": efforts, "current": model, "effort": effort}

    @app.get("/defaults", dependencies=[Depends(bearer)])
    async def get_defaults() -> dict:
        """The model and effort a new chat starts on: config.toml's [brain], as /default saves it."""
        model, effort = saved_brain_defaults() or (hub.cfg.brain.model, hub.cfg.brain.effort)
        return defaults_payload(model, effort)

    @app.post("/defaults", dependencies=[Depends(bearer)])
    async def set_defaults(body: dict) -> dict:
        """Save a new default, like /default in the terminal. Open chats keep their own model."""
        from vision.models import coerce_effort, find

        model = str(body.get("model") or "")
        if not find(model):
            raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
        effort, _ = coerce_effort(model, str(body.get("effort") or ""))
        try:
            save_brain_defaults(model, effort)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"could not save config.toml: {e}")
        # Also this server's own fallback, so new chats follow it even when a flag pinned the start-up model.
        hub.cfg.brain.model, hub.cfg.brain.effort = model, effort
        hub.log(f"default saved from the phone → {model} · {effort or 'effort off'}")
        return defaults_payload(model, effort)

    @app.get("/providers", dependencies=[Depends(bearer)])
    async def providers_list() -> dict:
        """Every provider the laptop knows, with how to draw it (icon, tint) and whether it is on and ready."""
        from vision.providers import REGISTRY

        rows = await asyncio.get_running_loop().run_in_executor(None, lambda: [p.describe(hub.cfg) for p in REGISTRY.values()])
        return {"providers": rows}

    @app.get("/sessions", dependencies=[Depends(bearer)])
    async def sessions() -> list[dict]:
        from vision.sessions import list_all_sessions

        rows = await asyncio.get_running_loop().run_in_executor(None, list_all_sessions)
        rows.sort(key=lambda s: s.last_active, reverse=True)  # every provider merged, newest first
        open_ids = {(c.provider, c.session_id): c.id for c in hub.chats.values()}
        return [
            {
                "id": s.id,
                "title": s.title,
                "age": s.age(),
                "updated": s.last_active,
                "provider": s.provider,
                "current": (s.provider, s.id) in open_ids,
                "chat": open_ids.get((s.provider, s.id)),
            }
            for s in rows
        ]

    @app.get("/usage", dependencies=[Depends(bearer)])
    async def usage(provider: str = "all", chat: str | None = None) -> dict:
        """Subscription usage for the phone's Usage page: every provider, or one. The on-screen chat's
        brain answers for its own provider; the others get a throwaway brain, as `/usage all` does."""
        from vision.usage import PROVIDERS, usage_all, usage_data

        c = hub.chat(chat)
        brain = getattr(c, "brain", None)  # a terminal-linked chat has no brain of its own here
        cfg = getattr(c, "cfg", None) or hub.cfg
        loop = asyncio.get_running_loop()
        if provider == "all":
            rows = await loop.run_in_executor(None, usage_all, cfg, brain)
        elif provider in PROVIDERS:
            rows = [await loop.run_in_executor(None, usage_data, cfg, brain, provider)]
        else:
            raise HTTPException(status_code=400, detail=f"provider must be all, {', '.join(PROVIDERS)}")
        return {"providers": rows, "at": time.time()}

    @app.post("/usage/reset", dependencies=[Depends(bearer)])
    async def use_reset(body: dict) -> dict:
        """Spend one of a provider's banked resets ({"provider", "id" (optional), "chat"}), then read its usage again:
        {"ok", "outcome", "message", "provider": the fresh usage row}."""
        from vision.usage import use_banked, usage_data

        from vision.providers import names

        provider = str(body.get("provider") or "")
        if provider not in names(banked_resets=True):
            raise HTTPException(status_code=400, detail=f"provider must be {' or '.join(names(banked_resets=True))}")
        c = hub.chat(body.get("chat"))
        brain = getattr(c, "brain", None)
        cfg = getattr(c, "cfg", None) or hub.cfg
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, use_banked, provider, body.get("id") or None)
        hub.log(f"banked {provider} reset from the phone → {result['outcome']}")
        result["provider"] = await loop.run_in_executor(None, usage_data, cfg, brain, provider)
        return result

    @app.get("/library", dependencies=[Depends(bearer)])
    async def library_items() -> list[dict]:
        """Pictures replies showed and everything sent from the phone, newest first."""
        from vision.library import library

        return await asyncio.to_thread(library, UPLOAD_DIR, image_path_allowed)

    @app.delete("/library", dependencies=[Depends(bearer)])
    async def library_delete(path: str) -> dict:
        """Delete one Library item (a reply's picture or something sent from the phone) from the laptop."""
        from vision.library import delete

        try:
            found = await asyncio.to_thread(delete, path, UPLOAD_DIR, image_path_allowed)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"could not delete it: {e}")
        if not found:
            raise HTTPException(status_code=404, detail="not in the library")
        hub.log(f"library item deleted from the phone: {path}")
        return {"ok": True}

    @app.get("/thumb", dependencies=[Depends(bearer)])
    async def thumb(path: str, size: int = 360) -> Response:
        """A small JPEG of a library picture (a video's first frame), for the Library grid."""
        from vision.library import thumbnail

        try:
            real = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(status_code=404, detail="no such picture")
        if real.suffix.lower() in VIDEO_TYPES and real.parent == UPLOAD_DIR.resolve():
            frames = (video_digest(real) or {}).get("frames") or []
            real = Path(frames[0][1]) if frames and os.path.isfile(frames[0][1]) else real
        if not real.is_file() or real.suffix.lower() not in IMAGE_TYPES or not image_path_allowed(real):
            raise HTTPException(status_code=404, detail="no such picture")
        try:
            data = await asyncio.to_thread(thumbnail, real, max(64, min(size, 1024)))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"could not read that picture: {e}")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/scheduled", dependencies=[Depends(bearer)])
    async def scheduled_list() -> list[dict]:
        return hub.schedule.list()

    @app.post("/scheduled", dependencies=[Depends(bearer)])
    async def scheduled_save(body: dict) -> dict:
        """Add a task, or edit one (`id` set): {title, prompt, repeat, time, date?, weekday?, enabled}."""
        try:
            task = hub.schedule.upsert(body)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        hub.log(f"scheduled task saved from the phone: {task['title']!r} ({task['repeat']} {task['time']})")
        hub.post({"type": "scheduled"})
        return task

    @app.delete("/scheduled/{tid}", dependencies=[Depends(bearer)])
    async def scheduled_delete(tid: str) -> dict:
        if not hub.schedule.delete(tid):
            raise HTTPException(status_code=404, detail="no such task")
        hub.post({"type": "scheduled"})
        return {"ok": True}

    @app.post("/scheduled/{tid}/run", dependencies=[Depends(bearer)])
    async def scheduled_run(tid: str) -> dict:
        """Run a task now, whatever its schedule; answers with the chat it opened."""
        task = hub.schedule.get(tid)
        if task is None:
            raise HTTPException(status_code=404, detail="no such task")
        chat = hub.run_task(task)
        return {"chat": chat.id}

    @app.post("/transcribe", dependencies=[Depends(bearer)])
    async def transcribe(file: UploadFile = File(...), preview: bool = False) -> dict:
        """`preview=1` is the phone's live words: the utterance so far, sent every second or so while
        you talk. Only served when Whisper is on the GPU (409 otherwise, the phone falls back to its own),
        and not logged."""
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty audio")
        if preview and not hub.stt().can_preview_live:
            raise HTTPException(status_code=409, detail="live preview needs Whisper on the GPU")
        try:
            text = await asyncio.get_running_loop().run_in_executor(None, hub.transcribe_bytes, data)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"could not decode audio: {e}")
        if preview:
            return {"text": text}
        hub.log(f"heard: {text!r}")
        return {"text": text}

    @app.post("/upload", dependencies=[Depends(bearer)])
    async def upload(file: UploadFile = File(...)) -> dict:
        """A photo, video or any other file from the phone, saved where every brain can read it; the message
        frame then names the path. A video is digested here (frames + transcript), so a bad clip fails the
        upload. Other files keep their name after the stamp, so the brain (and the phone's chip) sees it."""
        name = os.path.basename(file.filename or "")
        ext = os.path.splitext(name)[1].lower() or ".jpg"
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        if ext not in IMAGE_TYPES and ext not in VIDEO_TYPES:
            safe = re.sub(r"[^\w.\- ]", "_", name).strip(" .")[-120:] or "file"
            path = UPLOAD_DIR / f"{stamp}-{safe}"
        else:
            path = UPLOAD_DIR / f"{stamp}{ext}"

        def save() -> int:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            size = 0
            try:
                with path.open("xb") as out:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise HTTPException(status_code=413, detail=f"uploads are limited to {MAX_FILE_BYTES // (1024 * 1024)} MB")
                        out.write(chunk)
                if not size:
                    raise HTTPException(status_code=400, detail="empty file")
                return size
            except BaseException:
                path.unlink(missing_ok=True)
                raise

        size = await asyncio.to_thread(save)
        if ext not in IMAGE_TYPES and ext not in VIDEO_TYPES:
            hub.log(f"file: {path.name} ({size // 1024} KB)")
            return {"path": str(path)}
        if ext in IMAGE_TYPES:
            hub.log(f"photo: {path.name} ({size // 1024} KB)")
            return {"path": str(path)}
        started = time.monotonic()
        try:
            digest = await asyncio.get_running_loop().run_in_executor(
                None, digest_video, path, getattr(hub, "transcribe_bytes", None))
        except Exception as e:  # noqa: BLE001
            path.unlink(missing_ok=True)
            shutil.rmtree(path.with_name(path.stem + "-frames"), ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"could not read that video: {e}")
        hub.log(f"video: {path.name} ({size // 1024} KB, {clock(digest['duration'])}, "
                f"{len(digest['frames'])} frames, {len(digest['transcript'])} chars heard, {time.monotonic() - started:.1f} s)")
        return {"path": str(path), "frames": str(len(digest["frames"])), "duration": str(digest["duration"])}

    @app.get("/uploads/{name}", dependencies=[Depends(bearer)])
    async def uploaded(name: str) -> Response:
        """A photo sent earlier, by file name; the phone re-fetches thumbnails when it reloads a transcript.
        For a video that thumbnail is its first frame."""
        path = UPLOAD_DIR / os.path.basename(name)
        if path.parent != UPLOAD_DIR or not path.is_file():
            raise HTTPException(status_code=404, detail="no such photo")
        if path.suffix.lower() in VIDEO_TYPES:
            frames = (video_digest(path) or {}).get("frames") or []
            if not frames or not os.path.isfile(frames[0][1]):
                raise HTTPException(status_code=404, detail="no thumbnail for that video")
            path = Path(frames[0][1])
        data = await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)
        return Response(content=data, media_type=IMAGE_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/image", dependencies=[Depends(bearer)])
    async def reply_image(path: str) -> FileResponse:
        """A picture a reply shows (`![caption](/abs/path.png)`), for the phone to draw inline."""
        try:
            real = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(status_code=404, detail="no such image")
        if not real.is_file() or real.suffix.lower() not in IMAGE_TYPES or not image_path_allowed(real):
            raise HTTPException(status_code=404, detail="no such image")
        return FileResponse(real, media_type=IMAGE_TYPES[real.suffix.lower()])

    @app.get("/videos/{name}", dependencies=[Depends(bearer)])
    async def uploaded_video(name: str) -> FileResponse:
        """A video sent earlier, for the phone to play back; byte ranges supported, as AVPlayer asks for them."""
        path = UPLOAD_DIR / os.path.basename(name)
        if path.parent != UPLOAD_DIR or path.suffix.lower() not in VIDEO_TYPES or not path.is_file():
            raise HTTPException(status_code=404, detail="no such video")
        return FileResponse(path, media_type="video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4")

    @app.post("/speak", dependencies=[Depends(bearer)])
    async def speak(request: Request):
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="nothing to say")
        wav = await asyncio.get_running_loop().run_in_executor(None, hub.speak_bytes, text)
        return Response(content=wav, media_type="audio/wav")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        auth = websocket.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else websocket.query_params.get("token")
        # Accept first either way: a bad token then gets a clean close code (4401) the phone can recognise,
        # rather than a generic handshake failure it would keep retrying.
        await websocket.accept()
        if not hub.authorized(token):
            await websocket.close(code=4401, reason="bad token")
            return
        hub.clients.add(websocket)
        who = "terminal" if websocket.query_params.get("client") == "terminal" else "phone"
        # A terminal window following one chat (remote.py) says which, and its pid: one window per chat.
        mirror = websocket.query_params.get("chat") if who == "terminal" else None
        if mirror:
            pid = websocket.query_params.get("pid") or ""
            hub.mirror_joined(mirror, websocket, int(pid) if pid.isdigit() else None)
        hub.log(f"{who} connected ({len(hub.clients)} client{'s' if len(hub.clients) != 1 else ''})")
        try:
            await websocket.send_json(hub.hello())
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    await websocket.send_json({"type": "error", "text": "frames must be JSON objects"})
                    continue
                if isinstance(msg, dict):
                    await hub.handle(websocket, msg)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            hub.log(f"socket error: {e}")
        finally:
            hub.clients.discard(websocket)
            if mirror:
                hub.mirror_left(mirror, websocket)
            hub.log(f"{who} disconnected")

    @app.exception_handler(404)
    async def _404(request: Request, exc) -> JSONResponse:
        return JSONResponse({"detail": "not found"}, status_code=404)

    return app


# ---------------------------------------------------------------- start-up
def tailscale_url(port: int) -> str | None:
    """https://<machine>.<tailnet>.ts.net if Tailscale Serve/Funnel is publishing this port.

    Tailscale merely being up is not enough: without a serve config the https:// name answers nothing,
    and a QR pointing there would send the phone to a dead address.
    """
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=5, encoding="utf-8")
        name = json.loads(r.stdout).get("Self", {}).get("DNSName", "").rstrip(".")
        r = subprocess.run([exe, "serve", "status", "--json"], capture_output=True, text=True, timeout=5, encoding="utf-8")
        published = f":{port}" in r.stdout  # a proxy handler like "http://127.0.0.1:8765"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return f"https://{name}" if name and published else None


def lan_ip() -> str | None:
    """The address this machine has on its default route (what a phone on the same Wi-Fi dials)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent; the kernel just picks the source address
            return s.getsockname()[0]
    except OSError:
        return None


def pairing_payload(url: str, token: str) -> str:
    return json.dumps({"url": url, "token": token})


def qr_lines(payload: str) -> list[str]:
    """The pairing QR as terminal lines (half-block characters, two modules per row)."""
    import qrcode

    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(payload)
    qr.make(fit=True)
    m = qr.get_matrix()
    if len(m) % 2:
        m.append([False] * len(m[0]))
    blocks = {(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}
    return ["".join(blocks[(top, bot)] for top, bot in zip(m[y], m[y + 1])) for y in range(0, len(m), 2)]


def pairing_info(cfg: Config, host: str | None, port: int | None, public_url: str | None, regenerate_token: bool = False) -> dict:
    """Where the phone should connect and the token it should present (what the QR code carries)."""
    host = host or cfg.remote.host
    port = port or cfg.remote.port
    token = load_token(regenerate_token)
    # Bound to every interface: the phone dials the LAN address, not 0.0.0.0.
    reach = (lan_ip() or "127.0.0.1") if host in ("0.0.0.0", "") else host
    local = f"http://{reach}:{port}"
    url = public_url or cfg.remote.public_url or tailscale_url(port) or local
    return {"host": host, "port": port, "local": local, "url": url, "token": token, "payload": pairing_payload(url, token)}


RESTART_MARKER = STATE_DIR / "restart.json"  # a graceful restart is under way (see Hub.maybe_restart)


def write_restart_marker() -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        RESTART_MARKER.write_text(json.dumps({"at": time.time()}), encoding="utf-8")
    except OSError:
        pass


def take_restart_marker() -> bool:
    """True once after a graceful restart (a marker less than five minutes old), then gone."""
    try:
        at = json.loads(RESTART_MARKER.read_text(encoding="utf-8")).get("at") or 0
        RESTART_MARKER.unlink()
    except (OSError, ValueError, AttributeError):
        return False
    return time.time() - at < 300


RESTART_DROPPED = {"--new", "--continue", "-c", "--new-token"}  # one-off start-up flags a restart must not repeat


def restart_argv(argv: list[str]) -> list[str]:
    """The command line that brings `vision serve` back: the same interpreter and options, less the
    one-off ones (chats come back from the journal; the pairing token stays)."""
    return [sys.executable, "-m", "vision", *(a for a in argv[1:] if a not in RESTART_DROPPED)]


def watch_keys(hub: "Hub", on_clear):
    """`r` asks for a graceful restart and `c` clears the window (on_clear redraws the one-line
    banner), read a key at a time off a terminal. Returns what puts the terminal back as it was
    (a no-op when stdin is not a terminal)."""
    if not sys.stdin.isatty() or os.name == "nt":
        return lambda: None
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # keys arrive one at a time and are not echoed; Ctrl-C still stops the server

    def restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except termios.error:
            pass

    def loop() -> None:
        while True:
            try:
                key = os.read(fd, 1).decode(errors="ignore").lower()
            except OSError:
                return
            if not key:
                return
            if key == "r":
                if hub.loop is not None:
                    hub.loop.call_soon_threadsafe(hub.request_restart, "r pressed")
            elif key == "c":
                sys.stdout.write("\033[2J\033[3J\033[H")
                sys.stdout.flush()
                on_clear()

    threading.Thread(target=loop, daemon=True, name="serve-keys").start()
    return restore


def serve(cfg: Config, brain, info: dict, log=print, follow_defaults: bool = True,
          open_initial: bool = False, on_clear=None) -> None:
    import uvicorn

    from vision.remote import remove_serve_descriptor, write_serve_descriptor

    hub = Hub(cfg, brain, info["token"], log=log, follow_defaults=follow_defaults,
              open_initial=open_initial, journal_enabled=True)
    restore_keys = watch_keys(hub, on_clear or (lambda: None))

    def exec_restart() -> None:
        """Become a fresh `vision serve` on the new code: same terminal, same port (the listening
        socket is not inherited), chats back from the journal. The phone just reconnects."""
        hub.cancel()
        restore_keys()
        remove_serve_descriptor()
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, restart_argv(sys.argv))

    hub.exec_restart = exec_restart
    try:
        write_serve_descriptor(info["host"], info["port"])
    except OSError as e:
        log(f"terminals cannot find this server (serve.json): {e}")
    try:
        uvicorn.run(create_app(hub), host=info["host"], port=info["port"], log_level="warning", ws_ping_interval=20, ws_ping_timeout=20)
    finally:
        restore_keys()
        remove_serve_descriptor()

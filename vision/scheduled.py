"""Scheduled tasks: a prompt `vision serve` runs on its own at a set time, each run in a fresh chat.

Tasks live in ~/.local/share/vision/scheduled.json and only fire while `vision serve` is running;
one that falls due while it is down runs late if it is less than CATCH_UP old, otherwise it moves on
to its next time (a one-off is switched off). Times are local wall-clock times.

A task: {id, title, prompt, repeat: once|daily|weekdays|weekly, time: "HH:MM", date: "YYYY-MM-DD"
(once), weekday: 0-6 (weekly, Monday = 0), enabled, created, last_run, next_run, last_chat}.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import threading
import time
from pathlib import Path

STORE = Path.home() / ".local" / "share" / "vision" / "scheduled.json"
REPEATS = ("once", "daily", "weekdays", "weekly")
CATCH_UP = 30 * 60  # a run missed by less than this still happens when the server comes back


def _clock(text: str) -> tuple[int, int]:
    hh, mm = (int(x) for x in str(text).split(":")[:2])
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError("time must be HH:MM")
    return hh, mm


def next_run(task: dict, after: float) -> float | None:
    """The first time strictly after `after` this task is due, or None (a one-off already past)."""
    hh, mm = _clock(task.get("time") or "09:00")
    repeat = task.get("repeat") or "once"
    now = dt.datetime.fromtimestamp(after)
    if repeat == "once":
        day = dt.date.fromisoformat(task.get("date") or now.date().isoformat())
        at = dt.datetime.combine(day, dt.time(hh, mm)).timestamp()
        return at if at > after else None
    day = now.date()
    for _ in range(9):
        at = dt.datetime.combine(day, dt.time(hh, mm))
        ok = (repeat == "daily" or (repeat == "weekdays" and at.weekday() < 5)
              or (repeat == "weekly" and at.weekday() == int(task.get("weekday", 0))))
        if ok and at.timestamp() > after:
            return at.timestamp()
        day += dt.timedelta(days=1)
    return None


def clean(body: dict, old: dict | None = None) -> dict:
    """A task from the phone's form, checked; `old` supplies what the edit leaves out. Raises ValueError."""
    task = dict(old or {})
    for key in ("title", "prompt", "repeat", "time", "date", "weekday", "enabled"):
        if key in body:
            task[key] = body[key]
    task["prompt"] = str(task.get("prompt") or "").strip()
    if not task["prompt"]:
        raise ValueError("a scheduled task needs a prompt")
    task["title"] = str(task.get("title") or "").strip()[:80] or task["prompt"].splitlines()[0][:60]
    if task.get("repeat") not in REPEATS:
        raise ValueError(f"repeat must be one of {', '.join(REPEATS)}")
    _clock(task.get("time") or "")
    if task["repeat"] == "once":
        dt.date.fromisoformat(str(task.get("date") or ""))
    if task["repeat"] == "weekly":
        task["weekday"] = int(task.get("weekday", 0)) % 7
    task["enabled"] = bool(task.get("enabled", True))
    task.setdefault("id", secrets.token_hex(4))
    task.setdefault("created", time.time())
    task.setdefault("last_run", 0.0)
    task.setdefault("last_chat", "")
    task["next_run"] = next_run(task, time.time()) if task["enabled"] else None
    if task["next_run"] is None:
        task["enabled"] = False
    return task


class Schedule:
    """The task list on disk. Every method is safe to call from any thread."""

    def __init__(self, path: Path = STORE):
        self.path = path
        self._lock = threading.Lock()
        self._tasks: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [t for t in data if isinstance(t, dict) and t.get("id")] if isinstance(data, list) else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._tasks, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self) -> list[dict]:
        with self._lock:
            return sorted((dict(t) for t in self._tasks), key=lambda t: (not t.get("enabled"), t.get("next_run") or 0))

    def upsert(self, body: dict) -> dict:
        with self._lock:
            tid = body.get("id")
            idx = next((i for i, t in enumerate(self._tasks) if t["id"] == tid), None) if tid else None
            task = clean(body, self._tasks[idx] if idx is not None else None)
            if idx is None:
                self._tasks.append(task)
            else:
                self._tasks[idx] = task
            self._save()
            return dict(task)

    def delete(self, tid: str) -> bool:
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t["id"] != tid]
            if len(self._tasks) != before:
                self._save()
            return len(self._tasks) != before

    def get(self, tid: str) -> dict | None:
        with self._lock:
            return next((dict(t) for t in self._tasks if t["id"] == tid), None)

    def due(self, now: float | None = None) -> list[dict]:
        """Tasks to run now. Each is moved on to its next time (and saved) before it is handed back, so a
        crash mid-run never fires it twice; one missed by more than CATCH_UP is skipped."""
        now = time.time() if now is None else now
        out = []
        with self._lock:
            changed = False
            for t in self._tasks:
                at = t.get("next_run")
                if not t.get("enabled") or not at or at > now:
                    continue
                t["next_run"] = next_run(t, now)
                if t["next_run"] is None:
                    t["enabled"] = False
                changed = True
                if now - at <= CATCH_UP:
                    t["last_run"] = now
                    out.append(dict(t))
            if changed:
                self._save()
        return out

    def ran(self, tid: str, chat_id: str) -> None:
        with self._lock:
            for t in self._tasks:
                if t["id"] == tid:
                    t["last_chat"] = chat_id
                    t["last_run"] = time.time()
                    self._save()
                    return

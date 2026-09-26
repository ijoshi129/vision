"""The subagent rows of past replies, kept with the conversation so they are still there after a
restart, a reconnect or on the other device.

Claude Code's transcript has the Agent calls but not a workflow's agents, and Codex and Grok keep
neither, so Vision keeps its own: one JSON line per reply that ran agents, the rows as agent frames
(vision.brain.agent_frame), in ~/.local/share/vision/agents/<provider>-<session id>.jsonl. A reply
is found again by the start of its text (or, for a reply with no text, by the message before it).
"""

from __future__ import annotations

import json
import re
import threading
import time

from vision.config import DATA_DIR

AGENTS_DIR = DATA_DIR / "agents"
KEY_CHARS = 160  # how much of a reply's start identifies it
_lock = threading.Lock()


def _path(provider: str, session_id: str):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{provider or 'claude'}-{session_id}")
    return AGENTS_DIR / f"{safe}.jsonl"


def _key(text: str) -> str:
    return " ".join((text or "").split())[:KEY_CHARS]


def record(provider: str, session_id: str | None, user_text: str, reply: str, agents: list[dict]) -> None:
    """Keep one reply's agent rows. Rows still running are saved as cut off: nothing will finish them."""
    if not session_id or not agents:
        return
    rows = []
    for frame in agents:
        row = {k: v for k, v in frame.items() if k != "chat"}
        if not row.get("done"):
            row.update({"done": True, "failed": True, "cut_off": True})
        rows.append(row)
    line = json.dumps({"t": time.time(), "user": _key(user_text), "reply": _key(reply), "agents": rows})
    try:
        with _lock:
            AGENTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_path(provider, session_id), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


def load(provider: str, session_id: str) -> list[dict]:
    try:
        with open(_path(provider, session_id), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("agents"):
            out.append(entry)
    return out


def attach(history: list[dict], provider: str, session_id: str) -> list[dict]:
    """Give each assistant message of `history` the agent rows kept for it (an `agents` list),
    matching kept replies to messages in order. Messages that already carry rows keep them."""
    kept = load(provider, session_id) if session_id else []
    if not kept or not history:
        return history
    start = 0
    for entry in kept:
        reply, user = entry.get("reply") or "", entry.get("user") or ""
        for i in range(start, len(history)):
            m = history[i]
            if m.get("role") != "assistant":
                continue
            text = _key(m.get("text") or "")
            before = next((history[j] for j in range(i - 1, -1, -1) if history[j].get("role") == "user"), None)
            if (reply and text.startswith(reply[:KEY_CHARS]) and text) or (
                not reply and before is not None and _key(before.get("text") or "").endswith(user[-80:]) and user
            ):
                if not m.get("agents"):
                    m["agents"] = entry["agents"]
                start = i + 1
                break
    return history


def assistant_entry(reply: str, error: str, agents: list[dict] | None = None) -> dict:
    """A transcript row that keeps a failed turn's error, so a /history refetch (the phone does one
    right after `done`) does not replace the failure bubble with an empty reply, and the reply's
    subagent rows (`agents`, agent frames with `at`)."""
    entry = {"role": "assistant", "text": reply}
    if error and error != "cancelled":
        entry["error"] = error
    if agents:
        entry["agents"] = [{k: v for k, v in a.items() if k != "chat"} for a in agents]
    return entry


def turn_entries(user_text: str, reply: str, error: str, partial: str, agents: list[dict], steers: list[tuple]) -> list[dict]:
    """One turn's transcript rows: the message and its reply, the reply split where messages were
    sent into it (steered), each part with the agent rows that started in it (`at` made relative).
    `steers`: (reply characters so far, text, how many agent rows existed then), in order; `agents`
    in the order they started."""
    rows: list[dict] = [{"role": "user", "text": user_text}]
    # The split points count characters of the streamed reply; a reply that arrived otherwise stays whole.
    cuts = [(s[0], s[1], s[2] if len(s) > 2 else len(agents)) for s in steers if 0 <= s[0] <= len(reply)] if reply == partial else []
    bounds = [(0, 0)] + [(at, seen) for at, _, seen in cuts] + [(len(reply), len(agents))]
    for i in range(len(bounds) - 1):
        (lo, first), (hi, end), last = bounds[i], bounds[i + 1], i == len(bounds) - 2
        mine = [{**a, "at": max(0, a.get("at", 0) - lo)} for a in agents[first:end]]
        rows.append(assistant_entry(reply[lo:hi].strip(), error if last else "", mine))
        if not last:
            rows.append({"role": "user", "text": cuts[i][1]})
    if not cuts and reply != partial:
        rows.extend({"role": "user", "text": s[1]} for s in steers)  # still said, if not placed
    return rows


def record_turn(provider: str, session_id: str | None, entries: list[dict]) -> None:
    """Keep each reply part's agent rows with the conversation (vision.agentlog)."""
    user = ""
    for row in entries:
        if row.get("role") == "user":
            user = row.get("text") or ""
        elif row.get("agents"):
            record(provider, session_id, user, row.get("text") or "", row["agents"])

"""How long a Claude session's prompt cache stays warm, read from Claude Code's own transcript.

Each API call writes or reads the cached prefix, and either one restarts its clock at the tier it
was written with: 5 minutes, or an hour (Claude Code asks for the hour on subscriptions and drops to
5 minutes in overage). The call's `usage.cache_creation` names the tier. The clock starts when the
request goes out, which the transcript does not record; the entry just before the reply (the prompt,
a tool result, an attachment) is written a moment before it, so that is the anchor. The reply's
own timestamp comes after the thinking and would run the clock long by as much as a minute.

Reading the transcript rather than Vision's stream means the figure holds after a restart and
counts turns another Vision (the phone's server, the terminal) made on the same session.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

from vision.sessions import CLAUDE_PROJECTS, _parse_iso

TAIL_BYTES = 512 * 1024  # enough for the last few calls; the whole file is read if this misses them
DEFAULT_TTL = 300  # the API's own default, for a write that does not say its tier


@dataclass(frozen=True)
class CacheState:
    expires: float  # epoch seconds the prefix goes cold
    ttl: int  # the tier, in seconds
    model: str  # the model the prefix is cached for (another model starts cold)


def _tier(usage: dict) -> int | None:
    """The tier this call wrote at, None for a pure cache read. The shorter one if it wrote both:
    the conversation's tail is the part the next call needs."""
    split = usage.get("cache_creation")
    if isinstance(split, dict):
        tiers = [ttl for key, ttl in (("ephemeral_5m_input_tokens", 300), ("ephemeral_1h_input_tokens", 3600)) if split.get(key)]
        if tiers:
            return min(tiers)
    return DEFAULT_TTL if usage.get("cache_creation_input_tokens") else None


def parse(lines: list[bytes]) -> CacheState | None:
    """The cache state after the last main-thread model call in these transcript lines."""
    anchor = None  # timestamp of the newest entry so far
    last = None  # (model, anchor) of the newest reply: the anchor as it stood at its first block
    tier = None
    reply_id = None
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("isSidechain"):
            continue
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        if ev.get("type") == "assistant" and msg.get("usage") and msg.get("model") not in (None, "<synthetic>"):
            if msg.get("id") != reply_id or last is None:  # one entry per content block: the first one starts the reply
                reply_id, last = msg.get("id"), (msg["model"], anchor)
            tier = _tier(msg["usage"]) or tier
        stamp = _parse_iso(ev.get("timestamp") or "")
        if stamp is not None:
            anchor = stamp
    if last is None or last[1] is None:
        return None
    ttl = tier or DEFAULT_TTL
    return CacheState(last[1] + ttl, ttl, last[0])


_memo: dict[str, tuple[tuple[float, int], CacheState | None]] = {}


def _path(session_id: str) -> str | None:
    paths = glob.glob(os.path.join(CLAUDE_PROJECTS, "*", f"{session_id}.jsonl"))
    return max(paths, key=os.path.getmtime) if paths else None


def cache_state(session_id: str | None) -> CacheState | None:
    """The session's cache state, None before its first reply. Cheap enough for every repaint: the
    file is only read again when it changes."""
    if not session_id:
        return None
    path = _path(session_id)
    if not path:
        return None
    try:
        st = os.stat(path)
        key = (st.st_mtime, st.st_size)
        if (memo := _memo.get(path)) and memo[0] == key:
            return memo[1]
        with open(path, "rb") as f:
            f.seek(max(0, st.st_size - TAIL_BYTES))
            lines = f.read().splitlines()
            if st.st_size > TAIL_BYTES:
                lines = lines[1:]  # the first line is cut
        state = parse(lines)
        if state is None and st.st_size > TAIL_BYTES:
            with open(path, "rb") as f:
                state = parse(f.read().splitlines())
    except OSError:
        return None
    _memo[path] = (key, state)
    return state


def _base(model: str | None) -> str:
    return (model or "").split("[")[0]


def seconds_left(session_id: str | None, model: str | None, now: float) -> float | None:
    """Seconds until the session's cache goes cold for `model` (0 once it has, or when the last
    reply came from another model), None when there is nothing to time. A model given as an alias
    (`opus`) can't be matched against the transcript's id, so it is taken as the same one."""
    state = cache_state(session_id)
    if state is None:
        return None
    if model and _base(model).startswith("claude-") and _base(model) != _base(state.model):
        return 0.0
    return max(0.0, state.expires - now)


def label(left: float | None) -> str:
    """`cache 58:12`, `cache cold`, or "" when there is nothing to time."""
    if left is None:
        return ""
    if left <= 0:
        return "cache cold"
    s = int(left)
    return f"cache {s // 60}:{s % 60:02d}"

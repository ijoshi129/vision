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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from vision.config import DATA_DIR, STATE_DIR, WORKSPACE_DIR, BrainConfig, weather_ready
from vision import cachettl, clis, models
from vision.persona import system_prompt
from vision.reply import READING, THINKING, dedupe_status, retry_label

_RETRY_REASONS = {529: "overloaded", 429: "rate limited", 500: "API error", 502: "API error", 503: "API error", 504: "API error"}


def _retry_reason(status: int | None, message: str) -> str:
    """A short reason for the retry line: a known HTTP status, else the error's first words."""
    if status in _RETRY_REASONS:
        return _RETRY_REASONS[status]
    if not message:
        return "connection lost" if status is None else f"API error {status}"
    return message.strip().splitlines()[0][:40]


LAST_SESSION_FILE = STATE_DIR / "last_session"
USAGE_FILE = STATE_DIR / "usage.json"
WINDOWS_FILE = STATE_DIR / "claude_windows.json"  # model id → context window Claude Code last reported


def _saved_windows() -> dict[str, int]:
    try:
        data = json.loads(WINDOWS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return {k: int(v) for k, v in data.items() if isinstance(v, int) and v > 0} if isinstance(data, dict) else {}


def _save_window(model_id: str, window: int) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        WINDOWS_FILE.write_text(json.dumps({**_saved_windows(), model_id: window}))
    except OSError:
        pass


class BrainError(RuntimeError):
    pass


# -- plan mode ---------------------------------------------------------------
# Claude Code's plan mode: read-only tools until the model calls ExitPlanMode, which reaches us as a
# permission request carrying the plan. Approving it lets the same turn carry on with every tool,
# and Vision switches to auto mode for the turns after it.
PLAN_QUESTION = {
    "question": "Carry out this plan?",
    "header": "Plan",
    "multiSelect": False,
    "options": [
        {"label": "Yes", "description": "switch to auto mode and start"},
        {"label": "No", "description": "stay in plan mode; type what should change instead"},
    ],
}


def local_handoff(brain) -> str | None:
    """Read the full saved conversation text. None if there is nothing to carry.

    Used when the new model cannot resume the old session (Claude ↔ Codex ↔ Grok). Does not call the
    outgoing model, so a rate-limit on that side cannot wipe the conversation.
    """
    sid = getattr(brain, "session_id", None)
    if not sid:
        return None
    from vision.sessions import format_transcript, session_history

    text = format_transcript(session_history(brain.provider, sid, limit=0, include_context=True))
    return text or None


def inject_handoff(brain, prompt: str) -> str:
    """Prepend the previous provider's transcript to the first message of a fresh session."""
    note = getattr(brain, "handoff", None)
    if not note or brain.session_id:
        return prompt
    return (
        "You are taking over an ongoing conversation from another model. What happened so far:\n\n"
        f"{note}\n\n---\n\nContinue naturally from here; do not mention the handoff unless asked. "
        f"The user's next message:\n\n{prompt}"
    )


_SHIM = """#!/bin/sh
# Installed by Vision. A shell command inside a Vision turn must not start another Claude Code, Codex
# or Grok run: that would spend a subscription without the user knowing. The user switches brains with /model.
echo "blocked by Vision: '$(basename "$0")' cannot be run from inside a Vision turn (it would spend a subscription behind the user's back). Ask the user to switch with /model instead." >&2
exit 1
"""

_SHIM_NAMES = ("claude", "codex", "grok")
_PROVIDER_HOME = {
    "claude": "CLAUDE_CONFIG_DIR",
    "codex": "CODEX_HOME",
    "grok": "GROK_HOME",
}


def _shim_dir() -> str:
    """Directory holding fake `claude`, `codex` and `grok` commands; created/refreshed on demand ("" if it cannot be)."""
    d = DATA_DIR / "shims"
    try:
        d.mkdir(parents=True, exist_ok=True)
        for name in _SHIM_NAMES:
            f = d / name
            if not f.exists() or f.read_text() != _SHIM:
                f.write_text(_SHIM)
            f.chmod(0o755)
    except OSError:
        return ""
    return str(d)


def brain_env(provider: str) -> dict[str, str]:
    """Environment for a brain subprocess (the model's shell inherits it).

    Drops a parent Claude Code session's nesting guard, puts shims for `claude`, `codex` and `grok`
    first on PATH so a shell command cannot start another run, and points every *other* provider's
    CLI at an empty config dir so it has no credentials even if the shim is bypassed (a profile that
    reorders PATH, an absolute path). Cross-provider work only ever happens through /model."""
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    shims = _shim_dir()
    if shims:
        env["PATH"] = shims + os.pathsep + env.get("PATH", "")
    empty = DATA_DIR / "no-credentials"
    try:
        empty.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    for other, var in _PROVIDER_HOME.items():
        if other != provider:
            env[var] = str(empty)
    if provider == "grok":
        env["GROK_MEMORY"] = "0"  # Vision's MEMORY.md is the shared long-term store
        env["GROK_DISABLE_AUTOUPDATER"] = "1"
    return env


def find_claude() -> str:
    exe = shutil.which("claude")
    if exe:
        return exe
    for cand in (os.path.expanduser("~/.local/bin/claude"), "/usr/local/bin/claude", "/usr/bin/claude"):
        if os.path.exists(cand):
            return cand
    raise BrainError("Claude Code CLI ('claude') not found on PATH. Install it and run `claude` once to log in.")


def _sum_usage(per_message: dict[str, dict]) -> dict | None:
    """Claude Code's per-call usage dicts added up (the `result` event's total, when a turn never got
    one). None when nothing was counted, so the footer shows no numbers rather than zeros."""
    if not per_message:
        return None
    total: dict[str, int] = {}
    for usage in per_message.values():
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    return total


@dataclass
class Turn:
    text: str = ""
    session_id: str | None = None
    tools_used: list[str] = field(default_factory=list)
    is_error: bool = False
    error: str = ""
    cost_usd: float | None = None
    duration_ms: int | None = None
    model: str | None = None  # full model id Claude Code reported for this turn
    usage: dict | None = None  # this turn's token counts in the provider's own keys; None when unknown
    plan_approved: bool = False  # plan mode: the user approved the plan; Vision is in auto mode from here on
    agents: list[AgentRun] = field(default_factory=list)  # subagents the model spawned this turn, in start order
    tools: list[ToolCall] = field(default_factory=list)  # the main conversation's tool calls this turn, in order
    data: dict | None = None  # validated by the delegation boundary; never sent to speech directly


# -- subagents ----------------------------------------------------------------
# The Agent tool runs a nested Claude with its own context; its work only ever came back to us as one
# tool result. In the stream it is visible though: `system/task_started` opens it, every message the
# child sends or receives carries `parent_tool_use_id` (its Agent call), and `system/task_notification`
# closes it with a summary. AgentRun is that view, updated live through Brain.ask's on_agent.
@dataclass
class AgentRun:
    id: str  # tool_use_id of the Agent call
    kind: str  # subagent type: Explore, Plan, general-purpose… (or "agent": the voice model's task worker)
    label: str  # the one-line description the model gave the task
    steps: list[tuple[str, str]] = field(default_factory=list)  # (tool name, one-line detail) as the child calls them
    done: bool = False
    failed: bool = False
    cut_off: bool = False  # never finished: the turn ended, was cancelled or stalled while it ran (failed is set too)
    tool_uses: int = 0
    duration_ms: int = 0
    summary: str = ""  # the child's final report (what the model gets as the tool result)
    # The row says which brain is spending: its model and effort. A Claude subagent runs at its
    # parent's effort (it has no knob of its own) and, unless the Agent call names a model, on the
    # parent's model. The voice worker also shows the tokens it has produced so far while it runs
    # (`tokens_fn`, polled by the UI) and its usage once done.
    model: str = ""
    effort: str = ""
    started: float = 0.0  # time.monotonic() when it started; 0 = unknown (no live timer)
    tokens_fn: Callable[[], int] | None = None
    usage: dict | None = None  # the provider's own token counts for the run, once known
    # The supervisor's report once a launched agent stops (vision/supervisor.py): its state, the
    # result's summary, what changed, what was checked, what is open. Shown under the row.
    details: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """The done tail: time and tokens, `2.3s · ↑12,340 ↓1,204`; the red ✗ says it failed.
        A Claude subagent row, which has no usage of its own, shows its tool count first."""
        bits = []
        if not self.usage:
            n = self.tool_uses or len(self.steps)
            bits.append(f"{n} tool{'s' if n != 1 else ''}")
        bits.append(f"{self.duration_ms / 1000:.1f}s")
        if self.usage:
            bits.append(usage_summary(self.usage))
        return " · ".join(bits)


@dataclass
class ToolCall:
    """One tool call of the main conversation, shown live in the reply the way Claude Code shows its
    own: `⏺ Bash(git status)` and, once the result is back, its length under a `⎿` (a click on the
    row shows the whole result). Made when
    the call starts streaming (name only), given its detail when the full input arrives in the
    `assistant` message, and closed by the `user` message that carries the tool_result."""

    id: str  # tool_use_id
    name: str
    detail: str = ""  # tool_detail() of the input: the command, the path, the pattern…
    done: bool = False
    is_error: bool = False
    output: str = ""  # the text the model got back (the UI shows it on a click)


def context_figure(brain) -> dict | None:
    """How full the chat's context is, `{"tokens": read, "window": size}`, for the phone's gauge
    and its switch-model warning. None for brains that don't report it (Codex, Grok) or before
    the first turn."""
    ctx = getattr(brain, "context", None)
    if not ctx or not ctx[1]:
        return None
    return {"tokens": int(ctx[0]), "window": int(ctx[1])}


def cache_figure(brain) -> dict | None:
    """When the chat's prompt cache goes cold, `{"expires": epoch seconds, "ttl": 300 or 3600}`, for
    the phone to count down on its own clock. Claude only (the others don't promise a lifetime),
    and None before the session's first reply. Expired already (or cached for another model): 0."""
    if getattr(brain, "provider", "") != "claude":
        return None
    state = cachettl.cache_state(brain.session_id)
    if state is None:
        return None
    left = cachettl.seconds_left(brain.session_id, brain.resolved_model(), time.time())
    return {"expires": state.expires if left else 0, "ttl": state.ttl}


def usage_summary(usage: dict) -> str:
    """`↑12,340 ↓1,204`: what a run read (fresh plus cached input) and wrote."""
    return f"↑{context_read(usage):,} ↓{usage.get('output_tokens', 0):,}"


def context_read(usage: dict) -> int:
    """How much the model read on one call: its fresh input plus what came from the cache. After the
    last call of a turn this is the size of the conversation as the model sees it."""
    return usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)


def result_text(content) -> str:
    """The text of a tool_result's content: a plain string, or the text blocks of a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


STEP_WIDTH = 96  # longest tool detail kept per step (the UI trims further to fit the terminal)


def tool_detail(name: str, args: dict) -> str:
    """One line saying what a tool call does, from its input: the command for Bash, the path for
    file tools, the pattern for searches, the task for a nested Agent. Empty if nothing fits."""
    if not isinstance(args, dict):
        return ""
    if name == "Bash":
        detail = args.get("description") or args.get("command") or ""
    elif name in ("Read", "Edit", "Write", "NotebookEdit"):
        detail = args.get("file_path") or args.get("notebook_path") or ""
    elif name in ("Grep", "Glob"):
        detail = args.get("pattern") or ""
        if args.get("path"):
            detail += f"  in {args['path']}"
    elif name in ("Agent", "Task"):
        detail = args.get("description") or args.get("prompt") or ""
    elif name in ("WebSearch", "WebFetch"):
        detail = args.get("query") or args.get("url") or ""
    else:
        detail = next((v for v in args.values() if isinstance(v, str) and v.strip()), "")
    home = os.path.expanduser("~")
    detail = str(detail).strip().splitlines()[0] if str(detail).strip() else ""
    detail = detail.replace(home + "/", "~/").replace(home, "~")
    return detail if len(detail) <= STEP_WIDTH else detail[: STEP_WIDTH - 1] + "…"


def claude_default_model() -> str | None:
    """The model Claude Code itself would use when we pass no --model (env, then settings.json)."""
    env = os.environ.get("ANTHROPIC_MODEL")
    if env:
        return env
    for path in (
        os.path.expanduser("~/.claude/settings.local.json"),
        os.path.expanduser("~/.claude/settings.json"),
    ):
        try:
            with open(path) as f:
                model = json.load(f).get("model")
        except (OSError, ValueError, AttributeError):
            continue
        if model:
            return str(model)
    return None


class Brain:
    provider = "claude"

    def __init__(self, cfg: BrainConfig, voice_mode: bool = False, session_id: str | None = None):
        self.cfg = cfg
        self.voice_mode = voice_mode
        self.task_mode = False  # a separate worker session, never a typed conversation
        self.session_id = session_id
        self._proc: subprocess.Popen | None = None
        self._cancelled = False  # cancel() was called mid-turn (from another thread: Esc in the chat screen)
        self._lock = threading.Lock()
        self._plan_approved = False  # this turn's plan was approved: allow the tools that follow
        self.claude = find_claude()
        self.workdir = os.path.abspath(os.path.expanduser(cfg.workdir)) if cfg.workdir else os.getcwd()
        self.last_usage: dict | None = None
        # Live figures for the chat screen: output tokens received so far this turn (the `↓ 1.2k`
        # on the reply's live line) and how full the context is after the last model call, as
        # (tokens the model read, its window) for the `34% ctx` in the status row.
        self.output_tokens = 0
        self.context: tuple[int, int] | None = None
        self._window = 0  # context window Claude Code reported for the model, once seen
        self._window_for: str | None = None  # the cfg.model it was reported under
        # Actual model id Claude Code reported on the last turn, and the cfg.model it was observed under.
        self.model: str | None = None
        self._model_seen_for: str | None = None
        # Previous provider's transcript, sent with the first message of the next fresh session.
        self.handoff: str | None = None
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    def context_window(self) -> int:
        """The model's context window in tokens: what Claude Code reported for the current model
        setting in a `result` event's modelUsage, else what it last reported for that model id
        (saved across runs), else 200k for Haiku and 1M for the rest (what Claude Code gives them)."""
        if self._window and self._window_for == self.cfg.model:
            return self._window
        model_id = (self.resolved_model() or "").split("[")[0]
        saved = _saved_windows().get(model_id)
        if saved:
            return saved
        return 200_000 if "haiku" in model_id or "haiku" in (self.cfg.model or "") else 1_000_000

    def _note_window(self, model_usage) -> None:
        """Take the window of the chat's own model from modelUsage, which also lists helper models
        Claude Code ran on the side (Haiku, 200k), in no promised order."""
        if not isinstance(model_usage, dict):
            return
        windows = {k.split("[")[0]: int(v["contextWindow"]) for k, v in model_usage.items()
                   if isinstance(v, dict) and v.get("contextWindow")}
        if not windows:
            return
        own = (self.model or "").split("[")[0] if self._model_seen_for == self.cfg.model else ""
        model_id, window = (own, windows[own]) if own in windows else max(windows.items(), key=lambda kv: kv[1])
        self._window, self._window_for = window, self.cfg.model
        if self.context:
            self.context = (self.context[0], window)
        if _saved_windows().get(model_id) != window:
            _save_window(model_id, window)

    def resolved_model(self) -> str | None:
        """Best knowledge of the model actually in use: what Claude Code reported for the current
        model setting, else the setting itself, else Claude Code's own default. None if unknown."""
        if self.model and self._model_seen_for == self.cfg.model:
            return self.model
        return self.cfg.model or claude_default_model()

    # -- session helpers -------------------------------------------------
    @staticmethod
    def last_session_id() -> str | None:
        try:
            sid = LAST_SESSION_FILE.read_text().strip()
            return sid or None
        except FileNotFoundError:
            return None

    def _remember_session(self) -> None:
        if self.task_mode:
            return  # a voice worker must not replace the typed --continue target
        if self.session_id:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            LAST_SESSION_FILE.write_text(self.session_id)

    def new_session(self) -> None:
        self.session_id = None
        self.handoff = None
        self.context = None  # the `% ctx` figure belongs to the session it was measured in

    def resume(self, session_id: str) -> None:
        """Continue an earlier conversation from the next turn on (and make it the `-c` target)."""
        self.session_id = session_id
        self.handoff = None
        self.context = None
        self._remember_session()

    # -- main entry point -------------------------------------------------
    def _command(self) -> list[str]:
        from vision.delegation import RESULT_SCHEMA, worker_prompt

        persona = (worker_prompt(self.cfg, self.workdir, self.provider) if self.task_mode else
                   system_prompt(self.voice_mode, self.cfg.address_user_as, self.workdir, self.cfg.allowed_tools, mode=self.cfg.mode,
                                 weather=weather_ready(self.cfg)))
        cmd = [
            self.claude,
            "-p",
            "--output-format", "stream-json",
            "--input-format", "stream-json",  # keeps stdin open: AskUserQuestion answers go back on it
            "--permission-prompt-tool", "stdio",  # ... as control_response replies to control_request
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            persona,
        ]
        if self.task_mode:
            cmd += ["--json-schema", json.dumps(RESULT_SCHEMA)]
        if self.cfg.model:
            cmd += ["--model", self.cfg.model]
        if self.cfg.effort:
            cmd += ["--effort", self.cfg.effort]  # "ultracode" is accepted as-is: xhigh + workflow orchestration
        # A flag-scoped setting keeps Vision's /fast state local to this session instead of changing
        # Claude Code's user-wide preference.
        cmd += ["--settings", json.dumps({"fastMode": self.cfg.fast})]
        if self.cfg.mode == "plan":
            # Plan mode: Claude Code's own read-only mode (reads, searches and read-only shell commands
            # run freely; edits wait for the plan to be approved — see _answer_control).
            cmd += ["--permission-mode", "plan"]
        else:
            # Auto mode: the full tool set, nothing needs approval. Deny patterns below still apply.
            cmd += ["--dangerously-skip-permissions"]
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
        on_question: Callable[[list[dict]], dict[str, str] | None] | None = None,
        on_agent: Callable[[AgentRun], None] | None = None,
        on_tool: Callable[[ToolCall], None] | None = None,
    ) -> Turn:
        """Send one user turn. Streams text deltas to on_text; returns the completed Turn.
        on_status gets a tool name when a tool starts, `reading` once every tool of that batch has
        finished, and `thinking` when a real reasoning block starts (see vision.reply: repeats are
        deduped, `status_label` gives the words).
        on_question gets AskUserQuestion's `questions` list and blocks until it returns
        {question: answer} (multi-select answers comma-joined) or None for "cancelled".
        on_tool gets the ToolCall of a main-conversation tool call each time it changes: when it
        starts, when its input is complete, and when its result is back (subagents' own calls go
        to on_agent instead).
        on_agent gets the AgentRun of a subagent each time it changes: when it starts, after every
        tool call it makes, and when it finishes (`done`). The same object is passed each time."""
        env = brain_env("claude")
        turn = Turn(session_id=self.session_id)
        prompt = inject_handoff(self, prompt)
        streamed = []
        final_text_from_message = []
        saw_delta = False
        self._plan_approved = False
        self._cancelled = False
        on_status = dedupe_status(on_status)
        pending: set[str] = set()  # top-level tool calls still running (status goes to `reading` at zero)
        seen_usage: dict[str, dict] = {}  # token counts per model call so far, by message id (the tally a cancelled turn reports)
        self.output_tokens = 0
        got_result = False  # the `result` event arrived: the turn ran to its end
        agents: dict[str, AgentRun] = {}  # by the Agent call's tool_use_id
        tools: dict[str, ToolCall] = {}  # the main conversation's tool calls, by tool_use_id
        child_tools: set[str] = set()  # subagents' tool calls (their rows are AgentRun.steps), by tool_use_id
        agent_models: dict[str, str] = {}  # model override seen on an Agent call before its AgentRun exists

        def agent_changed(run: AgentRun) -> None:
            if on_agent:
                on_agent(run)

        def tool_changed(call: ToolCall) -> None:
            if on_tool:
                on_tool(call)

        def new_run(pid: str, kind: str, label: str) -> AgentRun:
            """A subagent's row: its model override if the Agent call had one, else the model it
            inherits from us; the effort is always ours (a native subagent has no effort knob)."""
            run = agents[pid] = AgentRun(pid, kind, label, model=agent_models.get(pid) or self.cfg.model or "", effort=self.cfg.effort or "",
                                         started=time.monotonic())  # the row's live timer runs from here
            turn.agents.append(run)
            return run

        def agent_for(ev: dict) -> AgentRun | None:
            """The subagent a message belongs to (None for the main conversation). A child seen before
            its task_started, e.g. after a resume, still gets a run so its steps are not lost."""
            pid = ev.get("parent_tool_use_id")
            if not pid:
                return None
            run = agents.get(pid)
            if run is None:
                run = new_run(pid, ev.get("subagent_type") or "agent", ev.get("task_description") or "")
                agent_changed(run)
            return run

        def emit(text: str) -> None:
            """Text we add to the reply ourselves (the plan awaiting approval)."""
            nonlocal saw_delta
            saw_delta = True
            streamed.append(text)
            if on_text:
                on_text(text)

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
        # stderr is read on its own thread: left in the pipe until exit, a chatty claude fills the
        # pipe buffer, blocks on the write, and Vision blocks with it on stdout — a silent deadlock.
        stderr_chunks: list[str] = []
        drain = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read() if proc.stderr else ""), daemon=True)
        drain.start()
        # A stall watchdog: no stdout line for cfg.stall_s kills the process, so a turn cannot wait
        # forever on a hung network call or a dead subagent. A long tool call is quiet too, hence
        # the generous default. Paused while an AskUserQuestion waits on the user.
        activity = [time.monotonic()]
        paused = [False]
        stalled = [False]
        stopped = threading.Event()

        def watchdog() -> None:
            limit = float(self.cfg.stall_s or 0)
            while limit > 0 and not stopped.wait(min(5.0, limit / 4)):
                if not paused[0] and time.monotonic() - activity[0] > limit and proc.poll() is None:
                    stalled[0] = True
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                    return

        threading.Thread(target=watchdog, daemon=True).start()

        def close_leftovers() -> None:
            """Rows the stream never closed (an agent killed with the process when the turn ended,
            a tool call cut short by a cancel or a stall) are marked done here, so they stop spinning."""
            reason = "cancelled" if self._cancelled else f"stalled: no output for {float(self.cfg.stall_s):.0f}s" if stalled[0] else "cut off when the turn ended"
            for run in agents.values():
                if not run.done:
                    run.done = run.failed = run.cut_off = True
                    run.summary = run.summary or reason
                    if run.started and not run.duration_ms:
                        run.duration_ms = int((time.monotonic() - run.started) * 1000)
                    agent_changed(run)
            for call in tools.values():
                if not call.done:
                    call.done = call.is_error = True
                    call.output = call.output or f"({reason})"
                    tool_changed(call)
            pending.clear()

        try:
            assert proc.stdin and proc.stdout
            proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                activity[0] = time.monotonic()
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
                    turn.model = ev.get("model") or turn.model
                    if turn.model:
                        self.model, self._model_seen_for = turn.model, self.cfg.model
                        # A model Claude Code's list does not know yet (it moved an alias or added one): re-read it.
                        if turn.model.split("[")[0] not in {m.model_id for m in models.CLAUDE_MODELS}:
                            clis.refresh_models_soon(("claude",), max_age=clis.CLAUDE_MODELS_RECHECK)
                elif t == "system" and ev.get("subtype") == "api_retry":
                    # The CLI is waiting out an API failure (529, rate limit, dropped stream): nothing else
                    # streams meanwhile, so say so instead of leaving "working…" up.
                    if on_status:
                        err = ev.get("error") or {}
                        message = err.get("message") if isinstance(err, dict) else str(err or "")
                        on_status(retry_label(ev.get("attempt"), ev.get("max_retries"), ev.get("retry_delay_ms"),
                                              _retry_reason(ev.get("error_status"), message or "")))
                elif t == "system" and ev.get("subtype") in ("task_started", "task_notification") and (
                    ev.get("task_type") == "local_bash" or ev.get("tool_use_id") in tools or ev.get("tool_use_id") in child_tools
                ):
                    # A Bash command that runs a few seconds becomes a background task and uses the same
                    # task events as a subagent, but is not one: its row is the tool call already shown
                    # (a subagent's, its step). Skipped, so no `agent · <its description> · 0 tools · 0.0s`
                    # ghost. Claude Code tags these `task_type: local_bash`; the id sets cover versions
                    # that don't, inside a subagent too (its tool_use arrives before the task_started).
                    pass
                elif t == "system" and ev.get("subtype") == "task_started" and ev.get("tool_use_id"):
                    run = agents.get(ev["tool_use_id"])
                    if run is None:
                        run = new_run(ev["tool_use_id"], "", "")
                    run.kind = ev.get("subagent_type") or run.kind or "agent"
                    run.label = ev.get("description") or run.label
                    agent_changed(run)
                elif t == "system" and ev.get("subtype") == "task_notification" and ev.get("tool_use_id") in agents:
                    run = agents[ev["tool_use_id"]]
                    usage = ev.get("usage") or {}
                    run.tool_uses = usage.get("tool_uses") or run.tool_uses
                    run.duration_ms = usage.get("duration_ms") or run.duration_ms
                    run.summary = ev.get("summary") or run.summary
                    run.failed = ev.get("status") not in (None, "completed")
                    run.done = True
                    agent_changed(run)
                elif ev.get("parent_tool_use_id"):
                    # Inside a subagent: its tool calls are the steps shown live; its text is its own
                    # business (the model gets it as the Agent result) and must not leak into the reply.
                    run = agent_for(ev)
                    if t == "assistant" and run is not None:
                        for block in ev.get("message", {}).get("content", []):
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                if block.get("id"):
                                    child_tools.add(block["id"])
                                run.steps.append((block.get("name", "tool"), tool_detail(block.get("name", ""), block.get("input") or {})))
                                agent_changed(run)
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
                            if cb.get("id"):
                                pending.add(cb["id"])
                                if name not in ("Agent", "Task"):  # a subagent has its own row (on_agent)
                                    call = tools[cb["id"]] = ToolCall(cb["id"], name)
                                    turn.tools.append(call)
                                    tool_changed(call)
                            if on_status:
                                on_status(name)
                        elif cb.get("type") in ("thinking", "redacted_thinking"):
                            if on_status:
                                on_status(THINKING)  # a real reasoning block: the one time "thinking…" is the truth
                        elif cb.get("type") == "text" and saw_delta and streamed and not streamed[-1].endswith("\n"):
                            # A new text block after a tool call: keep paragraphs separated.
                            streamed.append("\n\n")
                            if on_text:
                                on_text("\n\n")
                elif t == "user":
                    # Tool results come back as a user message. Once none of the batch is still running
                    # (parallel calls each get their own result), the model is thinking again.
                    results = [b for b in ev.get("message", {}).get("content", []) if isinstance(b, dict) and b.get("type") == "tool_result"]
                    pending.difference_update(b.get("tool_use_id") for b in results)
                    for b in results:
                        call = tools.get(b.get("tool_use_id"))
                        if call is not None:
                            call.output = result_text(b.get("content"))
                            call.is_error = bool(b.get("is_error"))
                            call.done = True
                            tool_changed(call)
                    if on_status and results and not pending:
                        on_status(READING)
                elif t == "control_request":
                    paused[0] = True  # the user may take as long as they like to answer
                    try:
                        self._answer_control(proc, ev, on_question, emit, turn)
                    finally:
                        activity[0] = time.monotonic()
                        paused[0] = False
                elif t == "rate_limit_event":
                    self._record_usage(ev.get("rate_limit_info"))
                elif t == "assistant":
                    msg = ev.get("message", {})
                    if msg.get("usage"):
                        seen_usage[msg.get("id") or str(len(seen_usage))] = msg["usage"]  # one event per content block: keep the last per message
                        self.output_tokens = sum(u.get("output_tokens", 0) for u in seen_usage.values())
                        self.context = (context_read(msg["usage"]), self.context_window())
                    # Fallback source of text when partial messages are unavailable.
                    for block in msg.get("content", []):
                        if block.get("type") == "text":
                            final_text_from_message.append(block["text"])
                        elif block.get("type") == "tool_use" and block.get("id") in tools:
                            call = tools[block["id"]]  # the input streamed as JSON deltas; here it is whole
                            call.detail = tool_detail(call.name, block.get("input") or {})
                            tool_changed(call)
                        elif block.get("type") == "tool_use" and block.get("name") in ("Agent", "Task"):
                            # An Agent/Task call's own input carries the model override, if any (no
                            # effort knob exists for a native subagent, unlike the voice worker's).
                            # Its AgentRun usually doesn't exist yet (task_started follows this
                            # message), so stash it for whichever creation site runs first.
                            model = (block.get("input") or {}).get("model")
                            if model and block.get("id"):
                                agent_models[block["id"]] = model
                                run = agents.get(block["id"])
                                if run is not None:
                                    run.model = model
                                    agent_changed(run)
                elif t == "result":
                    got_result = True
                    self._note_window(ev.get("modelUsage"))
                    turn.session_id = ev.get("session_id") or turn.session_id
                    turn.is_error = bool(ev.get("is_error")) or ev.get("subtype", "").startswith("error")
                    turn.cost_usd = ev.get("total_cost_usd")
                    turn.duration_ms = ev.get("duration_ms")
                    turn.usage = ev.get("usage") or turn.usage
                    if self.task_mode:
                        turn.data = ev.get("structured_output")
                    if turn.is_error:
                        turn.error = ev.get("result") or ev.get("error") or ev.get("subtype", "error")
                    elif not saw_delta and ev.get("result"):
                        final_text_from_message = [ev["result"]]
                    try:
                        proc.stdin.close()  # the turn is over; closing input lets claude exit
                    except OSError:
                        pass
            proc.wait()
            drain.join(timeout=5)
            stderr = "".join(stderr_chunks)
        except KeyboardInterrupt:
            self.cancel()
            turn.is_error = True
            turn.error = "cancelled"
            turn.text = "".join(streamed)
            turn.usage = _sum_usage(seen_usage)
            return turn
        finally:
            stopped.set()
            with self._lock:
                self._proc = None
            close_leftovers()

        if saw_delta:
            turn.text = "".join(streamed)
        else:
            turn.text = "\n\n".join(final_text_from_message)
            if turn.text and on_text:
                on_text(turn.text)

        if (self._cancelled and not got_result) or stalled[0] or (proc.returncode or 0) < 0:
            # Cut short from another thread (or by a signal, or by the stall watchdog): no `result`
            # event, so the token counts are what the model calls so far reported. The session is
            # kept: what was said is in it.
            turn.is_error = True
            turn.error = f"the brain stalled: no output for {float(self.cfg.stall_s):.0f}s" if stalled[0] and not self._cancelled else "cancelled"
            turn.usage = turn.usage or _sum_usage(seen_usage)
            if turn.session_id:
                self.session_id = turn.session_id
                self._remember_session()
        elif proc.returncode not in (0, None) and not turn.text:
            turn.is_error = True
            turn.error = turn.error or (stderr.strip().splitlines() or ["claude exited with code %s" % proc.returncode])[-1]
            if "resume" in turn.error.lower() or "session" in turn.error.lower():
                # Stale session id: drop it so the next turn starts clean.
                self.session_id = None
        else:
            self.session_id = turn.session_id
            self.handoff = None  # delivered with this turn; the session carries it from now on
            self._remember_session()
        return turn

    # -- control channel ------------------------------------------------
    def _answer_control(self, proc: subprocess.Popen, ev: dict, on_question, emit, turn: Turn) -> None:
        """Reply to a `can_use_tool` request. In auto mode only AskUserQuestion reaches us (everything
        else is pre-approved or denied by the flags): show the form, hand the answers back as
        `updatedInput.answers`; a cancelled form denies the call so the model can carry on.
        In plan mode ExitPlanMode brings the finished plan: show it, ask, and once approved allow
        whatever tools the rest of the turn needs."""
        req = ev.get("request", {})
        rid = ev.get("request_id")
        tool = req.get("tool_name", "tool")
        tool_input = req.get("input", {})
        if tool == "AskUserQuestion":
            if self.task_mode:
                msg = {"type": "control_response", "response": {"subtype": "success", "request_id": rid, "response": {
                    "behavior": "deny", "message": "The voice conversation asks the user. Return status needs_input with your question in the structured result instead."
                }}}
                try:
                    proc.stdin.write(json.dumps(msg) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                return
            questions = tool_input.get("questions")
            answers = on_question(questions) if questions and on_question else None
            if answers:
                result = {"behavior": "allow", "updatedInput": {**tool_input, "answers": answers}}
            else:
                result = {"behavior": "deny", "message": "The user dismissed the question form without answering; continue without these answers or ask in plain text."}
        elif tool == "ExitPlanMode":
            plan = (tool_input.get("plan") or "").strip()
            if plan:
                emit(f"\n\n{plan}\n\n")  # blank line after: the text that follows approval is a new paragraph, not the last bullet
            # Worker prose is deliberately suppressed. Put its plan in the approval form so the
            # user can still review the actual changes before approving them.
            question = {**PLAN_QUESTION, "question": f"Carry out this plan?\n\n{plan}"} if self.task_mode and plan else PLAN_QUESTION
            choice = ((on_question([question]) if on_question else None) or {}).get(question["question"], "")
            if choice == "Yes":
                self._plan_approved = turn.plan_approved = True
                self.cfg.mode = "auto"  # the turns after this one run without approval too
                result = {"behavior": "allow", "updatedInput": tool_input}
            elif choice and choice != "No":
                result = {"behavior": "deny", "message": f"The user did not approve the plan and said: {choice}\nStay in plan mode and revise the plan accordingly."}
            else:
                result = {"behavior": "deny", "message": "The user did not approve the plan. Stay in plan mode; ask what they would like changed, or wait."}
        elif self._plan_approved:
            result = {"behavior": "allow", "updatedInput": tool_input}
        elif self.cfg.mode == "plan":
            result = {"behavior": "deny", "message": f"Vision is in plan mode: {tool} has to wait until the plan is approved. Finish the plan and call ExitPlanMode."}
        else:
            result = {"behavior": "deny", "message": f"{tool} is not available in Vision."}
        msg = {"type": "control_response", "response": {"subtype": "success", "request_id": rid, "response": result}}
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass

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

    def usage_report(self) -> str | None:
        """Claude Code's own /usage text (session, week, per-model windows). No model call is made."""
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        cmd = [self.claude, "-p", "/usage", "--output-format", "json", "--no-session-persistence", "--tools", ""]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workdir, env=env, timeout=60)
            data = json.loads(r.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, OSError, ValueError, IndexError):
            return None
        text = data.get("result") or ""
        if data.get("is_error") or "used" not in text:
            return None
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            (STATE_DIR / "usage.txt").write_text(text)
        except OSError:
            pass
        return text

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
            self._cancelled = proc is not None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


def create_brain(
    cfg: BrainConfig,
    voice_mode: bool = False,
    *,
    continue_session: bool = False,
    session_id: str | None = None,
):
    """Construct the driver for cfg.model, optionally resuming that provider's last thread."""
    from vision.models import coerce_effort, provider_for

    clis.refresh_models_soon()  # once per process (then every few hours): new models, labels, efforts
    cfg.effort, _ = coerce_effort(cfg.model, cfg.effort)

    provider = provider_for(cfg.model)
    if provider == "codex":
        from vision.codex import CodexBrain

        cls = CodexBrain
    elif provider == "grok":
        from vision.grok import GrokBrain

        cls = GrokBrain
    elif provider == "local":
        from vision.local import LocalBrain

        cls = LocalBrain
    else:
        cls = Brain
    sid = session_id
    if sid is None and continue_session:
        sid = cls.last_session_id()
    return cls(cfg, voice_mode=voice_mode, session_id=sid)

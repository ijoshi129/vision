"""Configuration for Vision.

Config lives in ~/.config/vision/config.toml (created with defaults on first run).
Speech models and Vision's working directory live under ~/.local/share/vision/.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from vision.models import THINKING_OFF, provider_for

# Windows starts in plan mode: none of the Linux sandboxes exist there, and denied_tools only knows
# POSIX commands. "auto" still works; set it under [brain] or press Shift-Tab.
DEFAULT_MODE = "plan" if sys.platform == "win32" else "auto"
# Windows' counterparts of the default POSIX deny entries: wiping disks, the boot configuration and
# restore points, shutting down, and elevating (the sudo of Windows). Bash rules also reach Claude
# Code's PowerShell tool as PowerShell(...) ones (see Brain._command); the cmdlets only PowerShell
# has are listed as PowerShell rules.
WINDOWS_DENIED = [
    "Bash(format:*)", "Bash(diskpart:*)", "Bash(bcdedit:*)", "Bash(vssadmin delete:*)", "Bash(runas:*)",
    "PowerShell(Format-Volume:*)", "PowerShell(Clear-Disk:*)", "PowerShell(Initialize-Disk:*)",
    "PowerShell(Remove-Partition:*)", "PowerShell(Stop-Computer:*)", "PowerShell(Restart-Computer:*)",
    "PowerShell(Start-Process * -Verb RunAs*)",
]

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vision"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "vision"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "vision"
CONFIG_PATH = CONFIG_DIR / "config.toml"
# Vision's voice and example exchanges; replaces the built-in personality when it has any text.
PERSONALITY_PATH = CONFIG_DIR / "personality.md"
MODELS_DIR = DATA_DIR / "models"
WORKSPACE_DIR = DATA_DIR / "workspace"

LLAMA_DIR = DATA_DIR / "llama"
# Cloned / designed voices for the Qwen3-TTS engine: one folder per voice holding ref.wav (the clip the
# model imitates), ref.txt (its transcript) and, for designed voices, design.txt (the description).
VOICES_DIR = DATA_DIR / "voices"

# Qwen3-TTS 1.7B (Alibaba, Apache 2.0): Base clones any voice from a few seconds of audio; VoiceDesign
# invents a voice from a text description. Both come from the Hugging Face hub into its cache.
QWEN_TTS_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_TTS_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# Built-in voice designs: `vision voice design <name>` uses these when no description is given.
_DESIGN_LINE = (
    "Good afternoon. All systems are online and running within normal parameters. "
    "I've taken the liberty of reviewing today's schedule; nothing that can't be handled with a little finesse."
)
VOICE_DESIGNS = {
    "jarvis": (
        "A refined adult male voice with a distinctly British Received Pronunciation accent: crisp, clipped "
        "English consonants and rounded vowels, like a well-spoken Londoner. Low-to-mid pitch, warm and smooth "
        "timbre, precise articulation at a brisk, efficient conversational pace. Composed and quietly confident, "
        "with the faintest hint of dry wit: a discreet, highly capable assistant who is never flustered.",
        _DESIGN_LINE,
    ),
    "narrator": (
        "A mature British male narrator with a mellifluous, theatrical delivery in Received Pronunciation. "
        "Rich mid-to-low pitch, velvety timbre, immaculate diction. A fast talker: rapid, energetic, fluent "
        "delivery at a quick conversational clip, words tumbling out briskly with no dramatic pauses and no "
        "drawn-out vowels. Warm, wry and faintly amused, with a gently condescending gravitas, as if narrating "
        "a story whose every outcome he already knows.",
        "Stanley walked through the red door, exactly as he had been told. Correct choice, naturally. The narrator "
        "noted, with quiet satisfaction, that for once somebody was actually listening.",
    ),
    "friday": (
        "A calm, clear adult female voice with a light Irish accent. Mid pitch, bright and friendly timbre, "
        "crisp articulation at a relaxed conversational pace. Warm, quick-witted and self-assured, "
        "like a capable assistant who enjoys the work.",
        _DESIGN_LINE,
    ),
}

# Orpheus 3B (Canopy Labs' Llama-based TTS) as a Q4_K_M GGUF, run by llama.cpp's server, plus the SNAC
# codec decoder (ONNX) that turns its audio tokens into 24 kHz sound.
ORPHEUS_MODEL_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
ORPHEUS_MODEL_URL = "https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/resolve/main/orpheus-3b-0.1-ft-q4_k_m.gguf"
SNAC_MODEL_FILE = "snac_24khz_decoder.onnx"
SNAC_MODEL_URL = "https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx"
# Prebuilt llama.cpp (CUDA 13 build for Linux x64; the cudart bundle carries the CUDA runtime it needs).
LLAMA_BUILD = "b11007"
LLAMA_URLS = [
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-ubuntu-cuda-13.3-x64.tar.gz",
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-13.3-x64.tar.gz",
]
if sys.platform == "win32":  # the same build for Windows x64 (zips; the cudart one carries no build number)
    LLAMA_URLS = [
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/cudart-llama-bin-win-cuda-13.4-x64.zip",
    ]

DEFAULT_CONFIG = '''# Vision configuration. Edit freely; run `vision config` to see the resolved values.

[brain]
# Default model (Claude alias, Codex slug, or Grok id) and effort level.
# Claude's top tier is "ultracode"; Codex's is "ultra"; Grok uses its advertised effort list.
# Change them with `vision default` or /default inside a chat; /model and /effort change only the current session.
model = "opus"
effort = "high"
# The tools Vision describes to the model. Claude's auto mode has every tool regardless (only
# denied_tools and plan mode restrict it); Codex maps this list to the sandbox below.
# Bash = shell commands, Read/Glob/Grep = read files, Write/Edit = create and change files.
allowed_tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
# Claude command patterns Vision may never run, even with Bash allowed. Syntax: Bash(prefix:*)
# Codex receives these as prompt guidance; choose its read-only sandbox for mechanical enforcement.
denied_tools = ["Bash(sudo:*)", "Bash(rm -rf /:*)", "Bash(rm -rf ~:*)", "Bash(mkfs:*)", "Bash(dd:*)", "Bash(shutdown:*)", "Bash(reboot:*)", "Bash(claude:*)", "Bash(codex:*)", "Bash(grok:*)"]
# Directory Vision works in. Empty = the directory you launch `vision` from.
workdir = ""
# How you'd like Vision to address you now and then (e.g. your first name). Empty = nothing special.
# Its personality lives in ~/.config/vision/personality.md (see the README); delete that file for the default.
address_user_as = ""
# Starting mode. "auto": every tool runs without approval (denied_tools still apply; Codex gets
# danger-full-access). "plan": read-only — Vision investigates and proposes a plan, and carries it out
# once you approve it (Claude) or switch to auto. Shift-Tab or /mode switches during a chat.
mode = "auto"
# Approving a plan: "session" switches to auto for the rest of the session; "turn" runs only the
# approved plan and goes back to plan mode for your next message.
plan_approval = "session"
# Banked limit resets (/usage) and Grok's allowance come from undocumented endpoints that Vision calls
# with the logins Claude Code (~/.claude/.credentials.json) and Grok (~/.grok/auth.json) saved.
# false = Vision never reads those files; /usage then shows only what the CLIs report themselves.
read_cli_logins = true
# Stall watchdog for Claude turns: seconds without any output from the brain before the turn is
# killed and reported as stalled. Long tool calls still stream events; only true silence counts.
# 0 disables it.
stall_s = 900

[codex]
# Sandbox used for GPT/Codex models in auto mode: "auto" (= danger-full-access, matching Claude's auto
# mode), "workspace-write" (shell and file changes only inside the working directory) or "read-only".
# Plan mode is always read-only.
sandbox = "auto"
# Optional raw `codex exec -c key=value` entries. Most users should leave this empty.
extra_config = []
# How turns run: "app-server" (messages can go into a running reply, text streams in) or "exec"
# (`codex exec --json`, one message per turn, replies arrive whole).
transport = "app-server"

[grok]
# Sandbox used for Grok models in auto mode: "auto" (= off, unrestricted, matching Claude's auto mode),
# "workspace" (write only inside the working directory), "read-only" or "strict".
# Plan mode is always read-only. Grok enforces this with its kernel sandbox.
sandbox = "auto"
# Optional extra `grok` CLI flags for each turn. Most users should leave this empty.
extra_args = []

[local]
# llama-server hosting the Local models (see deploy/local-model): spoken to directly, no CLI, nothing
# leaves the tailnet. Vision runs its tool calls itself (Bash, Read, Write, Edit, WebSearch, WebFetch, as
# listed in [brain].allowed_tools). Point it at the machine running llama-server; a Tailscale name works away from home.
base_url = "http://localhost:8080/v1"
timeout_s = 180
# The server's context size (-c); the conversation is trimmed to stay inside it.
context = 131072
# WebSearch asks DuckDuckGo (no account) unless a Brave Search API key is set here (free tier at
# brave.com/search/api), which is steadier and rate-limited per key rather than per IP.
brave_api_key = ""

[conversation]
# Voice input goes to a conversation model with tools, hooks, skills and MCP disabled. It decides
# when to delegate and speaks the worker's results. It is the chat's own /model whenever that is a
# Claude or a Local model, so one model answers the whole chat; the model below talks only for chats
# on Codex or Grok, which cannot hold a voice conversation. A Claude model uses your Claude
# subscription; a Local model (qwen3.6) runs on the llama-server in [local] for free.
# /voicemodel <model> switches this one for the session (add "save" to write it here).
model = "sonnet"
# A Local conversation model never thinks (its replies are grammar-constrained), so its effort is always off.
effort = "low"
# Let the voice model search and read the web itself (read-only WebSearch/WebFetch) instead of
# delegating live facts such as the weather to the CLI worker. Everything else stays off.
web = true
# Maximum delegated tasks per spoken turn; the final response must just speak.
max_delegations = 3
# Timeout for each conversational response (the CLI worker keeps its normal lifetime).
timeout_s = 120
# Record stage durations only (no speech or tool content) in ~/.local/state/vision/voice-timing.jsonl.
timing = false

[router]
# The front end: the conversation model (a local Qwen, or a Claude) takes every request, answers the
# basic ones itself, fetches the weather from WeatherKit, runs a search-only web lookup for current
# facts, and hands everything substantive to an agent launched by Vision's supervisor (vision/supervisor.py).
#   "off"    the conversation model decides delegation on its own, as before, on the session's brain
#   "audit"  routes are worked out and logged (~/.local/state/vision/routing.jsonl, and a dim line in the
#            chat) but nothing changes: review them, then switch on
#   "on"     routes are enforced; substantive work goes to default_agent at default_effort
# /router off|audit|on switches for the session.
mode = "audit"
# With mode = "on", typed messages go through the front end too (false: only spoken ones; typed input
# keeps talking straight to the /model brain).
typed = true
# Where substantive work goes unless the user says otherwise ("use Codex high for this",
# /agent codex --effort high). Architecture, hard debugging, repo-wide changes, security-sensitive work,
# multi-step research and tasks that already failed at the default effort get high_effort instead.
default_agent = "opus"
default_effort = "medium"
high_effort = "high"
# The agents the supervisor may launch: name = catalogue model (see /model). "" for codex means the
# first Codex model in the catalogue. Anything else the user or the model names is refused, never
# swapped for something else.
[router.agents]
opus = "opus"
codex = ""
[router.limits]
# A run is stopped and reported as timed out after this many seconds.
timeout_s = 1800
# How many times one run may come back to you with a question before it is stopped.
max_rounds = 4
# A run that has written more than this many tokens is not resumed after a question.
max_output_tokens = 200000
[router.permissions]
# Commands the launched agent may not run on its own: it comes back with a question, you answer yes,
# and the run resumes with that command allowed. Syntax as denied_tools. The rest of [brain].denied_tools
# stays forbidden outright.
approval = ["Bash(git push:*)", "Bash(pip install:*)", "Bash(uv pip install:*)", "Bash(npm install -g:*)", "Bash(apt:*)", "Bash(apt-get:*)", "Bash(dnf:*)", "Bash(brew:*)", "Bash(rm -rf:*)", "Bash(gh pr create:*)", "Bash(gh release:*)", "Bash(ssh:*)", "Bash(scp:*)", "Bash(rsync:*)", "Bash(docker push:*)", "Bash(kubectl apply:*)", "Bash(terraform apply:*)", "Bash(mail:*)", "Bash(sendmail:*)", "Bash(curl -X POST:*)", "Bash(curl -d:*)"]

[voice]
# Speech engine: "qwen3" (Qwen3-TTS 1.7B: cloned or designed voices, the most natural) or
# "orpheus" (Orpheus 3B via llama.cpp, eight built-in voices).
engine = "qwen3"
# qwen3: the name of a voice under ~/.local/share/vision/voices (make one with `vision voice design jarvis`
# or `vision voice add <name> --from clip.wav`), or a path to a WAV clip to imitate.
# orpheus: a preset ("friday") or one of tara, leah, jess, leo, dan, mia, zac, zoe.
# Run `vision voices` to list what is available, `--preview` to hear them.
voice = "jarvis"
# Language the qwen3 engine speaks ("Auto" detects; English, Chinese, German, French, Spanish, ...).
language = "English"
# How qwen3 imitates the reference clip: "embedding" keeps its timbre and lets the model shape the delivery
# itself (cleanest, the default); "context" continues the clip's own audio, closer to its exact pacing
# and tone but it also inherits any roughness in the clip (needs ref.txt, the transcript).
clone_mode = "embedding"
# qwen3: each sentence continues from what the voice just said (its own audio is the context), so a reply
# flows as one performance instead of a string of separately read sentences. false = every sentence cold.
continuity = true
# Speaking rate applied to the audio after synthesis (pitch unchanged): 1.0 = as spoken, 1.15 = brisk,
# 1.3 = hurried. Fine-tune pace without changing the voice.
rate = 1.0
# Where the voice model runs: "auto" tries the GPU (CUDA) then falls back to CPU (and says why, e.g. another
# Vision still holding the GPU). "cuda" never falls back: the voice refuses to load and names what is in the way.
# On the CPU these models are several times slower than real time; the GPU is the intended home.
device = "auto"
# qwen3 on the GPU: "none" keeps the voice model's weights in bf16 (the default). "int8" stores the
# transformer weights as 8-bit integers: about 1.4 GB less VRAM and inaudible, but the lighter load lets
# this laptop's driver drop the GPU clocks, which can make it slower; see vision/int8.py.
quant = "none"
# orpheus only: local port for the llama.cpp server that hosts the voice model (127.0.0.1 only).
port = 8766
# Output device index or name substring (a PipeWire sink name such as "Snowball" works too). Empty = system default.
output_device = ""
# A spoken conversation: when the reply's first words have not started filler_after_ms after you stop
# talking (transcription and the model's own thinking time), the voice says a short "one sec" instead of
# leaving dead air, then the reply follows. The phrases are made in the current voice at warm-up and kept
# under ~/.local/state/vision/fillers, so they cost nothing on the turn. One is picked at random, never the
# same one twice running. If the reply still has not started filler_again_ms later, one of the later
# phrases is said, once (0 = never). false turns the filler off.
filler = true
filler_after_ms = 700
filler_again_ms = 10000
filler_phrases = ["One sec.", "Let me see.", "Hmm, let me think.", "Right, let me have a look.", "Hang on a moment.", "Let me check."]
filler_later_phrases = ["Still on it.", "Bear with me.", "Nearly there."]

[listen]
# Input device index or name substring (a PipeWire source name such as "Snowball" works too).
# Empty = system default (follows your PipeWire default source).
input_device = ""
# Whisper model: "large-v3-turbo" (GPU, best), "distil-large-v3", "small.en", "base.en" ...
whisper_model = "large-v3-turbo"
# "auto" tries CUDA then falls back to CPU. Or force "cuda" / "cpu".
device = "auto"
# Voice activity detection aggressiveness 0-3 (3 = strictest, best in noisy rooms).
vad_aggressiveness = 2
# Milliseconds of silence that ends an utterance.
end_silence_ms = 900
# Give up waiting for speech after this many seconds (0 = wait forever).
start_timeout_s = 0
# Maximum utterance length in seconds.
max_utterance_s = 45
# Play a short chime when Vision starts listening.
chime = true
# Show the words forming on screen while you talk when Whisper is on the GPU (re-transcribes the
# utterance so far every live_ms; 0 turns it off). CPU preview is skipped so it cannot stall /talk.
live_ms = 600
# Cut a reply short by voice while Vision is speaking (Esc / Ctrl-C always work):
#   "wake"   say Vision's name ("Vision, stop" / "Vision, what about…" runs straight away). Works on
#            speakers: a tiny Whisper on the CPU listens for the name over Vision's own voice.
#   "speech" just start talking (~a quarter second of voice). Needs headphones or echo cancellation
#            (`pactl load-module module-echo-cancel`, then input_device = "echo-cancel"; on Windows,
#            echo_cancel below), or Vision hears itself through the speakers and cuts itself off.
#   "off"    keyboard only.
barge_in = "wake"
# Take Vision's own voice out of the microphone in Vision itself (WebRTC's echo canceller, via the
# livekit package): "auto" = on for Windows, off elsewhere (PipeWire's echo-cancel module covers
# Linux); "on"; "off".
echo_cancel = "auto"

[wake]
# Say Vision's name to start talking without touching the keyboard (also /wake inside a chat).
# While the chat is idle a tiny Whisper on the CPU listens for the name; when it hears it the full ears
# and voice warm up and Vision listens ("Vision, what's the weather?" runs straight away).
enabled = false
# Spellings that count as the name (matched loosely: "Vision", "vision's", "envision" all wake).
names = ["vision"]
# The small Whisper that spots the name. It runs on the CPU, so keep it tiny.
model = "tiny.en"
# After a reply, keep listening this many seconds for a follow-up before going back to sleep.
follow_up_s = 8

[buddy]
# Pip, the pocket robot in the bottom-left input gutter. Set enabled = false to hide him.
name = "Pip"
enabled = true

[remote]
# `vision serve`: the HTTP + WebSocket server the Vision Remote iOS app talks to.
# 0.0.0.0 = reachable from the phone on the same Wi-Fi (the QR carries the laptop's LAN address);
# 127.0.0.1 = laptop only, for use behind an HTTPS tunnel (Tailscale Funnel or Cloudflare Tunnel).
# The pairing token lives in remote_token next to this file (`vision serve --new-token` rotates it).
host = "0.0.0.0"
port = 8765
# The https:// address the phone should use away from home (your tunnel hostname).
# Empty = detect Tailscale, else the LAN address.
public_url = ""

[weather]
# Live weather from Apple WeatherKit (included with an Apple Developer membership: 500,000 calls a month).
# When a spoken request is about the weather, Vision fetches the report while the voice model starts
# thinking and hands it over as data, so nothing is web-searched. `vision weather [place]` tests it.
# Setup at developer.apple.com: Certificates, Identifiers & Profiles → Keys → new key with WeatherKit
# enabled (download the .p8 once; it lives next to this file as weatherkit.p8), and Identifiers → Services
# IDs → a new identifier with WeatherKit enabled. Then fill in the three ids below.
enabled = true
team_id = ""       # 10-character Team ID (top right of the developer account page)
service_id = ""    # the Services ID identifier, e.g. com.example.vision.weatherkit
key_id = ""        # the Key ID shown on the WeatherKit key
key_file = "weatherkit.p8"   # path to the .p8; relative paths are under ~/.config/vision
# Where "the weather" means when no place is named. A place name is geocoded once (Open-Meteo, no key)
# and cached; or give latitude/longitude (and timezone) directly to skip geocoding.
location = ""
latitude = 0.0
longitude = 0.0
timezone = ""      # e.g. Europe/London; only used with latitude/longitude
country_code = ""  # e.g. GB; enables official weather alerts when using latitude/longitude
units = "metric"   # "metric" (°C, km/h) or "imperial" (°F, mph)
language = "en"
'''
if DEFAULT_MODE != "auto":
    DEFAULT_CONFIG = DEFAULT_CONFIG.replace('\nmode = "auto"\n', f'\nmode = "{DEFAULT_MODE}"\n', 1)
if sys.platform == "win32":
    DEFAULT_CONFIG = DEFAULT_CONFIG.replace(
        '"Bash(grok:*)"]\n', '"Bash(grok:*)", ' + ", ".join(f'"{r}"' for r in WINDOWS_DENIED) + "]\n", 1)

VOICE_ENGINES = ("qwen3", "orpheus")
# The voices baked into the Orpheus fine-tune, roughly in order of how polished they are.
VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
# Named Orpheus presets. "friday" is the calm, clear female voice closest to the FRIDAY character.
VOICE_PRESETS = {"friday": "tara"}


# What the voice says while a spoken reply is still coming ([voice] filler_phrases / filler_later_phrases).
FILLER_PHRASES = ("One sec.", "Let me see.", "Hmm, let me think.", "Right, let me have a look.", "Hang on a moment.", "Let me check.")
FILLER_LATER_PHRASES = ("Still on it.", "Bear with me.", "Nearly there.")


def voice_dir(name: str) -> Path:
    return VOICES_DIR / name


def saved_voices() -> list[str]:
    """Names of the cloned / designed voices on disk (qwen3 engine)."""
    if not VOICES_DIR.is_dir():
        return []
    return sorted(d.name for d in VOICES_DIR.iterdir() if (d / "ref.wav").is_file())


def voice_choices(vc: "VoiceConfig") -> list[tuple[str, str]]:
    """(name, note) pairs for the /voice picker and `vision voices`, for the configured engine."""
    if vc.engine == "orpheus":
        return [(n, f"preset → {v}") for n, v in VOICE_PRESETS.items()] + [(v, "") for v in VOICES]
    have = saved_voices()
    out = [(name, "designed" if (voice_dir(name) / "design.txt").is_file() else "cloned") for name in have]
    out += [(name, f"built-in design, not made yet → vision voice design {name}") for name in VOICE_DESIGNS if name not in have]
    return out


@dataclass
class CodexConfig:
    sandbox: str = "auto"
    extra_config: list[str] = field(default_factory=list)
    transport: str = "app-server"  # or "exec" (see vision/codex_app.py)


@dataclass
class GrokConfig:
    sandbox: str = "auto"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class LocalConfig:
    """The llama-server that hosts the Local models (deploy/local-model)."""

    base_url: str = "http://localhost:8080/v1"
    timeout_s: float = 180
    context: int = 131072  # the server's -c; history is trimmed to stay under it
    brave_api_key: str = ""  # WebSearch uses Brave's API with a key, DuckDuckGo's HTML results without


@dataclass
class BrainConfig:
    model: str = "opus"
    effort: str = "high"
    # Session-only provider speed tier. /fast toggles it; it is deliberately not saved with defaults.
    fast: bool = False
    allowed_tools: list[str] = field(
        default_factory=lambda: ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
    )
    denied_tools: list[str] = field(
        default_factory=lambda: ["Bash(sudo:*)", "Bash(rm -rf /:*)", "Bash(rm -rf ~:*)", "Bash(mkfs:*)", "Bash(dd:*)", "Bash(shutdown:*)", "Bash(reboot:*)", "Bash(claude:*)", "Bash(codex:*)", "Bash(grok:*)"]
        + (WINDOWS_DENIED if sys.platform == "win32" else [])
    )
    workdir: str = ""
    address_user_as: str = ""
    # "auto" (every tool pre-approved, Codex on danger-full-access; denied_tools still applies) or
    # "plan" (read-only until the plan is approved). Shift-Tab or /mode switches for the session.
    mode: str = DEFAULT_MODE
    # What approving a plan unlocks: "session" (Vision switches to auto for the rest of the session)
    # or "turn" (only the approved plan runs; the next message starts in plan mode again).
    plan_approval: str = "session"
    # Read Claude Code's and Grok's saved logins to query their usage endpoints directly (see
    # read_cli_logins in DEFAULT_CONFIG); False = Vision never opens those credential files.
    read_cli_logins: bool = True
    # Seconds of silence from the Claude process before the turn is killed as stalled; 0 = never.
    stall_s: float = 900
    codex: CodexConfig = field(default_factory=CodexConfig, repr=False)
    grok: GrokConfig = field(default_factory=GrokConfig, repr=False)
    local: LocalConfig = field(default_factory=LocalConfig, repr=False)
    weather: "WeatherConfig | None" = None


MODES = ("auto", "plan")

_read_cli_logins = True  # [brain].read_cli_logins from the last load_config()


def cli_logins_allowed() -> bool:
    """Whether Vision may read the Claude Code and Grok login files (usage.py, grok.py)."""
    return _read_cli_logins


@dataclass
class ConversationConfig:
    model: str = "sonnet"
    effort: str = "low"
    web: bool = True
    max_delegations: int = 3
    timeout_s: float = 120
    timing: bool = False


@dataclass
class RouterConfig:
    """The front end and its supervisor (vision/routing.py, vision/supervisor.py)."""

    mode: str = "audit"  # "off" | "audit" | "on"
    typed: bool = True  # in "on" mode typed input goes through the front end too
    default_agent: str = "opus"
    default_effort: str = "medium"
    high_effort: str = "high"
    agents: dict = field(default_factory=lambda: {"opus": "opus", "codex": ""})  # name → catalogue model ("" = provider default)
    timeout_s: float = 1800
    max_rounds: int = 4
    max_output_tokens: int = 200_000
    approval: list[str] = field(default_factory=lambda: [
        "Bash(git push:*)", "Bash(pip install:*)", "Bash(uv pip install:*)", "Bash(npm install -g:*)", "Bash(apt:*)",
        "Bash(apt-get:*)", "Bash(dnf:*)", "Bash(brew:*)", "Bash(rm -rf:*)", "Bash(gh pr create:*)", "Bash(gh release:*)",
        "Bash(ssh:*)", "Bash(scp:*)", "Bash(rsync:*)", "Bash(docker push:*)", "Bash(kubectl apply:*)", "Bash(terraform apply:*)",
        "Bash(mail:*)", "Bash(sendmail:*)", "Bash(curl -X POST:*)", "Bash(curl -d:*)",
    ])


ROUTER_MODES = ("off", "audit", "on")


@dataclass
class VoiceConfig:
    engine: str = "qwen3"
    voice: str = "jarvis"
    language: str = "English"
    clone_mode: str = "embedding"
    continuity: bool = True
    rate: float = 1.0
    device: str = "auto"
    quant: str = "none"
    output_device: str = ""
    port: int = 8766
    # Something to say while the reply is still coming: see [voice] filler in DEFAULT_CONFIG.
    filler: bool = True
    filler_after_ms: int = 700
    filler_again_ms: int = 10000
    filler_phrases: list[str] = field(default_factory=lambda: list(FILLER_PHRASES))
    filler_later_phrases: list[str] = field(default_factory=lambda: list(FILLER_LATER_PHRASES))


@dataclass
class ListenConfig:
    input_device: str = ""
    whisper_model: str = "large-v3-turbo"
    device: str = "auto"
    vad_aggressiveness: int = 2
    end_silence_ms: int = 900
    start_timeout_s: float = 0
    max_utterance_s: float = 45
    chime: bool = True
    live_ms: int = 600  # live transcription preview interval; 0 = off
    barge_in: str = "wake"  # "wake" | "speech" | "off"
    echo_cancel: str = "auto"  # "auto" (on for Windows) | "on" | "off"; see vision/echo.py


@dataclass
class WakeConfig:
    enabled: bool = False
    names: list[str] = field(default_factory=lambda: ["vision"])
    model: str = "tiny.en"
    follow_up_s: float = 8


@dataclass
class BuddyConfig:
    name: str = "Pip"
    enabled: bool = True


@dataclass
class RemoteConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    public_url: str = ""


@dataclass
class WeatherConfig:
    enabled: bool = True
    team_id: str = ""
    service_id: str = ""
    key_id: str = ""
    key_file: str = "weatherkit.p8"
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = ""
    country_code: str = ""
    units: str = "metric"
    language: str = "en"

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.team_id.strip() and self.service_id.strip() and self.key_id.strip())


@dataclass
class Config:
    brain: BrainConfig = field(default_factory=BrainConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    grok: GrokConfig = field(default_factory=GrokConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    buddy: BuddyConfig = field(default_factory=BuddyConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    def __post_init__(self) -> None:
        self.brain.codex = self.codex
        self.brain.grok = self.grok
        self.brain.local = self.local
        self.brain.weather = self.weather


def weather_ready(brain: BrainConfig) -> bool:
    """True when the WeatherKit integration is configured for this brain's config."""
    w = getattr(brain, "weather", None)
    return bool(w is not None and w.configured)


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, MODELS_DIR, WORKSPACE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    ensure_dirs()
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")
    try:
        raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Config error in {CONFIG_PATH}: {e}")
    cfg = Config()
    for section, target in (("brain", cfg.brain), ("codex", cfg.codex), ("grok", cfg.grok), ("local", cfg.local), ("conversation", cfg.conversation), ("voice", cfg.voice), ("listen", cfg.listen), ("wake", cfg.wake), ("buddy", cfg.buddy), ("remote", cfg.remote), ("weather", cfg.weather)):
        for k, v in raw.get(section, {}).items():
            if hasattr(target, k):
                if section == "brain" and k == "model" and not v:
                    continue  # migrate the old "Claude default" setting to Vision's Opus default
                if section == "brain" and k == "effort" and not v and not raw.get("brain", {}).get("model"):
                    continue  # same migration; an explicit model + empty effort is meaningful (e.g. Haiku)
                setattr(target, k, v)
    cfg.brain.mode = str(cfg.brain.mode).strip().lower()
    if cfg.brain.mode not in MODES:
        cfg.brain.mode = DEFAULT_MODE
    if cfg.brain.plan_approval not in ("session", "turn"):
        cfg.brain.plan_approval = "session"
    global _read_cli_logins
    _read_cli_logins = cfg.brain.read_cli_logins is not False
    vc = cfg.voice
    vc.filler_after_ms = max(0, int(vc.filler_after_ms or 0))
    vc.filler_again_ms = max(0, int(vc.filler_again_ms or 0))
    vc.filler_phrases = [str(x).strip() for x in (vc.filler_phrases if isinstance(vc.filler_phrases, list) else []) if str(x).strip()]
    vc.filler_later_phrases = [str(x).strip() for x in (vc.filler_later_phrases if isinstance(vc.filler_later_phrases, list) else []) if str(x).strip()]
    router = raw.get("router", {})
    for k, v in router.items():
        if k in ("agents", "limits", "permissions"):
            continue
        if hasattr(cfg.router, k):
            setattr(cfg.router, k, v)
    if isinstance(router.get("agents"), dict):
        cfg.router.agents = {str(k).strip().lower(): str(v or "").strip() for k, v in router["agents"].items() if str(k).strip()}
    for k, v in (router.get("limits") or {}).items():
        if k in ("timeout_s", "max_rounds", "max_output_tokens"):
            setattr(cfg.router, k, v)
    if isinstance((router.get("permissions") or {}).get("approval"), list):
        cfg.router.approval = [str(p) for p in router["permissions"]["approval"]]
    cfg.router.mode = str(cfg.router.mode or "off").strip().lower()
    if cfg.router.mode in ("false", "no", "0"):
        cfg.router.mode = "off"
    if cfg.router.mode not in ROUTER_MODES:
        cfg.router.mode = "audit"
    cfg.router.default_agent = str(cfg.router.default_agent or "opus").strip().lower()
    cfg.router.default_effort = str(cfg.router.default_effort or "medium").strip().lower()
    cfg.router.high_effort = str(cfg.router.high_effort or "high").strip().lower()
    cfg.router.timeout_s = max(30.0, float(cfg.router.timeout_s or 1800))
    cfg.router.max_rounds = max(1, int(cfg.router.max_rounds or 4))
    cfg.router.max_output_tokens = max(1000, int(cfg.router.max_output_tokens or 200_000))
    cfg.conversation.max_delegations = min(8, max(1, int(cfg.conversation.max_delegations)))
    if provider_for(cfg.conversation.model) == "local":
        cfg.conversation.effort = THINKING_OFF  # LocalConversation always sends enable_thinking=False (grammar mode)
    cfg.conversation.timeout_s = max(10, float(cfg.conversation.timeout_s))
    cfg.brain.stall_s = max(0.0, float(cfg.brain.stall_s or 0))
    if cfg.weather.units not in ("metric", "imperial"):
        cfg.weather.units = "metric"
    if cfg.voice.engine not in VOICE_ENGINES:
        cfg.voice.engine = "qwen3"
    cfg.voice.rate = min(2.0, max(0.5, float(cfg.voice.rate or 1.0)))
    if cfg.voice.clone_mode not in ("embedding", "context"):
        cfg.voice.clone_mode = "embedding"
    if isinstance(cfg.wake.names, str):
        cfg.wake.names = [cfg.wake.names]
    cfg.wake.names = [n.strip().lower() for n in cfg.wake.names if n.strip()] or ["vision"]
    cfg.listen.barge_in = str(cfg.listen.barge_in or "off").strip().lower()
    if cfg.listen.barge_in in ("false", "no", "none", "0"):
        cfg.listen.barge_in = "off"
    if cfg.listen.barge_in not in ("wake", "speech", "off"):
        cfg.listen.barge_in = "wake"
    if re.fullmatch(r"[a-z]{2}_\w+(:[\d.]+)?(,.*)?", cfg.voice.voice):
        cfg.voice.voice = "friday"  # a Kokoro-era voice or blend
    if cfg.voice.engine == "qwen3" and cfg.voice.voice in VOICE_PRESETS and cfg.voice.voice not in saved_voices():
        cfg.voice.voice = "jarvis"  # an Orpheus-era preset left in the config; jarvis is the qwen3 default
    # Existing configs that still deny only claude/codex pick up the grok shim-matching rule.
    if "Bash(grok:*)" not in cfg.brain.denied_tools and "Bash(codex:*)" in cfg.brain.denied_tools:
        cfg.brain.denied_tools = list(cfg.brain.denied_tools) + ["Bash(grok:*)"]
    # The drivers receive BrainConfig for historical compatibility; attach the provider-specific
    # settings without changing every public constructor.
    cfg.brain.codex = cfg.codex
    cfg.brain.grok = cfg.grok
    cfg.brain.local = cfg.local
    cfg.brain.weather = cfg.weather
    return cfg


def saved_brain_defaults() -> tuple[str, str] | None:
    """The default model and effort as config.toml has them now, or None when the file cannot be read.

    For long-running processes (`vision serve`) that must pick up /default without a restart and must
    not die on a half-edited config: load_config exits on a TOML error, so that is caught here too."""
    try:
        brain = load_config().brain
    except (Exception, SystemExit):
        return None
    return brain.model, brain.effort


def save_brain_defaults(model: str, effort: str) -> None:
    """Persist the default model and effort in config.toml, editing the lines in place so comments survive."""
    ensure_dirs()
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else DEFAULT_CONFIG
    lines = text.splitlines()
    wanted = {"model": model, "effort": effort}
    section, done = None, set()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
            continue
        if section == "brain":
            m = re.match(r"^\s*(model|effort)\s*=", line)
            if m and m.group(1) not in done:
                lines[i] = f'{m.group(1)} = "{wanted[m.group(1)]}"'
                done.add(m.group(1))
    missing = [k for k in ("model", "effort") if k not in done]
    if missing:
        try:
            at = next(i for i, l in enumerate(lines) if l.strip() == "[brain]") + 1
        except StopIteration:
            lines += ["", "[brain]"]
            at = len(lines)
        for k in reversed(missing):
            lines.insert(at, f'{k} = "{wanted[k]}"')
    out = "\n".join(lines) + "\n"
    tomllib.loads(out)  # refuse to write a broken config
    CONFIG_PATH.write_text(out, encoding="utf-8")


def save_voice_default(name: str) -> None:
    """Persist `[voice] voice = name` in config.toml, editing the line in place so comments survive."""
    ensure_dirs()
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else DEFAULT_CONFIG
    lines = text.splitlines()
    section, done = None, False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
            continue
        if section == "voice" and not done and re.match(r"^\s*voice\s*=", line):
            lines[i] = f'voice = "{name}"'
            done = True
    if not done:
        try:
            at = next(i for i, l in enumerate(lines) if l.strip() == "[voice]") + 1
        except StopIteration:
            lines += ["", "[voice]"]
            at = len(lines)
        lines.insert(at, f'voice = "{name}"')
    out = "\n".join(lines) + "\n"
    tomllib.loads(out)  # refuse to write a broken config
    CONFIG_PATH.write_text(out, encoding="utf-8")


def save_config_value(section: str, key: str, literal: str) -> None:
    """Persist one `[section] key = literal` in config.toml, editing the line in place so comments survive."""
    ensure_dirs()
    text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else DEFAULT_CONFIG
    lines = text.splitlines()
    current, done = None, False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1].strip()
            continue
        if current == section and not done and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            lines[i] = f"{key} = {literal}"
            done = True
    if not done:
        try:
            at = next(i for i, l in enumerate(lines) if l.strip() == f"[{section}]") + 1
        except StopIteration:
            lines += ["", f"[{section}]"]
            at = len(lines)
        lines.insert(at, f"{key} = {literal}")
    out = "\n".join(lines) + "\n"
    tomllib.loads(out)  # refuse to write a broken config
    CONFIG_PATH.write_text(out, encoding="utf-8")


def save_wake_enabled(enabled: bool) -> None:
    """Persist `[wake] enabled` in config.toml."""
    save_config_value("wake", "enabled", "true" if enabled else "false")


def save_input_device(spec: str) -> None:
    """Persist `[listen] input_device` in config.toml ("" = the system default)."""
    save_config_value("listen", "input_device", '"' + spec.replace('\\', '\\\\').replace('"', '\\"') + '"')


_OPEN_LOCK = threading.Lock()


@dataclass(frozen=True)
class AudioDevice:
    """A sounddevice index (None = system default) plus, for a name that only matched a PipeWire
    node, the node to route to.

    PortAudio lists every PipeWire node under its JACK host API, but those entries hang on open
    and only run at the graph rate, so a node is reached through the ALSA `pipewire` PCM with
    PIPEWIRE_NODE set while the stream is constructed: pipewire-alsa reads it at snd_pcm_open.
    """

    index: int | None = None
    node: str | None = None

    @contextlib.contextmanager
    def opening(self):
        """Wrap the sd.InputStream/OutputStream/play call that opens this device."""
        if self.node is None:
            yield
            return
        with _OPEN_LOCK:  # the variable is process-wide: one open at a time
            saved = os.environ.get("PIPEWIRE_NODE")
            os.environ["PIPEWIRE_NODE"] = self.node
            try:
                yield
            finally:
                if saved is None:
                    os.environ.pop("PIPEWIRE_NODE", None)
                else:
                    os.environ["PIPEWIRE_NODE"] = saved


def pipewire_nodes(kind: str) -> list[tuple[str, str]]:
    """(node.name, description) of each PipeWire source (input) or sink (output); [] without PipeWire."""
    import json
    import subprocess

    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5, check=True, encoding="utf-8").stdout
        objects = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    want = "Audio/Source" if kind == "input" else "Audio/Sink"
    nodes = []
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") == want and props.get("node.name"):
            nodes.append((props["node.name"], props.get("node.description") or props.get("node.nick") or props["node.name"]))
    return nodes


def resolve_device(spec: str | int | None, kind: str) -> AudioDevice:
    """Turn a device index or name fragment into an AudioDevice (the system default when empty)."""
    if spec is None or spec == "":
        return AudioDevice()
    import sounddevice as sd

    if isinstance(spec, int) or str(spec).isdigit():
        return AudioDevice(int(spec))
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = str(spec).lower()
    devices = sd.query_devices()
    # Through PipeWire first: its `pipewire` PCM shares the device, whereas PortAudio's raw `hw:` entry
    # takes it exclusively, so PipeWire itself can no longer open it and everything else (the desktop,
    # a second Vision going through PipeWire) waits for audio that never comes.
    for name, description in pipewire_nodes(kind):
        if needle in description.lower() or needle in name.lower():
            alsa = next((i for i, d in enumerate(devices) if d["name"] == "pipewire" and d[key] > 0), None)
            return AudioDevice(alsa, name)
    order = _windows_devices(sd, key) if sys.platform == "win32" else enumerate(devices)
    for i, d in order:
        if d[key] > 0 and needle in d["name"].lower() and "JACK" not in sd.query_hostapis(d["hostapi"])["name"]:
            return AudioDevice(i)
    raise SystemExit(f"No {kind} device matching {spec!r}. Run `vision doctor` to list devices.")


# Windows lists every device once per host API. DirectSound and MME convert to whatever rate Vision
# asks for (16 kHz in, 24 kHz out); WASAPI shared mode only runs at the mixer's rate, and WDM-KS
# opens the hardware exclusively. DirectSound first: MME truncates names to 31 characters.
_WINDOWS_HOST_APIS = ("Windows DirectSound", "MME")


def _windows_devices(sd, key: str) -> list[tuple[int, dict]]:
    """(index, device) for the usable Windows host APIs, in _WINDOWS_HOST_APIS order."""
    rank = {name: n for n, name in enumerate(_WINDOWS_HOST_APIS)}
    found = []
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        if d[key] > 0 and api in rank:
            found.append((rank[api], i, d))
    return [(i, d) for _, i, d in sorted(found, key=lambda r: (r[0], r[1]))]


def input_device_choices() -> list[tuple[str, str, str]]:
    """(value, label, desc) rows for a microphone picker: the system default first, then every PipeWire
    source by node name (what resolve_device matches exactly), or the plain PortAudio inputs without PipeWire."""
    rows = [("", "system default", "follows your PipeWire default source")]
    nodes = pipewire_nodes("input")
    if nodes:
        return rows + [(name, desc, name) for name, desc in nodes]
    import sounddevice as sd

    if sys.platform == "win32":
        rows[0] = ("", "system default", "follows your Windows default microphone")
        first = _WINDOWS_HOST_APIS[0]
        devices = [(i, d) for i, d in _windows_devices(sd, "max_input_channels") if sd.query_hostapis(d["hostapi"])["name"] == first]
        return rows + [(str(i), d["name"], f"#{i} · {d['max_input_channels']} in") for i, d in devices]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "JACK" not in sd.query_hostapis(d["hostapi"])["name"]:
            rows.append((str(i), d["name"], f"#{i} · {d['max_input_channels']} in"))
    return rows

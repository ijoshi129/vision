"""Configuration for Vision.

Config lives in ~/.config/vision/config.toml (created with defaults on first run).
Models and Claude's working directory live under ~/.local/share/vision/.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vision"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "vision"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "vision"
CONFIG_PATH = CONFIG_DIR / "config.toml"
MODELS_DIR = DATA_DIR / "models"
WORKSPACE_DIR = DATA_DIR / "workspace"

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

DEFAULT_CONFIG = '''# Vision configuration. Edit freely; run `vision config` to see the resolved values.

[brain]
# Claude model alias or full name. Leave empty to use your Claude Code default.
model = ""
# Effort level passed to Claude Code ("low", "medium", "high"). Empty = default.
effort = ""
# Tools Vision may use without prompting (headless mode cannot ask, so anything not listed is denied).
# Bash = shell commands, Read/Glob/Grep = read files, Write/Edit = create and change files.
# Deleting files happens through Bash (rm). Remove "Bash" for a read-only assistant.
allowed_tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
# Command patterns Vision may never run, even with Bash allowed. Syntax: Bash(prefix:*)
denied_tools = ["Bash(sudo:*)", "Bash(rm -rf /:*)", "Bash(rm -rf ~:*)", "Bash(mkfs:*)", "Bash(dd:*)", "Bash(shutdown:*)", "Bash(reboot:*)"]
# Directory Vision works in. Empty = the directory you launch `vision` from.
workdir = ""
# How you'd like Vision to address you (e.g. "boss", your first name). Empty = nothing special.
address_user_as = "boss"

[voice]
# Kokoro voice. A single name ("bf_emma") or a blend ("bf_emma:0.6,bf_isabella:0.4").
# Run `vision voices --preview` to audition them. British female voices start with "bf_".
voice = "friday"
speed = 1.0
# Where Kokoro runs: "auto" tries the GPU (CUDA) then falls back to CPU. Or force "cuda" / "cpu".
device = "auto"
# Output device index or name substring. Empty = system default.
output_device = ""

[listen]
# Input device index or name substring. Empty = system default (follows your PipeWire default source).
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
'''

# Named voice presets. "friday" is a warm, light British blend chosen to sit close to
# the FRIDAY character: calm, crisp, slightly bright.
VOICE_PRESETS = {
    "friday": "bf_emma:0.55,bf_isabella:0.30,bf_lily:0.15",
    "emma": "bf_emma",
    "isabella": "bf_isabella",
    "lily": "bf_lily",
    "alice": "bf_alice",
    "heart": "af_heart",
    "bella": "af_bella",
}


@dataclass
class BrainConfig:
    model: str = ""
    effort: str = ""
    allowed_tools: list[str] = field(
        default_factory=lambda: ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
    )
    denied_tools: list[str] = field(
        default_factory=lambda: ["Bash(sudo:*)", "Bash(rm -rf /:*)", "Bash(rm -rf ~:*)", "Bash(mkfs:*)", "Bash(dd:*)", "Bash(shutdown:*)", "Bash(reboot:*)"]
    )
    workdir: str = ""
    address_user_as: str = "boss"


@dataclass
class VoiceConfig:
    voice: str = "friday"
    speed: float = 1.0
    device: str = "auto"
    output_device: str = ""


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


@dataclass
class Config:
    brain: BrainConfig = field(default_factory=BrainConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, MODELS_DIR, WORKSPACE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    ensure_dirs()
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
    try:
        raw = tomllib.loads(CONFIG_PATH.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Config error in {CONFIG_PATH}: {e}")
    cfg = Config()
    for section, target in (("brain", cfg.brain), ("voice", cfg.voice), ("listen", cfg.listen)):
        for k, v in raw.get(section, {}).items():
            if hasattr(target, k):
                setattr(target, k, v)
    return cfg


def resolve_device(spec: str | int | None, kind: str):
    """Turn a device index or name fragment into a sounddevice index (None = default)."""
    if spec is None or spec == "":
        return None
    import sounddevice as sd

    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and str(spec).lower() in d["name"].lower():
            return i
    raise SystemExit(f"No {kind} device matching {spec!r}. Run `vision doctor` to list devices.")

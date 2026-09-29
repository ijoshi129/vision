"""Which optional builds are installed and how to add one. Standard library only: `vision.__main__` uses it
to turn a missing dependency into an install hint before anything heavy has imported."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the checkout, when Vision runs from one
SETUP = ROOT / "scripts" / "setup.sh"


def _path(p: Path) -> str:
    """A path to paste into a shell: ~ for the home directory, quoted when it has to be."""
    home = Path.home()
    try:
        rel = p.relative_to(home)
    except ValueError:
        return shlex.quote(str(p))
    quoted = shlex.quote(str(rel))
    return f"~/{rel}" if quoted == str(rel) else shlex.quote(str(p))


def voice_backend() -> str | None:
    """Which voice build is installed: "nvidia" (voice-nvidia), "cpu" (voice-cpu) or None (no voice).

    Read from torch's own metadata, so it never imports torch and packages an earlier build left behind
    don't count: PyPI's Linux torch requires the CUDA libraries, the CPU index's torch (and macOS's) doesn't.
    """
    from importlib.metadata import PackageNotFoundError, requires, version

    try:
        v = version("torch")
    except PackageNotFoundError:
        return None
    if "+cpu" in v or not sys.platform.startswith("linux"):
        return "cpu"
    deps = requires("torch") or []
    return "nvidia" if any(d.startswith(("nvidia-", "cuda-toolkit")) for d in deps) else "cpu"


def setup_hint(backend: str = "") -> str:
    """The command that installs the voice, optionally for a given backend, runnable from any directory."""
    flag = {"cpu": " --cpu", "nvidia": " --nvidia"}.get(backend, "")
    if SETUP.exists():
        return f"{_path(SETUP)} --voice{flag}"
    # Installed without a checkout: uv, not pip, because only uv reads the torch sources in pyproject.toml.
    if backend:
        return f"uv pip install 'vision[voice-{backend}]'"
    return "uv pip install 'vision[voice-nvidia]' (NVIDIA GPU) or 'vision[voice-cpu]' (no NVIDIA GPU)"


def extra_hint(extra: str) -> str:
    """The command that installs one optional extra ("voice", "serve", "weather")."""
    if extra == "voice":
        return setup_hint()
    if SETUP.exists():
        if extra == "serve":
            return f"{_path(SETUP)} --serve"
        return f"uv sync --project {_path(ROOT)} --frozen --inexact --extra {extra}"
    return f"pip install 'vision[{extra}]'"

"""Entry point for `vision` and `python -m vision`."""
from __future__ import annotations

import sys

# Optional packages and the extra that brings each one in (see [project.optional-dependencies]).
_EXTRAS = {
    "numpy": "voice",
    "sounddevice": "voice",
    "soundfile": "voice",
    "faster_whisper": "voice",
    "ctranslate2": "voice",
    "webrtcvad": "voice",
    "onnxruntime": "voice",
    "qwen_tts": "voice",
    "torch": "voice",
    "transformers": "voice",
    "fastapi": "serve",
    "starlette": "serve",
    "uvicorn": "serve",
    "qrcode": "serve",
    "multipart": "serve",
    "cryptography": "weather",
}


def run() -> None:
    """Run the CLI; a missing optional dependency becomes an install hint instead of a traceback."""
    try:
        from vision.cli import main

        main()
    except ModuleNotFoundError as e:
        extra = _EXTRAS.get((e.name or "").split(".")[0])
        if extra is None:
            raise
        if extra == "voice":  # two builds; the user's hardware picks. uv, not pip: only uv reads the torch sources.
            install = "uv pip install -e '.[voice-nvidia]'   (NVIDIA GPU)   or   uv pip install -e '.[voice-cpu]'   (no NVIDIA GPU)"
        else:
            install = f"pip install -e '.[{extra}]'   (or '.[all]' for everything)"
        print(
            f"vision: this needs the optional '{extra}' dependencies ({e.name} is not installed).\n"
            f"Install them with:  {install}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    run()

# Vision on Windows

Vision runs natively on Windows 10/11: no WSL. This page covers setup, what works, what doesn't, and
the differences from Linux.

Tested on Windows 11 Pro with an RTX 3070 (8 GB), NVIDIA driver 596, Python 3.12 (via uv) and the
native Claude Code 2.1.283. Codex, Grok and a local llama-server were not installed on the test
machine, so those brains are ported but untested.

## Setup

You need:

- **[Git for Windows](https://git-scm.com/download/win).** Claude Code's shell and the local model's Bash tool use its `bash.exe`.
- **Native Claude Code.** Install it with `irm https://claude.ai/install.ps1 | iex`, then run `claude` once to log in.
  Vision refuses the npm `claude.cmd` launcher: cmd.exe cuts Vision's multi-line system prompt at its
  first newline and drops every flag after it, including `--permission-mode plan`.
- **uv.** Install it with `winget install astral-sh.uv`.
- **For voice:** an NVIDIA GPU with 8 GB+ and a driver new enough for CUDA 13.

```powershell
git clone https://github.com/domdevz/vision.git; cd vision
uv venv --python 3.12 .venv
uv sync --frozen                  # text chat only (~12 MB)
uv sync --frozen --extra voice    # or: with voice (~6 GB, CUDA torch from download.pytorch.org)
bin\vision.cmd doctor --no-usage  # checks the CLIs, the Claude login (one small Haiku call), GPU, audio
bin\vision.cmd setup              # voice only: ~10 GB of models into %USERPROFILE%\.cache\huggingface
bin\vision.cmd                    # chat
```

- **Using `vision` from anywhere.** `bin\vision.cmd` is the launcher. Add the `bin` folder to your PATH, or set `VISION_HOME`
  and copy the file anywhere. It sets `PYTHONUTF8=1`. `.venv\Scripts\vision.exe` works too.
- **Use a real console.** Run it in Windows Terminal, PowerShell or cmd. The chat screen needs a real Windows console, so
  mintty (the Git Bash window) and redirected output do not work.
- **Python version.** Use Python 3.12 or 3.13. `onnxruntime-gpu` 1.22 and `webrtcvad-wheels` have no Windows wheels for 3.14.
- **Plain pip.** `pip install -e .[voice]` installs PyPI's torch, which is CPU-only on Windows, so the voice runs several
  times slower than real time. Use uv, or install torch from `https://download.pytorch.org/whl/cu130` first.
- **Paths.** Config, data and state live where they do on Linux, under your user folder:
  - `%USERPROFILE%\.config\vision`
  - `%USERPROFILE%\.local\share\vision`
  - `%USERPROFILE%\.local\state\vision`

## Safer settings

Windows starts in **plan mode** (Linux starts in auto). In plan mode the brain may read and propose,
and nothing changes until you approve. Two more settings under `[brain]` are worth turning on:

```toml
mode = "plan"
plan_approval = "turn"     # an approved plan runs; your next message starts in plan mode again
read_cli_logins = false    # Vision never opens ~/.claude/.credentials.json or ~/.grok/auth.json
```

- **What `read_cli_logins` changes.** With the default `true`, `/usage` shows Claude's banked limit resets and Grok's allowance. It gets them from
  undocumented endpoints, using the logins those CLIs saved on disk. With `false`, `/usage` shows only what the CLIs report themselves.
- **Auto mode.** Auto mode is Claude Code's `--dangerously-skip-permissions`: tools run without asking.
- **`denied_tools` offers little protection on Windows.** It is a prefix match on Bash commands, and the default list only
  names POSIX commands (`sudo`, `rm -rf /`, `mkfs`…). Nothing in it covers `Remove-Item`, `rd /s`, `format`
  or PowerShell. Treat plan mode as the protection, not this list.

## What works

Tested on the machine above:

- **Text chat with Claude.**
  - `vision ask`, and the chat screen's rendering and key-handling tests, in Windows Terminal/conhost.
  - Plan mode is enforced: a write request produced a plan and no file.
  - Cancelling a turn ends Claude's whole process tree (`taskkill /T`), not just `claude.exe`.
- **`vision doctor`, `vision setup`, `vision config --edit`.** The editor falls back to Notepad.
- **Voice on the GPU:**
  - Qwen3-TTS on CUDA (`vision say`, voice design).
  - Whisper large-v3-turbo on CUDA (`vision listen`, from file and from the microphone).
  - A full chain: microphone → transcription → Claude → spoken reply.
  - Playback and capture go through PortAudio's DirectSound/MME devices.
- **Blocking other brains.** A Claude shell cannot start `claude`, `codex` or `grok`. There are `.cmd` shims for
  cmd/PowerShell and `#!/bin/sh` shims for Git Bash.
- **Clipboard copy** from the chat screen, through the Windows clipboard API.

Ported, but not tested on real installs:

- **`vision talk`, the wake word and barge-in.** The keyboard reader uses `msvcrt` on Windows.
- **Codex.** Vision runs the `codex.exe` behind npm's `codex.cmd`. Plan mode relies on Codex's own Windows sandbox.
- **The local llama-server brain.** Its Bash tool runs Git for Windows' `bash.exe`, never
  `System32\bash.exe`, which is the WSL launcher. Set `CLAUDE_CODE_GIT_BASH_PATH` if Git is somewhere unusual.
- **The Orpheus voice engine (`setup --orpheus`).**
  - It uses the Windows CUDA 13.4 build of llama.cpp.
  - A job object stops `llama-server` when Vision exits, in place of the `sh` watchdog used on Linux.

## What doesn't work on Windows

- **`vision serve`, the phone app and `vision --join`: not ported.**
  - The terminal↔server link uses Unix sockets, which CPython does not have on Windows. The chat skips the link quietly.
  - "Open on laptop" launches Linux terminal emulators.
  - The image route allows `/tmp`.
- **Grok with a sandbox (plan mode, or `[grok] sandbox` other than `off`): refused.**
  - Grok always runs with `--always-approve`, and its sandbox is Linux/macOS kernel machinery.
  - Rather than run "read-only" with nothing enforcing it, Vision says so and does not run the turn.
  - Grok in auto mode with `sandbox = "off"` is allowed (the same as the Linux default), but untested.
- **Echo cancellation: none.** There is no PipeWire echo-cancel module on Windows. Use headphones, or keep
  `barge_in = "wake"` so only the wake word interrupts a reply.
- **Hiding the command line in the process list** (`proctitle`): Linux only, and harmless to skip.

## Known issues

- **Model downloads.** Without Developer Mode, the Hugging Face cache stores copies instead of symlinks, so it uses a bit more
  disk. Vision works around a huggingface_hub race that used to fail the download with WinError 1314.
- **VRAM.** 8 GB is tight.
  - `vision setup` loads Whisper right after designing a voice, while that model still holds VRAM, so setup may report "Whisper ready on cpu/small.en".
  - A normal session loads Whisper on CUDA. Close other GPU-heavy apps if the voice falls back to the CPU.
- **`vision doctor` without the voice extra** exits with an install hint at the voice checks (as on Linux).
- **Test suite on Windows.** 480 of 517 pass in a real console. The rest are tests that assume Linux:
  - fake CLIs written as `#!` scripts, `/tmp`, `chmod 0600`, `/proc`, Unix sockets;
  - optional extras that weren't installed (`serve`, `weather`, pytest);
  - the auto-mode default.
  - With mode forced to auto, the local-model, Grok, Codex app-server and memory tests pass.
  - On Linux the suite gives identical results before and after the Windows changes.
- **Quiet speech.** Very quiet speech can lose its first word to the voice detector. Speak at normal volume,
  or pick a closer microphone with `/mic` or `[listen] input_device`.

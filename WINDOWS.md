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

The quick way, from a clone, is the setup script. It installs uv for your user if it is missing (after
asking), needs no administrator rights, and is safe to re-run:

```powershell
git clone https://github.com/domdevz/vision.git; cd vision
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 -All -AddToPath   # or -Voice, -Serve; nothing for text only
```

The same by hand:

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
- **`denied_tools` on Windows.** Claude Code's PowerShell tool is its main shell on Windows, and Bash rules don't
  reach it. So Vision passes every `Bash(x:*)` rule to Claude a second time, as `PowerShell(x:*)`.
  - **The default list on Windows** adds the catastrophic cases: `format`, `diskpart`, `Format-Volume`, `Clear-Disk`, `Remove-Partition`,
    `bcdedit`, `vssadmin delete`, `Stop-Computer` and `Restart-Computer`, and elevation (`runas`, `Start-Process -Verb RunAs`).
  - **It is still a prefix match, not a sandbox.** A command can be reworded or wrapped to get past it. Treat plan mode as the
    protection, and the list as a guard against accidents.

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
- **Echo cancellation.** Windows has no PipeWire echo-cancel module, so Vision cancels its own voice itself.
  Every block it plays is fed to WebRTC's echo canceller (the `livekit` package, part of the voice
  extra), and the microphone's audio passes through it before the voice detector, the wake word and
  Whisper hear it.
  - **Tested through speakers:** in an acoustic loop, its own voice came back about 27 dB quieter, and the voice
    detector stopped mistaking it for you. A second voice in the room still came through and was
    transcribed word for word.
  - **The first reply:** the canceller learns the room from the first second or two of Vision talking, so the very
    first reply of a session can still cut itself off with `barge_in = "speech"`; later ones don't.
  - **Settings:** `[listen] echo_cancel = "auto"` (on for Windows) / `"on"` / `"off"`.

Ported, but not tested on real installs:

- **`vision talk`, the wake word and barge-in.** The keyboard reader uses `msvcrt` on Windows.
- **Codex.** Vision runs the `codex.exe` behind npm's `codex.cmd`. Plan mode relies on Codex's own Windows sandbox.
- **The local llama-server brain.** Its Bash tool runs Git for Windows' `bash.exe`, never
  `System32\bash.exe`, which is the WSL launcher. Set `CLAUDE_CODE_GIT_BASH_PATH` if Git is somewhere unusual.
- **The Orpheus voice engine (`setup --orpheus`).**
  - It uses the Windows CUDA 13.4 build of llama.cpp.
  - A job object stops `llama-server` when Vision exits, in place of the `sh` watchdog used on Linux.
- **`vision serve`, the phone app and `vision --join`.** Pairing and chatting from the iPhone app were tested
  against a Windows machine. The terminal link and joining a phone chat are covered by the test suite.
  - **Link transport.** CPython has no Unix sockets on Windows, so a terminal chat listens on a `127.0.0.1` port with a random
    secret instead. The secret is kept in `%USERPROFILE%\.local\state\vision\live`. A connection that
    doesn't present it gets nothing.
  - **"Open on laptop"** uses Windows Terminal (`wt`) when it is installed, else a new console window.
  - **The server listens on your LAN.** `vision serve` binds to `0.0.0.0` (the LAN, so the phone can reach it). Windows Firewall
    asks the first time: allow **private networks only**, or set `[remote] host = "127.0.0.1"` and use a
    tunnel.
  - **The token file.** Windows has no file mode bits, so `remote_token` is protected by your profile folder's permissions
    rather than `chmod 600`.
  - **It can run anything.** Anyone with the token can drive the brain in whatever mode your config starts in. That is plan on
    Windows by default, but a paired phone can still approve a plan.

## What doesn't work on Windows

- **Grok with a sandbox (plan mode, or `[grok] sandbox` other than `off`): refused.**
  - Grok always runs with `--always-approve`, and its sandbox is Linux/macOS kernel machinery.
  - Rather than run "read-only" with nothing enforcing it, Vision says so and does not run the turn.
  - Grok in auto mode with `sandbox = "off"` is allowed (the same as the Linux default), but untested.
- **Hiding the command line in the process list** (`proctitle`): Linux only, and harmless to skip.

## Known issues

- **Model downloads.** Without Developer Mode, the Hugging Face cache stores copies instead of symlinks, so it uses a bit more
  disk. Vision works around a huggingface_hub race that used to fail the download with WinError 1314.
- **VRAM.** 8 GB is tight.
  - `vision setup` loads Whisper right after designing a voice, while that model still holds VRAM, so setup may report "Whisper ready on cpu/small.en".
  - A normal session loads Whisper on CUDA. Close other GPU-heavy apps if the voice falls back to the CPU.
- **`vision doctor` without the voice extra** exits with an install hint at the voice checks (as on Linux).
- **Test suite on Windows.** It passes, and CI (`.github/workflows/tests.yml`) runs it on Windows and Linux for every
  push and pull request. To run it locally the way CI does:
  `uv sync --extra serve --extra weather`, then `uv pip install numpy soundfile pillow pytest httpx2`, then
  `uv run --no-sync python -m unittest discover -s tests`.
- **Quiet speech.** Very quiet speech can lose its first word to the voice detector. Speak at normal volume,
  or pick a closer microphone with `/mic` or `[listen] input_device`.

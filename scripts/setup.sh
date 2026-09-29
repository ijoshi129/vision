#!/usr/bin/env bash
# Set up Vision on Linux or macOS: a Python 3.12 virtualenv in the checkout, the dependencies and, with
# --voice, the speech models. Needs uv; installs nothing system-wide. Safe to re-run: an existing .venv is
# reused and nothing already installed is removed (except the CUDA libraries when you switch to --cpu).
#
#   scripts/setup.sh [--voice] [--serve] [--all] [--nvidia | --cpu] [--link] [--yes]
#
#   --voice    speech in and out, then `vision setup` for the speech models (about 10 GB, into ~/.cache/huggingface)
#   --nvidia   the voice build for an NVIDIA GPU: CUDA torch and the NVIDIA runtime libraries (about 6 GB)
#   --cpu      the voice build for everything else: CPU-only torch (about 2 GB), several times slower than real time
#              Without either, the script looks for an NVIDIA GPU (nvidia-smi) and asks.
#   --serve    `vision serve`, for the Vision Remote iPhone app
#   --all      voice, serve and weather
#   --link     symlink bin/vision into ~/.local/bin
#   --yes      don't ask: take the detected voice build and go ahead with the downloads
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
voice=0 serve=0 all=0 link=0 yes=0 backend=""

say() { echo "vision setup: $*"; }
usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --voice) voice=1 ;;
        --serve) serve=1 ;;
        --all) all=1 ;;
        --nvidia) backend=nvidia ;;
        --cpu) backend=cpu ;;
        --link) link=1 ;;
        --yes | -y) yes=1 ;;
        -h | --help) usage; exit 0 ;;
        *) say "unknown option $1"; usage; exit 2 ;;
    esac
    shift
done
[ "$all" = 1 ] && voice=1 serve=1
[ -n "$backend" ] && voice=1

confirm() {  # confirm "question" -> 0 for yes
    [ "$yes" = 1 ] && return 0
    local answer
    read -r -p "$1 [y/N] " answer
    [[ "$answer" =~ ^([yY]|[yY][eE][sS])$ ]]
}

uv="$(command -v uv || true)"
if [ -z "$uv" ]; then
    say "uv (the Python package manager Vision uses) is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
command -v claude >/dev/null || say "Claude Code is not installed (https://code.claude.com); Vision can also use Codex, Grok or a local model."

extras=()
if [ "$voice" = 1 ]; then
    gpu=""
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    if [ -z "$backend" ]; then
        detected=cpu && [ -n "$gpu" ] && detected=nvidia
        if [ -n "$gpu" ]; then say "found an NVIDIA GPU: $gpu"; else say "no NVIDIA GPU found (nvidia-smi)"; fi
        backend="$detected"
        if [ "$yes" = 0 ]; then
            read -r -p "Voice build: [n]vidia (CUDA, ~6 GB) or [c]pu (~2 GB)? [${detected:0:1}] " answer
            case "$answer" in
                [nN]*) backend=nvidia ;;
                [cC]*) backend=cpu ;;
            esac
        fi
    elif [ "$backend" = nvidia ] && [ -z "$gpu" ]; then
        say "--nvidia, but no NVIDIA GPU was found: the voice will fall back to the CPU and the CUDA libraries go unused."
    fi
    size=$([ "$backend" = nvidia ] && echo "about 6 GB" || echo "about 2 GB")
    if confirm "Voice ($backend) downloads $size of packages now and about 10 GB of speech models after. Continue?"; then
        extras+=("voice-$backend")
    else
        say "skipping voice; run this again with --voice when you want it."
        voice=0
    fi
fi
[ "$serve" = 1 ] && extras+=(serve)
[ "$all" = 1 ] && extras+=(weather)

cd "$root"
if [ ! -x .venv/bin/python ]; then
    say "creating .venv (Python 3.12; uv downloads it if needed)"
    "$uv" venv --python 3.12 .venv
fi
sync=(sync --frozen --inexact)  # --inexact: a re-run never removes what an earlier one installed
for extra in "${extras[@]}"; do sync+=(--extra "$extra"); done
say "installing: text chat${extras[*]:+ + ${extras[*]}}"
"$uv" "${sync[@]}"

# Switching an NVIDIA build to --cpu: --inexact left the CUDA libraries behind, and nothing uses them now.
if [ "$voice" = 1 ] && [ "$backend" = cpu ]; then
    stale=$("$uv" pip list --python .venv/bin/python --format freeze 2>/dev/null | sed -n 's/^\(nvidia-[^=]*\|triton\)==.*/\1/p')
    if [ -n "$stale" ]; then
        say "removing the CUDA libraries the NVIDIA build left behind"
        # shellcheck disable=SC2086
        "$uv" pip uninstall --python .venv/bin/python $stale
    fi
fi

if [ "$voice" = 1 ]; then
    say "fetching the speech models (vision setup)"
    .venv/bin/python -m vision setup
fi

if [ "$link" = 1 ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$root/bin/vision" "$HOME/.local/bin/vision"
    say "linked $HOME/.local/bin/vision"
    case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) say "~/.local/bin is not on your PATH; add it in your shell profile" ;; esac
fi

if command -v claude >/dev/null; then
    say "checking the install (vision doctor: one small Claude request)"
    .venv/bin/python -m vision doctor --no-usage || true
fi

start=$([ "$link" = 1 ] && echo vision || echo "$root/bin/vision")
say "done. Start a chat with:  $start"

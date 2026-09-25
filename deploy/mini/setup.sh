#!/bin/bash
# One-time setup of the Mac mini as Vision's local-model server, run on the mini as your user
# (or over ssh): bash setup.sh. No Homebrew, no sudo: the llama.cpp release tarball goes under
# ~/.local/llama.cpp, the model under ~/Library/Caches/llama.cpp, the server is a user LaunchAgent.
# Idempotent: re-run after editing the plist.
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_BUILD="${LLAMA_BUILD:-b11064}"  # 2026-09-20; any newer tag works
MODEL_URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
MODEL="$HOME/Library/Caches/llama.cpp/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"

mkdir -p ~/.local/llama.cpp ~/Library/Caches/llama.cpp ~/Library/Logs ~/Library/LaunchAgents
if [ ! -x ~/.local/llama.cpp/llama-server ]; then
  curl -L -o /tmp/llama.tar.gz "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-macos-arm64.tar.gz"
  tar xzf /tmp/llama.tar.gz -C ~/.local/llama.cpp --strip-components=1
  rm /tmp/llama.tar.gz
  xattr -dr com.apple.quarantine ~/.local/llama.cpp 2>/dev/null || true
fi
~/.local/llama.cpp/llama-server --version

if [ ! -f "$MODEL" ]; then
  curl -L -C - -o "$MODEL.part" "$MODEL_URL"   # 16.8 GB, resumable
  mv "$MODEL.part" "$MODEL"
fi

sed "s|__HOME__|$HOME|g" com.vision.llama-server.plist > ~/Library/LaunchAgents/com.vision.llama-server.plist
launchctl bootout "gui/$(id -u)/com.vision.llama-server" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.vision.llama-server.plist

cat <<MSG

llama-server is starting (tail -f ~/Library/Logs/llama-server.log); ready when this answers:
  curl -s localhost:8080/health
By hand, once: System Settings > Users & Groups > Automatic login: on, so the agent comes back
after a reboot with no keyboard attached. Power settings (no sleep, autorestart, wake-on-LAN) and
Remote Login were already on when this was first set up; check with: pmset -g
MSG

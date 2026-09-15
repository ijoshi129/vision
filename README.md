# Vision

A local voice + text AI assistant for the terminal, driven by **your Claude Code subscription**.

- **Text → text**: `vision` (interactive chat) or `vision ask "..."`
- **Text → speech**: `vision say "..."` or chat with `--speak`
- **Speech → text**: `vision listen`
- **Speech → speech**: `vision talk`

Everything except the model's thinking runs on this machine: Kokoro (ONNX) speaks, faster-whisper on
the RTX 5050 listens, and Claude Code's documented headless mode (`claude -p`) does the thinking with
your normal login. No API keys, no scraped tokens, nothing outside Anthropic's supported CLI surface.

## Quick start

```bash
vision                # text chat (Ctrl-D or /quit to leave)
vision --speak        # text chat, replies read aloud
vision talk           # hands-free voice conversation; say "goodbye" to exit
vision talk --ptt     # push-to-talk (Enter to start/stop) for noisy rooms
vision ask "what's the tallest building in Dublin?"
vision say "Good morning, boss."
vision say --out hello.wav "Saved to a file."
vision listen         # transcribe one utterance from the mic
vision listen -f clip.wav
vision voices --preview   # audition the English voices
vision doctor         # check login, models, GPU, audio devices
vision config --edit  # edit ~/.config/vision/config.toml
```

Inside chat: `/speak` toggles voice, `/talk` switches to a spoken conversation, `/listen` speaks one
turn, `/voice bf_emma` changes the voice, `/new` starts a fresh conversation, `/model opus` swaps model.
`vision -c` / `vision talk -c` continues the last conversation.

## The voice

The default preset `friday` is a blend of Kokoro's British female voices (`bf_emma`, `bf_isabella`,
`bf_lily`) tuned toward a calm, crisp, slightly bright delivery. Try alternatives:

```bash
vision say -v bf_emma "Testing."
vision say -v "bf_isabella:0.7,bf_lily:0.3" "Testing a blend."
vision say --speed 1.08 "A touch quicker."
```

Set your favourite in the config under `[voice]`. Vision streams sentences to the speaker while Claude is
still writing, so the first words arrive within a second or so of the reply starting.

## How it talks to Claude

Each turn runs `claude -p --output-format stream-json --resume <session>` inside
`~/.local/share/vision/workspace`, passing Vision's persona via `--append-system-prompt`. Conversation
memory is Claude Code's own session store, so you can even pick a Vision session up with `claude --resume`.
Tools are limited to what `[brain].allowed_tools` lists (web search/fetch and read-only file tools by
default); anything else is denied automatically. Add `"Bash"` there only if you want Vision to run
commands for you.

## Layout

```
vision/brain.py    Claude Code headless driver (streaming, sessions)
vision/tts.py      Kokoro TTS, markdown→speech cleanup, streaming sentence player
vision/stt.py      faster-whisper (CUDA with CPU fallback), hallucination filtering
vision/audio.py    mic capture with WebRTC VAD end-pointing / push-to-talk
vision/cli.py      Typer CLI
vision/persona.py  Vision's system prompt (text vs voice output rules)
vision/config.py   config + paths
```

Data: models in `~/.local/share/vision/models`, Whisper weights in `~/.cache/huggingface`,
config in `~/.config/vision/config.toml`, chat history and last session in `~/.local/state/vision`.

## Reinstall from scratch

```bash
git clone <this repo> ~/Repos/vision && cd ~/Repos/vision
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
install -m755 bin/vision ~/.local/bin/vision
vision setup && vision doctor
```
Requires `espeak-ng` (`sudo dnf install espeak-ng`) for Kokoro's phonemizer fallback.

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
vision usage          # subscription rate-limit windows used
vision doctor         # check login, models, GPU, audio devices
vision config --edit  # edit ~/.config/vision/config.toml
```

In `vision talk` you can also type a message and press Enter at any time instead of speaking.
The chat is a full-screen view like Claude Code's: the conversation scrolls at the top and a framed
message box stays pinned to the bottom (Enter sends, Ctrl-J adds a line, Up/Down recall history,
PgUp/PgDn scroll the transcript, Esc cancels a reply in progress). Your messages appear as highlighted
bands, and replies stream beside a `Vision ›` prefix. `/clear` wipes the screen.
`/model` on its own opens an arrow-key picker (default, Fable, Opus, Sonnet, Haiku); `/model sonnet`
sets one directly. Inside chat: `/speak` toggles voice, `/talk` switches to a spoken conversation, `/listen` speaks one
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
Vision works in the directory you launch it from (or `[brain].workdir`). By default it has the full
toolset: Bash for shell commands (including deleting files), Read/Glob/Grep to read, Write/Edit to
create and change files, and web search/fetch. Headless mode cannot ask for permission, so listed tools
run without prompts; `[brain].denied_tools` blocks patterns such as `sudo`, `rm -rf /`, `mkfs` and `dd`.
Remove `"Bash"`, `"Write"` and `"Edit"` from `allowed_tools` for a read-only assistant.

`vision usage` (or `/usage` inside chat and talk) runs Claude Code's own `/usage` report headlessly and
shows the session window, the weekly all-models window, and per-model windows such as Fable, as bars.
`--full` (or `/usage full`) adds Claude Code's breakdown of what has been contributing.

## Layout

```
vision/brain.py    Claude Code headless driver (streaming, sessions)
vision/tts.py      Kokoro TTS, markdown→speech cleanup, streaming sentence player
vision/stt.py      faster-whisper (CUDA with CPU fallback), hallucination filtering
vision/audio.py    mic capture with WebRTC VAD end-pointing / push-to-talk
vision/cli.py      Typer CLI
vision/ui.py       input box, model picker, message rendering (prompt_toolkit + rich)
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

## Notes

- Kokoro runs on the GPU through `onnxruntime-gpu` 1.22 (the CUDA 12 build, sharing the pip `nvidia-*-cu12`
  libraries with Whisper). A 7-second reply synthesises in about 0.2 s; CPU fallback is automatic
  (`[voice] device = "cpu"` forces it). Newer `onnxruntime-gpu` releases need CUDA 13 libraries instead.
- Hands-free mode listens only after Vision has finished speaking, so laptop speakers work, but a headset
  gives cleaner end-pointing.

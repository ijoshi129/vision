# Vision

A voice and text AI assistant for the terminal, driven by the **Claude Code, Codex or Grok subscription
you already pay for**. No API keys, no scraped tokens: Vision runs the official CLIs headless with your
normal login and puts a proper chat, a voice and a pair of ears around them.

- **Chat** in a full-screen terminal UI with streaming replies, live subagent rows, a plan mode and
  real selectors when the model needs you to choose.
- **Switch brains** mid-conversation with `/model`: Claude, Codex, Grok, or a model on your own
  `llama-server`. The conversation comes with you.
- **Talk to it** hands-free (`vision talk`, or `/talk` in a chat) with a wake word and barge-in. Speech
  recognition (faster-whisper) and the voice (Qwen3-TTS) run locally on your GPU.
- **Give it any voice**: design one from a text description, or clone a few seconds of audio you have
  the right to use.
- **Keep it on a leash**: `denied_tools` patterns, sandboxes for Codex and Grok, and a read-only plan
  mode. Vision also stops one provider from quietly spending another's subscription.

> [!WARNING]
> Vision starts in **auto mode**: the model runs shell commands and edits files without asking, apart
> from the `denied_tools` patterns. Press Shift-Tab (or set `mode = "plan"` under `[brain]`) if you
> would rather approve a plan first. Only run it where you would run the underlying CLI unattended.

## Requirements

| For | You need |
| --- | --- |
| Text chat | Python 3.11+, Linux (macOS should work but is untested), and at least one brain: [Claude Code](https://code.claude.com), the [Codex CLI](https://github.com/openai/codex), the Grok CLI, any OpenAI-compatible server (Ollama, LM Studio, `llama-server`, a hosted API), or an agent that speaks ACP |
| Voice | PipeWire and a microphone. Best on an NVIDIA GPU with 8 GB+ of VRAM; without one it runs on the CPU, several times slower than real time. Up to about 10 GB of downloads for the speech models |
| The voice conversation model | A Claude Code login, or a local model (see [the guide](docs/guide.md#the-voice)) |
| Weather | Optional: an Apple Developer account for WeatherKit. Without it Vision searches the web |

## Install

```bash
git clone https://github.com/domdevz/vision.git && cd vision
scripts/setup.sh --all --link     # or --voice, --serve; nothing for text chat only
```

For voice, the script looks for an NVIDIA GPU and asks which build to install: `--nvidia` (CUDA,
about 6 GB, Linux only) or `--cpu` (about 2 GB; macOS always gets this one). Pass either to skip the
question. It then fetches the speech models (`vision setup`) and runs `vision doctor`, which checks the
CLIs, logins, GPU and audio devices.

The same by hand:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --inexact --extra voice-nvidia --extra serve --extra weather   # or --extra voice-cpu
ln -s "$PWD/bin/vision" ~/.local/bin/vision               # any directory on your PATH
vision setup      # fetches the speech models (skip for text only)
vision doctor
```

The extras are `voice-nvidia` or `voice-cpu` (speech in and out; pick one), `serve` (the remote API),
`weather` (Apple WeatherKit), and `all` / `all-cpu` for everything. `voice` is the old name for
`voice-nvidia`. Install the voice with uv, not plain pip: only uv takes torch from PyTorch's CPU or
CUDA index. If a command needs an extra you don't have, Vision tells you which to install.

Log in to whichever brains you want with their own tools first: `claude`, `codex login`, `grok login`.
No subscription? `vision provider setup` walks you through it: a CLI (installed and logged in for you
on request), Ollama or any OpenAI-compatible server or API, or an ACP agent. `vision` starts that
setup by itself the first time it runs with nothing ready; `/providers` picks which brains `/model`
offers afterwards.

## Use

```bash
vision                  # chat (Ctrl-D or /quit to leave)
vision talk             # hands-free voice conversation; say "goodbye" to end it
vision ask "what's the tallest building in Dublin?"
vision say "Good morning."
vision listen           # transcribe one utterance
vision default          # pick and save the default model and effort
```

Inside a chat, type `/` for the command menu. The ones you'll use most: `/model`, `/effort`, `/talk`,
`/speak`, `/wake`, `/session`, `/new`, `/cd`, and Shift-Tab to flip between auto and plan mode.

Everything else, including the voice pipeline, the router, provider details and the remote API, is in
**[docs/guide.md](docs/guide.md)**.

## Make it yours

Config lives in `~/.config/vision/config.toml`, written with commented defaults on first run
(`vision config --edit` opens it).

- **Personality.** Vision ships with a direct, lightly witty voice. To change it, write your own in
  `~/.config/vision/personality.md`: a paragraph on how it should talk plus a few `User:` / `Vision:`
  example exchanges. It replaces the built-in one outright; delete the file to go back.
- **What it calls you.** `address_user_as` under `[brain]`, e.g. your first name. Empty by default.
- **Its voice.** `vision voice design <name> "<description>"` invents one; `vision voice add <name>
  --from clip.wav` clones a recording; `/voice <name>` switches. Name three clones
  `<name>-conversational`, `<name>-expressive` and `<name>-reassuring`, set `voice =
  "<name>-conversational"`, and each reply is spoken in whichever style suits it.
- **Memory.** Tell it to remember something and it appends a line to
  `~/.local/share/vision/memory/MEMORY.md`, which every brain reads.
- **A model at home.** `deploy/local-model` sets up `llama-server` on a Mac (Apple silicon) as a free,
  private brain; point `[local].base_url` at it. Any other OpenAI-compatible server (Ollama, LM Studio,
  vLLM, OpenRouter, a hosted API) is a `[providers.<name>]` table away, and `/providers` picks which
  providers `/model` offers.

## Privacy

The model's thinking happens wherever your chosen brain runs (Anthropic, OpenAI, xAI, or your own
`llama-server`). Speech recognition and the voice run on your machine, and nothing else leaves it
unless you turn on the weather, web search, or `vision serve`. `vision serve` needs a bearer token for
every request; put it behind Tailscale or a similar private network rather than the open internet.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

[GPL-3.0-or-later](LICENSE). Qwen3-TTS, faster-whisper and the other models Vision downloads keep their
own licences.

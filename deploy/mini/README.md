# Mac mini: Vision's local model

One model, always resident, for both conversation and Codex coding:
**Qwen 3.6 35B-A3B**, Unsloth `UD-Q3_K_XL` (16.8 GB), MTP build, served by `llama-server` on
port 8080. The base M4 (120 GB/s) does ~45 tok/s on this ~3B-active MoE, ~70+ with MTP; dense
models above ~8B crawl on this bandwidth, and Q4 of this model is 22 GB, over the 24 GB budget.

## Mac side (once)
Copy this folder to the mini and run `bash setup.sh` (no Homebrew, no sudo), then turn on
automatic login. The 128k context needs Metal's wired limit raised (one-time sudo):
`sudo cp com.vision.wired-limit.plist /Library/LaunchDaemons/ && sudo chown root:wheel /Library/LaunchDaemons/com.vision.wired-limit.plist && sudo launchctl bootstrap system /Library/LaunchDaemons/com.vision.wired-limit.plist`. First set up 2026-09-20 over ssh.

## Laptop side
Vision talks to the server directly (`vision/local.py`), no CLI in between: `[local].base_url` in
`~/.config/vision/config.toml` names it. `/model qwen3.6` (Local tab) makes it the typed brain and
`[conversation].model = "qwen3.6"` the voice conversation model, at zero subscription cost. As the
typed brain (and as the voice worker when `[brain].model` is `qwen3.6`) it has Bash, Read, Write and
Edit, WebSearch (DuckDuckGo, or Brave with `[local].brave_api_key`) and WebFetch, run by Vision itself
(`vision/localtools.py`, `denied_tools` enforced). The voice
conversation model stays tool-free and delegates, like the Claude one. Effort `high` turns thinking on.
Measured 2026-09-20 from the laptop: 32-39 tok/s, first token 0.2-0.7 s (3-4 s on a cold session
while the server reads the persona), MTP draft acceptance 55-75 %.

## Knobs
- Thinking is decided per request by Vision (`chat_template_kwargs.enable_thinking`); the plist's
  `--reasoning-budget -1` only sets the server default for other clients.
  `launchctl kickstart -k gui/$(id -u)/com.vision.llama-server` restarts after edits.
- Memory: macOS caps Metal at 75 % of RAM (18 GB); the 16.8 GB model plus KV cache sits right on
  it. If the log shows Metal allocation failures or the Mac swaps, either install
  `com.vision.wired-limit.plist` as a root LaunchDaemon (lifts the cap to 20 GB) or drop to
  `UD-IQ3_S` (13.7 GB) in the plist.
- `-np 1` is forced by MTP: a voice turn during a long Codex job queues behind it. Without
  MTP, `-np 2` runs both at ~half speed instead.

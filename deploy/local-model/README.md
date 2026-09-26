# A local model on a Mac

Vision's **Local** brain is any OpenAI-compatible `llama-server`. This folder sets one up on an
Apple-silicon Mac as an always-on server for both voice conversation and coding:
**Qwen 3.6 35B-A3B**, Unsloth `UD-Q3_K_XL` (16.8 GB), MTP build, on port 8080. It was sized for a
base M4 Mac mini with 24 GB: the 120 GB/s memory bandwidth gives ~45 tok/s on this ~3B-active MoE
(~70+ with MTP), dense models above ~8B crawl, and Q4 of this model (22 GB) doesn't fit. With more
memory, pick a bigger quant in `setup.sh` and the plist.

Any other machine running `llama-server` works too; skip straight to [Vision side](#vision-side).

## Mac side (once)

Copy this folder to the Mac and run `bash setup.sh` (no Homebrew, no sudo). It installs a prebuilt
llama.cpp under `~/.local/llama.cpp`, downloads the model and starts the server as a user
LaunchAgent. Turn on automatic login so it comes back after a reboot.

The 128k context needs Metal's wired-memory limit raised, a one-time sudo (the plist sets 21504 MB,
right for 24 GB; adjust it for your RAM):

```bash
sudo cp com.vision.wired-limit.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.vision.wired-limit.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.vision.wired-limit.plist
```

## Vision side

Set `[local].base_url` in `~/.config/vision/config.toml` to the server, e.g.
`http://my-mac.local:8080/v1` (a Tailscale name works away from home). Then `/model qwen3.6`
(Local tab) makes it the typed brain, and `[conversation].model = "qwen3.6"` makes it the voice
conversation model, at no subscription cost.

As the typed brain it has Bash, Read, Write, Edit, WebSearch (DuckDuckGo, or Brave with
`[local].brave_api_key`) and WebFetch, run by Vision itself (`vision/localtools.py`, with
`denied_tools` enforced). As the voice model it stays tool-free and delegates, like the Claude one.
Effort `high` turns thinking on. Over a LAN expect 32-39 tok/s and a first token in 0.2-0.7 s
(3-4 s on a cold session while the server reads the persona).

## Knobs

- Thinking is decided per request by Vision (`chat_template_kwargs.enable_thinking`); the plist's
  `--reasoning-budget -1` only sets the server default for other clients.
  `launchctl kickstart -k gui/$(id -u)/com.vision.llama-server` restarts after edits.
- Memory: macOS caps Metal at 75 % of RAM (18 GB on a 24 GB Mac); the 16.8 GB model plus KV cache sits
  right on it. If the log shows Metal allocation failures or the Mac swaps, install the wired-limit
  LaunchDaemon above or drop to `UD-IQ3_S` (13.7 GB) in the plist.
- `-np 1` is forced by MTP: a voice turn during a long coding job queues behind it. Without MTP,
  `-np 2` runs both at ~half speed instead.

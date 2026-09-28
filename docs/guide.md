# Vision guide

The full reference: every command, the voice pipeline, the providers and the optional pieces.
For installing, start with the [README](../README.md).

## Commands

```bash
vision                # text chat (Ctrl-D or /quit to leave)
vision --speak        # text chat, replies read aloud
vision talk           # hands-free voice conversation; say "goodbye" to exit
vision talk --ptt     # push-to-talk (Enter to start/stop) for noisy rooms
vision ask "what's the tallest building in Dublin?"
vision ask -m gpt-6-astra --effort ultra "review this repository"
vision say "Good morning."
vision say --out hello.wav "Saved to a file."
vision listen         # transcribe one utterance from the mic
vision listen -f clip.wav
vision voices --preview   # list and audition the voices
vision voice design jarvis   # invent a voice from a description (built-in: jarvis, narrator, friday)
vision usage          # subscription rate-limit windows used
vision --version      # Vision plus the Claude Code, Codex and Grok CLI versions
vision update         # update those CLIs with their own updaters (or: vision update grok)
vision doctor         # check CLI versions, logins, usage, models, GPU, audio devices
vision default        # pick and save the default model + effort
vision config --edit  # edit ~/.config/vision/config.toml
```

In `vision talk` you can also type a message and press Enter at any time instead of speaking.
While a voice conversation is on, typed input goes to the same conversation model as speech, so one
model answers the whole chat; enabling speech playback alone does not turn a typed message into a
voice conversation.
The chat is a full-screen view like Claude Code's: the conversation scrolls at the top and a framed
message box stays pinned to the bottom and grows as your text wraps (Enter sends, Ctrl-J adds a line,
Up/Down recall your messages in this conversation, Esc cancels a reply in progress). PgUp/PgDn, Shift-↑/↓ or the mouse wheel scroll
the transcript, Alt-Home/Alt-End jump to the top/bottom, and a reply that is still streaming will not
pull you back down while you are reading; scroll to the bottom (or send a message) to follow again.
Mouse support is on, so hold Shift while dragging to select text in the terminal. Your messages appear as highlighted
bands, and replies stream beside a `Vision ›` prefix. `/clear` wipes the screen.
While a reply is being worked out a spinner line says what Vision is doing (`thinking…`, `using Bash…`).
When it delegates to subagents (Claude Code's Agent tool: Explore, Plan, general-purpose…) each one gets
its own block above the reply — `⠋ Explore · what it was asked` with the tool calls it makes listed
underneath as they happen, ending in `✓ … · 3 tools · 4.2s` — so you can watch what they are up to
instead of just seeing `using Agent…`. The block stays in the transcript afterwards.
Vision runs on Opus 5 at high effort out of the box; the input box's bottom-right corner shows the model
typed input actually goes to and its effort (e.g. `Opus 5 (high)`, where the Grok CLI puts its own),
and while `/talk` is on the status line under it names the voice model too. `/model` and `/effort` open arrow-key pickers for Claude,
Codex and Grok models and only show effort levels supported by the selected model. `/model sonnet`,
`/model gpt-6-astra`, `/model grok-4.6`, or `/effort low` set one directly, and
`/model default` puts the session back on the saved default model and effort.
Switching models on the same provider keeps the existing session and its full transcript, with no
summary-generation turn. Claude ↔ Codex ↔ Grok cannot share a session, so Vision reads all saved
user/assistant messages from the previous provider's session files and sends them with your next message
in a fresh session. Provider-specific tool traces are omitted from these cross-provider transfers.
Transfers retain context imported from earlier providers, so chatting between repeated switches keeps
the earlier conversation too.
The legacy `full` suffix (`/model sonnet full`) still works but is no longer needed; old `handoff`
config settings are ignored. The incoming model's normal context limits still apply.
`/default` (or `vision default`, optionally with `--model opus --effort high`) walks through both pickers
and saves the choice to the config file; `/default reset` (or `vision default --reset`) puts it back to
Opus 5 · high.
Vision has two modes, and the bottom-left corner of the status line always shows which one is on:
`⏵⏵ auto` runs every tool without asking (the `denied_tools` patterns still apply); `⏸ plan` is read-only —
Vision reads, searches and runs read-only commands, then presents a plan and a `Carry out this plan?`
selector. Yes switches to auto mode and carries it out in the same turn; No (or typing what should change)
keeps it planning. Shift-Tab toggles the modes, `/mode plan`, `/mode auto` (or `/plan`, `/auto`) set one,
and `mode` under `[brain]` picks the one Vision starts in (auto by default; plan on Windows). With
`plan_approval = "turn"` a Yes carries out only that plan and the next message starts in plan mode again.
Codex and Grok have no Vision-wired plan-approval
UI, so plan mode gives them a read-only sandbox and asks them to present the plan for you to approve by switching to auto.
Banked limit resets and Grok's allowance on the usage page come from undocumented endpoints that Vision
calls with the logins Claude Code and Grok saved on disk; `read_cli_logins = false` under `[brain]` keeps
Vision out of those files. Windows is covered in [WINDOWS.md](../WINDOWS.md).
Inside chat: `/speak` toggles voice, `/talk` starts a hands-free spoken conversation right there in the
chat screen (type to answer instead of speaking; Esc, `/talk` again or saying "goodbye" ends it), `/wake`
turns the wake word on: whenever the chat is idle a tiny Whisper on the CPU listens for "Vision"; say it and
the real ears and voice warm up and it listens (say "Vision, what's the weather?" in one breath and that runs
straight away), then it dozes off again after a quiet follow-up window (`[wake]` in the config: `enabled`
to start that way, `names`, `follow_up_s`; `vision --wake` too), `/listen`
speaks one turn, `/voice narrator` changes the voice, `/mic` picks the microphone from a list (or `/mic snowball`,
`/mic default`; it is saved as `input_device` and a running `/talk` or `/wake` moves over to it), `/new` starts a fresh conversation, `/cd <dir>` changes the working directory for this session (`/cd ~`, `/cd ..`, `/cd -` for the previous one, `~` and relative paths as in a shell; every brain works there from the next reply on, the status row shows it, bare `/cd` says where you are).
You can cut a spoken reply short by voice (`barge_in` under `[listen]`, in `/talk`, `vision talk` and
with the wake word on): `"wake"` (the default) means say "Vision" — a tiny CPU Whisper keeps listening
over Vision's own voice, so it works on speakers, and "Vision, what about Mars?" runs straight away, while
"Vision" alone stops it and it listens; `"speech"` means just start talking (a quarter second of voice
cuts in, so it needs headphones or PipeWire echo cancellation: `pactl load-module module-echo-cancel`,
then `input_device = "echo-cancel"`; on Windows Vision cancels its own voice itself, `[listen] echo_cancel`);
`"off"` leaves it to Esc / Ctrl-C.
Typing `/` pops up the command menu above the input box and it narrows as you type (`/mo` → `/model`);
↑/↓ choose, Enter runs the highlighted command, Tab fills it in, Esc hides the menu. After a command
that takes an argument (`/model `, `/effort `, `/voice `, `/session `) the menu switches to its choices.
`/session` lists earlier Vision conversations in Claude, Codex and Grok tabs (←/→ switches provider)
and resumes the one you pick from the next message on. Same provider keeps that native thread;
another provider continues the transcript here on the current brain. `/session 7eb17c64` resumes by
id prefix (current provider first, then the others), `/session id` shows the current id.
Conversations are read from Claude Code's, Codex's and Grok's own session stores, so nothing extra
is written. `vision -c` / `vision talk -c` continues the last conversation.
Changing between Claude, Codex and Grok with `/model` starts a fresh conversation because their native
thread stores are separate; changing models within one provider keeps that provider's conversation.
`--continue` resumes the most recent thread for whichever provider owns the selected model.
Pip uses the Obsidian design in the bottom-left gutter beside the input box: a graphite shell,
recessed face, and jade eyes. He blinks while idle, passes a highlight across three dots while
thinking, and scans a small cursor while tools run. While listening, the side rails pulse gently
and a waveform follows actual microphone levels. He speaks along with the voice and dozes off
after ten idle minutes. He slides in from the left when a chat opens (`slide_in = false` under
`[buddy]` turns that off). Rename or hide him under `[buddy]` in the config.
During `/talk` and `/listen` warmup, Pip's lower rim fills as speech recognition, voice, and the
optional interruption listener finish loading. The percentage counts ready components, not elapsed
time: it holds during a slow load, excludes failed loads, and resets for the next warmup. Narrow
terminals show the same percentage beside his inline face.

## The voice

Spoken turns (`vision talk`, `/talk`, `/listen`, wake words, barge-in and the iPhone's live voice mode) use
an independent conversational Claude model, defaulting to **Sonnet at low effort**. It handles ordinary
conversation itself. For work requiring files, commands or current facts, it returns a validated
task with an objective, context, constraints and success criteria. The selected CLI agent executes
that task in a separate worker session and returns status, findings, changes, checks and any question.
Only the conversation model's spoken response reaches TTS; worker prose, JSON and tool logs stay private.
Plan approvals still show the actual plan for review. Worker clarification questions return to the
conversation model so it can ask naturally.

The conversational process uses a replacement system prompt, read-only web tools (or `--tools ""` when disabled), an empty MCP configuration,
and safe mode to disable project instructions, hooks, plugins and skills while keeping subscription
authentication. It never resumes a coding session or gains tools when a task is delegated. These
options require a recent Claude CLI; see the [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).
Voice therefore needs a Claude login even when `/model` selects Codex as the worker.

Optional settings in `~/.config/vision/config.toml` (existing configs receive these defaults):

```toml
[conversation]
model = "sonnet"
effort = "low"
web = true          # read-only WebSearch/WebFetch for the voice model; false = no tools at all
max_delegations = 3
timeout_s = 120
timing = false      # optional stage durations in ~/.local/state/vision/voice-timing.jsonl
```

`/model` and `/effort` control the typed agent and voice worker. Name a brain in the request and that
task's worker runs on it instead: "get Opus to fix the tests", "use Codex for this", "have Haiku do it at
low effort". The worker shows as a row above the reply, agent · model effort · description · time · tokens: a
live timer and output tokens so far while it runs, total time and tokens in and out once done. A model name
that is not in the catalogue fails the task rather than quietly running on the default. In talk mode the
chat's own model does the talking when it is a Claude, Codex or Local model; a Grok chat talks through
`[conversation].model` instead (set it in the config) and keeps its model as the worker. Its conversation history stays in memory for the current Vision process, with a bounded
recent context; `/new` or resuming a different session resets it. `--continue` continues the typed
agent's native history, which the voice model can read as context. Private worker sessions do not
replace that continue target or appear in `/session`. The remote server keeps a combined typed/voice
transcript for reconnecting phones until the server stops or the conversation is reset.

Desktop voice sessions reuse an isolated Claude process, sending context once and then only new
turns, worker results and changed context. Complete responses are validated before speech or task
dispatch, and each speech response is immediately flushed into the streaming speaker. The process
closes when desktop voice mode ends, on cancellation or failure, and on a new session. Configuration
changes restart it on the next request. Native context is recycled after twenty responses, when
sent context exceeds 64,000 characters, or when local history is pruned. A fresh connection receives
the bounded local history; failed or interrupted actions are never automatically retried.

To diagnose desktop response delays, set `[conversation].timing = true` and restart Vision.
Each spoken turn appends one JSON line containing only stage labels, relative milliseconds, and
whether the connection was cold. It includes endpoint detection, transcription, model requests,
web calls, delegated work, synthesis and estimated playback. Acknowledgement and answer readiness
are separate events. Playback time includes the output device's reported latency; it is an estimate,
not an acoustic measurement. Wake-word and push-to-talk input may lack a speech-end timestamp.
Paths follow `XDG_STATE_HOME` when set. Diagnostics are off by default and log no conversation content.

A repeatable transport benchmark is available from the repository:

```bash
.venv/bin/python scripts/bench_conversation.py --transport one-shot --count 10 --output /tmp/voice-before.json
.venv/bin/python scripts/bench_conversation.py --transport persistent --count 10 --output /tmp/voice-after.json
```

This uses your Claude subscription and performs fresh weather searches. It reports a cold request
and ten warm requests per category, with median and slowest answer times. It does not record the
microphone or play audio, and excludes transcription, synthesis and speaker latency.

The voice is [Qwen3-TTS 1.7B](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0): it imitates a short
reference clip, so Vision's voice is whatever you give it. Two ways to make one, both saved under
`~/.local/share/vision/voices/<name>/` (`ref.wav` + `ref.txt`, plus `design.txt` for designed voices):

```bash
vision voice design narrator                  # a built-in description; generates takes, you pick one
vision voice design butler "A dry, precise English butler..." --takes 3 --use
vision voice add mine --from me.wav           # clone a recording (transcribed with Whisper); only voices you may use
vision voices --preview                       # list them, hear them
vision say -v narrator "Testing."
```

`[voice] voice` in the config picks the default (or `/voice <name>` in a chat). `clone_mode` chooses
how the clip is imitated: `embedding` (default) keeps its timbre and lets the model shape the delivery,
`context` continues the clip's own audio for its exact pacing and tone (and any roughness in it).
`rate` stretches the pace after synthesis without changing the pitch (1.13 is brisk), `language` sets
what it speaks.

A **styled voice** is three clones of the same speaker named `<name>-conversational`,
`<name>-expressive` and `<name>-reassuring`. Set `voice = "<name>-conversational"` and every reply is
spoken in the style its text suits: reassuring for "don't worry" and apologies, expressive when it
exclaims, conversational otherwise. `~/.local/state/vision/voice.log` records which one fired and why.

In a spoken conversation the reply's first words can take a while (transcription, then the model's own
thinking: a second on a good turn, twenty on a hard one). Rather than dead air, the voice says a short
"One sec." or "Let me see." when nothing has started `filler_after_ms` after you stop talking, and the
reply follows it. The phrases (`filler_phrases` under `[voice]`) are made in your voice at warm-up and cached
under `~/.local/state/vision/fillers`, so they cost nothing on the turn; `filler = false` turns it off.

Vision streams the model rather than waiting for whole clips. `vision/talker.py` replaces the library's
`transformers.generate()` loop: the prompt is prefilled once, then every 80 ms frame is one CUDA-graph
replay of the 1.9B talker step and one of the whole 15-step residual-code sub-model (sampling included,
Gumbel-max so nothing syncs) on static KV caches. Each frame goes straight to the causal codec decoder,
which renders 4 frames for the first piece and 8 thereafter behind 72 frames of context. On an RTX 5050 laptop GPU (8 GB)
that is first audio ~0.2 s after a sentence is ready and ~2× real time end to end, both graphs
sitting at the GPU's memory-bandwidth floor (bf16 weights, ~4.5 GB of VRAM alongside Whisper).

The previous engine, [Orpheus 3B](https://github.com/canopyai/Orpheus-TTS) under a private
`llama-server`, is still there behind `[voice] engine = "orpheus"` (eight built-in voices, `<chuckle>`-style
inline sounds; `vision setup --orpheus` fetches it).

## Weather (Apple WeatherKit)

Entering `/talk` starts the conversation process and prefetches configured home weather during
audio warm-up. Successful reports are cached in memory for two minutes, shared by startup and
follow-up questions. Different locations have separate cache entries; expired reports are not
served after a failed refresh. Say “refresh the weather” to bypass the cache. Simple current-weather
questions request a short answer rather than an unsolicited multi-day forecast.

"Vision, do I need an umbrella?" used to mean a web search inside the voice model, several seconds of
it. With an Apple Developer membership the same answer comes straight from WeatherKit's REST API
(500,000 calls a month are included). Vision spots a weather question in what you said, fetches the
report while the rest of the turn is prepared and hands it to the voice
model as data, so the model never searches. The report only goes as far as the question: "what's it
like out?" gets now, the next twelve hours and today; "tomorrow" adds tomorrow; "the weekend", "this
week" or a day name gets the whole outlook. When the voice model needs a report it wasn't handed
(another place, another day) it asks WeatherKit itself through the `weather` field of its reply. Its
web tools never serve the weather: a Claude voice model's WebSearch or WebFetch call for the weather
(or to a weather site) is refused in code and answered with the WeatherKit report, and so is a
front-end search for it. The typed brain runs `vision weather [place]` for the
same report. Apple requires the attribution, so every report ends with "Apple Weather".

Setup, once, at [developer.apple.com](https://developer.apple.com/account/resources):

1. **Keys → +**: name it, tick WeatherKit, download the `.p8` (only offered once) and save it as
   `~/.config/vision/weatherkit.p8`. Note its Key ID.
2. **Identifiers → + → Services IDs**: an identifier such as `com.yourname.vision.weatherkit`, with
   WeatherKit ticked under its capabilities.
3. In `~/.config/vision/config.toml` under `[weather]`, fill in `team_id` (top right of the account page),
   `service_id`, `key_id`, and where home is: a `location` name (geocoded once through Open-Meteo and
   cached in `~/.local/state/vision/weather-places.json`) or `latitude`/`longitude`/`timezone`.

```bash
vision weather                 # the home report
vision weather Lisbon          # anywhere; "in <Place>" in a spoken request does the same, and a place
                               # named in the last few turns carries over ("time in NYC?" … "and the weather?")
vision weather --week          # tomorrow and the days after too (--tomorrow for just tomorrow)
vision weather --raw | jq .    # the WeatherKit JSON
```

`units = "imperial"` switches to °F and mph; `enabled = false` turns the integration off and the voice
model goes back to searching. New keys can take a few minutes to start returning 200s.

## The front end (router + supervisor)

The conversation model (the local Qwen in `[conversation]`, or a Claude) can be Vision's front end:
it talks to you, answers genuinely basic questions itself, gets the weather from WeatherKit and
current facts from a search-only lookup, and hands everything substantive to an agent that Vision
launches and supervises. The model never runs anything: it has no shell, files or browser, and it
cannot grant a permission. `[router] mode` picks how far this goes:

- `off`: the voice model decides delegation on its own, on the `/model` brain, as it always did.
- `audit` (the default): every request is classified and the decision logged to
  `~/.local/state/vision/routing.jsonl` and shown as a dim `router (audit): …` line; nothing changes.
  Live with it for a while, read the log, then switch on.
- `on`: routes are enforced (`/router on`, `/router on save` to keep it). Typed messages go through the
  front end too unless `typed = false`.

Routing is code, not a prompt (`vision/routing.py`), in this order: your explicit choice; a weather
request; a request for current information (search only); a basic question (the front end answers);
everything else goes to `default_agent` at `default_effort` (Opus 5 · medium), or at `high_effort`
for architecture, hard debugging, repository-wide changes, security-sensitive work, multi-stage
research, work spanning several systems and a request that already failed at medium. Say it or type it:

```
use Opus high for this: refactor the parser     /agent opus --effort high refactor the parser
have Codex handle this                          /agent codex        (applies to the next message, or the last request)
answer this locally with Qwen                   /local what is JSON
only search the web for this                    /search latest python release
                                                /weather Lisbon        /cancel        /router on
```

A choice that cannot be honoured is an error you see, never a quiet substitute: an agent not in
`[router.agents]`, an effort the model does not support, a CLI that is not installed. The supervisor
(`vision/supervisor.py`) launches the agent as a task-mode worker in its own session, shows
`Launching Opus 5 · medium effort` and then real states (`Opus 5 is running tests…`), enforces the
`[router.limits]` timeout, question rounds and token budget, refuses to launch the same request twice
while a run is alive, and reports only what the worker's structured result said: a run with no valid
`completed` result is failed, whatever it wrote in prose. When the agent stops to ask, its question is
shown word for word, the run waits, and your next plain message goes back into that same session (an
explicit command still wins). `[router.permissions] approval` lists commands the agent may not run on
its own (`git push`, package installs, `rm -rf`, `ssh`, deploys…): it comes back with a question, your
`yes` lifts that one rule for that run, and only your answer can. Search results reach the model as
titles, links and snippets marked untrusted; on a search route a task is refused outright, so nothing
in a snippet can start work.

## Model providers

Claude turns run through `claude -p --output-format stream-json --resume <session>`, passing Vision's
persona via `--append-system-prompt`. Claude conversation memory stays in Claude Code's session store.
Auto mode is `--dangerously-skip-permissions` with `denied_tools` enforced by Claude Code's deny rules;
plan mode is Claude Code's own `--permission-mode plan`, and the plan arrives over the control channel as
an `ExitPlanMode` permission request that Vision turns into the approval selector.

Claude can also ask *you* things: its `AskUserQuestion` tool (always on when any tool is allowed) opens a
real selector in the terminal, like Claude Code's own — single or multi-select, several questions at
once (←/→ or Tab between them), 1-9 to pick, Space to tick, Enter to confirm, or type your own answer in
the box. Esc dismisses it and Claude carries on without the answers. The answers travel over Claude
Code's stdio control channel (`--input-format stream-json --permission-prompt-tool stdio`); Codex and
Grok have no equivalent.

GPT turns run through `codex exec --json` and resume with Codex's native thread IDs. Codex does not have
Claude's per-tool allow/deny interface, so the modes map to its sandbox: plan mode is `read-only`, auto
mode is `danger-full-access` unless `[codex].sandbox` names `workspace-write` or `read-only`.
`denied_tools` is included as firm model guidance but cannot be mechanically enforced by Codex.

Codex and Grok both like to announce a tool before using it ("I'm checking the forecast."), which the
persona cannot reliably talk them out of. Vision holds a short one-sentence message back until the next
event and drops it when a tool call follows (`vision/reply.py`); the reply then starts with the answer.
Anything longer than a sentence, such as a warning before a destructive command, streams through as
written, and text that is followed by nothing is always shown.

Grok turns run through `grok --prompt-file --output-format streaming-json --resume <id>`, passing
Vision's persona via `--rules`. Auto mode is `--always-approve` with Grok's kernel sandbox (`off` unless
`[grok].sandbox` names `workspace`, `read-only` or `strict`); plan mode is the read-only sandbox.
`denied_tools` are passed as `--deny` rules, which Grok does enforce. Headless Grok cannot show
AskUserQuestion or ExitPlanMode, so those tools are removed for Vision turns.

Neither brain can spend another subscription behind your back. A turn's shell gets fake `claude`,
`codex` and `grok` commands first on its PATH (`~/.local/share/vision/shims/`, they print "blocked by Vision" and
exit 1), the other providers' CLIs are pointed at an empty config dir (`CODEX_HOME` / `CLAUDE_CONFIG_DIR` /
`GROK_HOME`) so they have no credentials even if the shim is bypassed, and the default `denied_tools` also lists
`Bash(claude:*)`, `Bash(codex:*)` and `Bash(grok:*)`. The persona tells the model to point you at `/model` instead of
trying. Cross-provider work is always an explicit `/model` switch.

All three providers work in the directory where you launch Vision (or `[brain].workdir`). Install and log in
to the provider you want with `claude`, `codex login` or `grok login`; `vision doctor` checks all three
(version, login, and subscription usage; `--no-usage` skips the usage tables).

`vision --version` (or `vision version`, `/version` in chat) lists Vision's version next to the installed
Claude Code, Codex and Grok CLI versions, with an update column (`up to date` / `2.1.280 available`); name one
or more (`vision version codex`, `/version grok`) for a subset. The chat's bottom status bar shows the same for the
provider in use (`Claude Code up to date`, or `Codex 0.156.0 available · /update`); it is checked in the background
once every six hours, and again right after `/update`. Latest versions come from Claude Code's release feed, npm
for Codex, and `grok update --check` for Grok. `vision update` (or `/update`) runs each CLI's own updater — `claude update`,
`codex update`, `grok update` — and reports the before → after version; name one or more providers
(`vision update codex grok`, `/update claude`) to update only those.

`vision usage` (or `/usage` inside chat and talk) shows usage for the active model's provider; pass
`all` (`vision usage all` or `/usage all`) to show every provider together. Claude
uses Claude Code's own `/usage` report; Codex reads subscription windows from its local thread rollout;
Grok reads the grok.com weekly allowance (the shared SuperGrok pool). `--full` adds Claude's
contribution breakdown, or Grok's per-product split, when available.

## Remote (`vision serve`)

`vision serve` starts with no chat open. The phone can open a new chat with its first message or
the new-chat button. Use `--new` to open a fresh chat at startup, or `--continue` to reopen the
last conversation for the selected model's provider.

To pick up new code without cutting anyone off, restart it gracefully: `vision restart`, `r` in the
`vision serve` window, or `/restart` on the phone. It waits until every chat has finished its reply,
then relaunches itself in the same window on the same port; the phone reconnects on its own and chats
come back from their journals. A message that lands in the last instant runs in the new process. `c`
clears the window.

`vision serve` puts the brain, ears and voice behind an HTTP + WebSocket server on port 8765
for a remote client (the author's **Vision Remote** iPhone app, which is not part of this repository): text chat with streamed replies, dictation
through the laptop's Whisper, replies read aloud sentence by sentence through the voice engine, AskUserQuestion
forms, model switching and resuming conversations. On the same Wi-Fi the phone talks to the laptop
directly (the QR carries the LAN address). Away from home, Tailscale Serve (`tailscale serve --bg 8765`)
carries it to your own tailnet devices only, and Vision puts its https address in the QR; with
`[remote] host = "127.0.0.1"` nothing listens on the LAN at all. Tailscale Funnel or a Cloudflare Tunnel
publish it to the whole internet instead, so prefer Serve. Every request needs the bearer token in `~/.config/vision/remote_token`. `vision serve` prints the address, the token and a QR code
the app scans to pair. The API lives in `vision/server.py` if you want to write your own client.

A `vision` chat open in a terminal on the same machine shows up in the app's chat list too (terminal
icon): the terminal announces itself on a Unix socket under `~/.local/state/vision/live/` (on Windows,
which has no Unix sockets, a 127.0.0.1 port plus a secret the connection must present), `vision serve`
picks it up, and what is typed on the phone runs in that terminal while its replies stream to both.
It leaves the list when the terminal quits, and closing it on the phone quits that terminal (`vision/link.py`).

The other way round, a chat started on the phone can be joined from a terminal: while `vision serve`
runs, `/session` in a `vision` chat gets a **Phone** tab listing its chats; pick one and the terminal
follows it (`vision/remote.py`). The server keeps running the conversation, what is typed in either
place shows in both, and the terminal's own chat drops out of the phone's list while it is joined.
`/model` and `/effort` switch the chat on the server; `/new` or resuming another session leaves it
(the phone chat carries on). Resuming a phone chat's session id the old way would give you a second
copy of the conversation; joining gives you the same one.

## Layout

```
vision/brain.py    Claude Code headless driver (streaming, sessions)
vision/codex.py    OpenAI Codex headless driver (JSONL, threads, usage)
vision/grok.py     Grok CLI headless driver (streaming-json, sessions, usage)
vision/conversation.py  tool-free Claude voice conversation and delegation loop
vision/delegation.py    structured task/result contracts and the silent worker prompt
vision/weather.py  Apple WeatherKit client: signed tokens, geocoding, the spoken report the voice loop prefetches
vision/models.py   shared model catalogue, provider routing, effort capabilities
vision/link.py     terminal chats announce themselves so `vision serve` lists and drives them from the phone
vision/remote.py   a terminal joins a chat `vision serve` runs (the phone's): one conversation, read from both
vision/tts.py      Speaker: Qwen3-TTS engine (streamed frames, cloned/designed voices, WSOLA rate) and the
                   Orpheus engine (llama-server + SNAC); markdown→speech cleanup, streaming sentence player
vision/talker.py   CUDA-graph decode loop for the Qwen3-TTS talker (static caches, Gumbel-max sampling)
vision/stt.py      faster-whisper (CUDA with CPU fallback), hallucination filtering
vision/audio.py    mic capture with Silero/WebRTC VAD end-pointing / push-to-talk
vision/wake.py     wake word: a tiny CPU Whisper spots "Vision" at the start of an utterance, hands it to the voice loop
vision/cli.py      Typer CLI
vision/ui.py       input box, model picker, message rendering (prompt_toolkit + rich)
vision/buddy.py    Pip, the input-gutter robot: state machine + sprite frames
vision/persona.py  Vision's system prompt (text vs voice output rules)
vision/config.py   config + paths
vision/server.py   `vision serve`: HTTP + WebSocket API for the iOS app (token, pairing QR, history)
```

Data: models in `~/.local/share/vision/models`, Whisper weights in `~/.cache/huggingface`,
config in `~/.config/vision/config.toml`, chat history and last session in `~/.local/state/vision`.

## Notes

- `vision setup` fetches Qwen3-TTS Base (4.3 GB) into the Hugging Face cache and, if the configured
  voice is one of the built-in designs and does not exist yet, fetches VoiceDesign (4.3 GB) and makes it.
  PyTorch brings its own CUDA 13 runtime (`nvidia-*-cu13`); Whisper (CTranslate2, int8_float16) and the
  Silero VAD stay on the pip `nvidia-*-cu12` libraries with `onnxruntime-gpu` 1.22, and the two coexist.
  `vision setup --orpheus` adds the Orpheus GGUF (2.4 GB), the SNAC decoder and a prebuilt CUDA 13
  llama.cpp. CPU fallback is automatic but several times slower than real time, so only fit for `vision say`.
- Hands-free mode listens only after Vision has finished speaking, so laptop speakers work, but a headset
  gives cleaner end-pointing.

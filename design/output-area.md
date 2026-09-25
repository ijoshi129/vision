# Output area redesign — draft for review

What the transcript looks like today, what Claude Code / Codex / Grok each do
better, and a proposed new look. Nothing here is implemented yet.

## Today

```
                                                              ← header: empty unless "resumed"
what's eating the disk in ~/Downloads?                        ← user: bold on grey19, flush left
⏺ Bash(du -sh ~/Downloads/* | sort -h | tail)                 ← tool: name(detail)
  ⎿  … 10 lines · click to show more                          ← result: only a count
⠋ thinking… · 3.2s                                            ← live line: fixed word + timer
Mostly the Fedora ISO and the TTS tarball. Everything else    ← reply: markdown, flush left,
is under 50 meg.                                                 no marker
2.1s · in 12,345 (11,900 cached) · out 84                     ← footer: every reply
  ⏵⏵ auto  ·  Opus 5 high  ·  ~/Repos/vision        wake off ← status row
```

Weak spots, in the order they bite:

1. **No rhythm.** A user turn and a reply differ only by background. In a long
   scroll-back nothing anchors the eye; code blocks, prose, tool rows and system
   notices all start at column 0.
2. **Tool results say nothing.** `… 10 lines · click to show more` hides the one
   line you wanted. Every other CLI shows a peek.
3. **The working line is flat.** `thinking…` for 40 s tells you nothing; there's
   no "esc to interrupt", no sign tokens are arriving.
4. **Footer noise.** Full token counts under every reply, even a one-liner.
5. **System messages look like replies.** `screen.add(Text(msg, style="dim"))`
   for errors, notices, /memory output, all identical.
6. **Empty header.** Model, cwd, voice and session are only in the status row.

## What to steal

| From | Worth taking | Leave |
|---|---|---|
| **Claude Code** | `>` / `⏺` gutter marks; `⎿` result hook with a short preview + `+N lines (ctrl+o)`; rotating verbs with `(12s · ↓ 1.2k tokens · esc to interrupt)`; coloured `+`/`−` diff previews for Edit; consecutive Reads folded into `Read 3 files` | the welcome box (too tall for a voice assistant that opens 20× a day) |
| **Codex** | `›` user prompt echo; `Worked for 1m 12s` as the *only* per-turn footer; bold dim "reasoning headline" while it thinks; `% context left` on the right of the status row; key hints in the status row rather than the placeholder; `… +12 lines` collapse | the magenta; the "codex" label above every reply |
| **Grok CLI** | tokens-per-session counter in the bar; bright role markers; playful status verbs | boxed tool cards (heavy, and TTS/copy-hostile) |

Vision's own edge none of them have: the reply is *spoken*, and the reveal is
already gated to the voice. That stays untouched; everything below is around it.

## Proposed

```
 ◆ Vision   Opus 5 · high · ~/Repos/vision · speech on · resumed           ← 1 line, dim, accent ◆

 › what's eating the disk in ~/Downloads?                                  ← accent ›, text plain bold

 ⏺ Bash(du -sh ~/Downloads/* | sort -h | tail)
   ⎿ 1.2G  Fedora-Workstation-Live-44.iso                                  ← first 2 lines shown
     680M  qwen3-tts.tar.zst
     … +8 lines · click to expand                                          ← dim italic

 ⠋ having a look… · 3.2s · ↓ 1.2k · esc to stop                           ← rotating verb + hint

 ● Mostly the Fedora ISO and the TTS tarball! Everything else is           ← accent ●, prose
   under 50 meg.                                                              hangs under it

   ```bash
   rm ~/Downloads/Fedora-Workstation-Live-44.iso
   ```
   2.1s                                                                    ← footer: time only

 › ta                                                                      ← short turns stay short

 ● No worries.
   0.8s

 · speech off                                                              ← system notice: dim ·
 ⚠ mic not found, staying on text                                          ← warning: yellow ⚠
 ✗ Codex exited 1: sandbox denied write to /etc                            ← error: red ✗

 ⏸ plan  ·  Opus 5 high  ·  ~/Repos/vision       ⏎ send · ⌃J newline · / commands · wake off
```

### The changes, smallest first

**A. Two-column gutter** (`ui.py` `_reply_grid`, `_ReplyEntry.lines`)
User rows get `› ` in accent, reply rows `● ` in accent, continuation rows two
spaces. Everything else (tools, agents, live line, footer) already carries its
own mark. `dedent_rows` already strips the gutter from a copied selection.
*Note: the `you ›` / `Vision ›` word labels were removed on the 17th; this is
two cells, not sixteen, and Pip's gutter is unaffected.*

**B. Drop the highlighter background on user turns.**
With the `›` mark it's redundant, and `bold on grey19` across a wrapped
paragraph is the heaviest thing on screen. Keep bold.

**C. Tool result preview** (`tool_activity`)
Show the first two non-blank lines under `⎿`, then `… +N lines · click to
expand`. Errors: first line in red. `(no output)` stays as is. Expanded view
unchanged.

**D. Live line** (`_ReplyEntry.lines`, `brain.py` status strings)
- Verbs rotate every ~4 s from a Vision-flavoured list: `having a look…`,
  `on it…`, `thinking…`, `rummaging…`, `nearly there…`. Tool status stays
  literal: `running Bash…`, `reading ui.py…`.
- Append `↓ 1.2k` once output tokens start arriving (where the provider gives
  a running count; Claude Code does, Codex doesn't → omit).
- Append `esc to stop`.

**E. Footer: time only** (`cli._footer`, `end_reply`)
`2.1s`, or `2.1s · cancelled`. Tokens move to:
- the status row right side as `34% ctx` (Codex-style, Claude Code brain only:
  it reports context; Codex/Grok show nothing), and
- `/usage`, which already exists.
*Alternative if you want the counts kept: `2.1s · 12.3k in · 84 out`, k-formatted,
cached figure dropped.*

**F. Open card** (`header_renderable`, `cli.paint_open`)
Rounded cyan box, Pip idle on the left, labeled `model` / `directory` / `voice`
on the right, `◆ Vision` on the cap row, `resumed` when it applies. First block
of an empty transcript; `/clear` puts it back. A resumed thread shows history
instead. Talk and serve print the same card, then their extra dim lines.

**G. System block styles** (new `notice_grid(kind, text)` in `ui.py`; ~12 call
sites in `cli.py`)
`·` dim for info, `⚠` yellow for warnings, `✗` red for errors. Multi-line
output (/memory, /usage) keeps the `·` on the first row and hangs under it.

**H. Status row hints** (`_status_text`, `PLACEHOLDER`)
Placeholder becomes `Message Vision…`. The keys move to the right of the status
row and show only while the input is empty and idle:
`⏎ send · ⌃J newline · / commands`. `wake on/off` stays at the far right.

**I. Hearing line** (`hearing_grid`)
`◉ listening…` in accent while the mic is open, live words italic after it,
then it settles into a normal `›` user row. Replaces the bare trailing `…`.

### Bigger, optional (separate pass)

**J. Fold consecutive reads.** Three `Read` calls in a row → one
`⏺ Read 3 files · ui.py, cli.py, brain.py`. Needs a grouping step over
`e.tools` in `lines()`.

**K. Diff preview for Edit/Write.** `⏺ Edit(vision/ui.py)` → `⎿ +3 −1` and
three coloured lines. `ToolCall` would need to keep `old_string`/`new_string`
from the input; today it only keeps `detail`.

**L. Agent nesting.** `⏺ Explore · find the status row` on top, its
`steps` as dim `⎿ Grep status_fn` rows underneath while it runs (Claude Code
style), collapsing to the `⎿  done · 4 tools · 6.1s` line when done. *(The rows themselves sit where the agent was launched among the prose since 2026-09-20, as tool rows do.)*

### Decisions I'd like from you

1. Gutter marks `›` / `●` — yes, or keep flush left?
2. Footer: time only + `% ctx` in the bar, or keep compact token counts?
3. Code blocks: leave flush and bare (as now), or give them a faint
   `grey11` background so they read apart from prose? (Copy is unaffected
   either way.)

Not proposing: rules between turns, boxes, speaker names, emoji.

---

**Status 2026-09-18:** A–I implemented (`ui.py`, `cli.py`, `brain.py`; tests in
`tests/test_transcript_look.py`). Decisions taken: gutter marks on; footer time
only with `% ctx` in the status row (Claude Code brain only); code blocks left
bare. `using Bash…` kept as the literal tool status (Pip's face keys off it).
J–L still open.

**Status 2026-09-19:** Next pass mocked in `design/cli-next.html`. Pip directions
(Visor recommended, Capsule, Orb, shipped Obsidian) plus a Grok-shaped
scrollback: selectable blocks, Tab between transcript and composer, user
bubbles, quote-bar replies, diamond tool headers, first/last peek. Nothing
implemented yet.

**Status 2026-09-20:** Open card is in (`header_renderable`, `cli.paint_open`).
Pip idle on the left, labeled model / directory / voice on the right, no
`/model to change`. Mock in `design/open.html`.

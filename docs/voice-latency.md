# Voice latency validation — 2026-09-19

## Follow-up: configured WeatherKit and first-question delay

A later check of the configured WeatherKit path measured 1.65 s fetching weather plus 3.09 s
for the first Claude answer (4.74 s total before synthesis). The warmed Phoenix voice generated
its first audio in 0.28 s in an isolated GPU check, without microphone recognition running.

The desktop `/talk` path now starts the CLI and fetches home weather during audio warm-up.
Successful weather reports are cached for 120 seconds, and simple weather answers are kept short.
Two subsequent silent checks produced answers in 2.72 s and 2.16 s, with cached weather available
in under 1 ms. These are small-sample text-to-answer measurements, not an acoustic latency guarantee.
Content-free stage diagnostics were enabled in the user's configuration for the next real voice turn.
The terminal's reply timer includes playback (or time until cancellation), not just initial silence.

## Earlier general web-search comparison

The persistent CLI connection reduced median warm chat answer generation from 3.69 s to
2.27 s (38%). Live weather requests did not show a consistent improvement: their final median
was 9.88 s versus 9.36 s before. First-use latency also remains variable.

Ten warm requests per category and one cold request were measured on this machine using the
configured Sonnet/low conversation model. The baseline used the unchanged pre-implementation
conversation module. The final run used the persistent transport with incremental context.
Weather requests asked for a fresh New York City lookup each time; every final-run weather
request emitted WebSearch start and completion events.

| Request | Before median | Before slowest | Before cold | After median | After slowest | After cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chat | 3.69 | 4.51 | 2.97 | 2.27 | 2.48 | 5.67 |
| web | 9.36 | 11.49 | 11.27 | 9.88 | 22.72 | 9.87 |

All times are seconds from submitting text to receiving a validated answer. These are sequential
live-service measurements, not controlled provider benchmarks. They exclude microphone endpointing,
transcription, synthesis and speaker output. They do not establish a 20% improvement in acoustic
response time. The current voice, worker model/effort and 900 ms silence threshold were preserved.

Use `scripts/bench_conversation.py` to repeat the transport comparison. Enable
`[conversation].timing = true` to collect the complete desktop pipeline during actual conversations;
playback timestamps are estimates based on the device latency, not microphone measurements.

Validation covers connection reuse, incremental context, native-context recycling, timeout,
cancellation, configuration changes, malformed responses, duplicate-task prevention, separate
worker results, immediate synthesis of short replies, and timing logs without conversation content.
The two existing Phoenix style-selection tests fail identically with the timing edits removed;
they expect sentence-level styles where the current speaker chooses a style for a merged chunk.
The existing remote resume test also needs an actual or mocked `old-typed` session; it fails when
that session is absent. These unrelated behaviors were left intact.

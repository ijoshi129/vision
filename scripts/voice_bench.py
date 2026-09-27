"""Time the voice conversation model per brain: first spoken words and the whole reply.

    .venv/bin/python -m scripts.voice_bench [model ...]        (default: sonnet gpt-5.6-luna)

Each model gets the same three turns on one connection (the first is cold), with the weather supplied
as data the way talk mode does. It spends real usage on every model named.
"""
import sys
import threading
import time

from vision.config import load_config
from vision.conversation import validate_response

TURNS = [
    ("hey, how's it going", None),
    ("what's the weather like", "Now: mostly cloudy, 65F. Today: rain from 7pm through the night, high 73F."),
    ("can you fix the failing test in the vision repo", None),
]


def conversation_for(cfg):
    from vision.models import provider_for

    provider = provider_for(cfg.conversation.model)
    if provider == "codex":
        from vision.codex_voice import CodexConversation
        return CodexConversation(cfg)
    if provider == "local":
        from vision.local import LocalConversation
        return LocalConversation(cfg)
    from vision.conversation import ClaudeConversation
    return ClaudeConversation(cfg)


def bench(model: str) -> None:
    cfg = load_config()
    cfg.conversation.model = model
    cfg.conversation.effort = "low" if not model.startswith("qwen") else "off"
    voice = conversation_for(cfg)
    history = []
    try:
        for n, (text, weather) in enumerate(TURNS):
            turn = {"id": str(n), "user": text, "events": []}
            packet = {"history": list(history), "turn": turn, "coding_context": "", "memory": "",
                      "worker_mode": "auto", "channel": "voice", "delegation_allowed": True}
            if weather:
                packet["weather"] = weather
            first = {}
            start = time.monotonic()

            def on_speech(piece, first=first, start=start):
                first.setdefault("t", time.monotonic() - start)

            try:
                reply = validate_response(voice.complete(packet, threading.Event(), on_speech=on_speech))
            except Exception as e:  # noqa: BLE001
                print(f"{model:>14} turn {n + 1}: FAILED {e}")
                return
            total = time.monotonic() - start
            said = reply["speech"].replace("\n", " ")
            task = f"  [task: {reply['task']['objective'][:50]}]" if reply["task"] else ""
            print(f"{model:>14} turn {n + 1}: first words {first.get('t', total):4.1f}s, done {total:4.1f}s  “{said}”{task}")
            history.append({"id": str(n), "user": text, "speech": reply["speech"], "events": []})
    finally:
        voice.close()


if __name__ == "__main__":
    for m in sys.argv[1:] or ["sonnet", "gpt-5.6-luna"]:
        bench(m)

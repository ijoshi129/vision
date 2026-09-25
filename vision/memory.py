"""Vision's own long-term memory, shared by every brain.

One markdown file, one line per fact, under Vision's data dir (not Claude Code's per-project memory
folder, which only the Claude brain would ever see). The persona hands the whole file to whichever
brain is active and tells it how to add to it; /remember and /forget are the manual path.
"""
from __future__ import annotations

from datetime import date

from vision.config import DATA_DIR

MEMORY_DIR = DATA_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
HEADER = "# Vision's memory\n\nOne fact per line. Vision reads this at the start of every conversation, whichever brain is active.\n\n"
MAX_PROMPT_CHARS = 12_000  # past this the file is truncated in the prompt and the brain is asked to tidy it


def read() -> str:
    """The raw file ("" when it does not exist yet)."""
    try:
        return MEMORY_FILE.read_text()
    except FileNotFoundError:
        return ""


def facts() -> list[str]:
    """The bullet lines, without the header."""
    return [ln.rstrip() for ln in read().splitlines() if ln.startswith("- ")]


def remember(fact: str) -> str:
    """Append one fact (dated); returns the line written."""
    fact = " ".join(fact.split())
    if not fact:
        raise ValueError("nothing to remember")
    line = f"- {date.today().isoformat()}: {fact}"
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    text = read()
    if not text:
        text = HEADER
    elif not text.endswith("\n"):
        text += "\n"
    MEMORY_FILE.write_text(text + line + "\n")
    return line


def forget(needle: str) -> list[str]:
    """Drop every fact line containing `needle` (case-insensitive); returns the lines removed."""
    needle = needle.strip().lower()
    if not needle:
        raise ValueError("say what to forget")
    kept, gone = [], []
    for ln in read().splitlines():
        (gone if ln.startswith("- ") and needle in ln.lower() else kept).append(ln)
    if gone:
        MEMORY_FILE.write_text("\n".join(kept).rstrip("\n") + "\n")
    return gone


def prompt_section(text: str | None = None) -> str:
    """The MEMORY block of the persona: the file's contents plus the rules for adding to it."""
    text = read() if text is None else text
    body = "\n".join(ln for ln in text.splitlines() if ln.startswith("- ")).strip()
    tidy = ""
    if len(body) > MAX_PROMPT_CHARS:
        body = body[:MAX_PROMPT_CHARS].rsplit("\n", 1)[0]
        tidy = " It has grown past what fits here and is truncated: consolidate it into fewer lines when you next have a moment."
    contents = f"Its current contents:\n{body}" if body else "It is empty so far."
    return (
        f"MEMORY: your long-term memory is the file {MEMORY_FILE} (Vision's own; it is shared by every brain "
        f"and is not any Claude Code, Codex or Grok memory folder). {contents}{tidy}\n"
        "Add to it when the user asks you to remember something, corrects how you work, confirms an approach worked, "
        "or shares a durable fact about themselves, their machine or their projects. Append one line per fact, "
        "`- YYYY-MM-DD: fact`, using your file or shell tools, and fix or delete lines that turn out wrong. "
        "Do not store secrets, and do not save things this conversation alone needs. Mention a save in a few words, "
        "no more. If you cannot write the file (read-only sandbox, plan mode), tell the user to run /remember <fact>."
    )

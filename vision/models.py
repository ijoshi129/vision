"""Catalogue of the models Vision can think with, on every provider, and what each one supports.

Claude models run through Claude Code (`claude -p`); GPT models run through the OpenAI Codex CLI
(`codex exec`); Grok models run through the Grok CLI (`grok -p` / `--prompt-file`); local models are
served by llama-server on the LAN and spoken to directly over HTTP (`vision.local`). Effort levels
differ per model, and Claude/Codex each have a top tier: "ultracode" on Claude (xhigh effort plus
dynamic multi-agent workflows) and "ultra" on Codex (maximum reasoning plus automatic task
delegation). Grok has no extra top tier beyond its advertised effort list. This module has no UI imports.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

from vision.providers import REGISTRY as _REGISTRY

EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")  # base levels, ascending
THINKING_OFF = "off"  # below "low": no reasoning at all; only the local models offer it (their thinking switch)
CLAUDE_DEFAULT_EFFORT = "high"  # what a Claude model gets when it inherits an empty effort
TOP_TIERS = {
    "ultracode": "xhigh effort plus multi-agent workflows; slowest, most thorough",
    "ultra": "maximum reasoning plus automatic task delegation; slowest, most thorough",
}
EFFORT_WORDS = (THINKING_OFF, *EFFORT_ORDER, *TOP_TIERS)  # every level a user or the voice model may name
EFFORT_DESCRIPTIONS = {
    THINKING_OFF: "no thinking; answers straight away",
    "low": "quickest, least thinking; fine for chat and voice",
    "medium": "balanced",
    "high": "more thinking for harder problems",
    "xhigh": "deep thinking; slow",
    "max": "everything it has got; very slow",
    **TOP_TIERS,
}

CODEX_HOME = os.path.expanduser(os.environ.get("CODEX_HOME", "~/.codex"))
CODEX_MODELS_CACHE = os.path.join(CODEX_HOME, "models_cache.json")
GROK_HOME = os.path.expanduser(os.environ.get("GROK_HOME", "~/.grok"))
GROK_MODELS_CACHE = os.path.join(GROK_HOME, "models_cache.json")
# Typed nickname for a provider's newest model (`/model grok` → Grok's current default); see provider_default.
NICKNAMES = {"grok": "grok"}
PROVIDER_LABELS = {p.name: p.label for p in _REGISTRY.values()}
PROVIDER_CLIS = {p.name: p.product for p in _REGISTRY.values()}


@dataclass(frozen=True)
class ModelInfo:
    provider: str  # "claude" | "codex" | "grok" | "local"
    alias: str  # what cfg.brain.model holds
    label: str  # "Fable 5.1", "GPT-5.6 Sol"
    description: str
    efforts: tuple[str, ...]  # base levels the model honours, ascending; () = no effort setting at all
    top: str | None = None  # "ultracode" | "ultra" | None
    default_effort: str = ""  # Codex: the model's own default reasoning level
    model_id: str = ""  # Claude: what the alias resolves to today ("claude-opus-5-5"), from Claude Code

    def resting_effort(self) -> str:
        """The level to use when no effort is set and this model has one (e.g. after Haiku → Opus)."""
        if not self.efforts:
            return ""
        return self.default_effort if self.default_effort in self.efforts else CLAUDE_DEFAULT_EFFORT

    def levels(self) -> tuple[str, ...]:
        return self.efforts + ((self.top,) if self.top else ())


# Fallback if Claude Code has not been asked yet; normally its own catalogue replaces this list (below).
# Verified on Claude Code 2.1.273: Fable 5.1, Opus 5 and Sonnet 5 honour every level including
# xhigh/max and accept ultracode; Haiku 4.5 has no effort parameter at all (every value is dropped).
_CLAUDE_FALLBACK = [
    ModelInfo("claude", "fable", "Fable 5.1", "most capable, slowest; best for hard problems", EFFORT_ORDER, "ultracode"),
    ModelInfo("claude", "opus", "Opus 5", "very capable, good all-rounder", EFFORT_ORDER, "ultracode"),
    ModelInfo("claude", "sonnet", "Sonnet 5", "fast and strong; best for voice conversations", EFFORT_ORDER, "ultracode"),
    ModelInfo("claude", "haiku", "Haiku 4.5", "fastest and lightest; no effort setting", ()),
]

# Every provider's list comes from the provider itself and is refreshed while Vision runs (clis.refresh_models),
# so a new model, family or effort level shows up without an edit: Claude Code's `initialize` control
# response (cached below), the Codex and Grok CLIs' own caches, and llama-server's /v1/models.
# Tests use the fallbacks and never ask: their expectations name those models.
CATALOGUES_ON = "unittest" not in sys.modules
CLAUDE_MODELS_CACHE = os.path.join(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")), "vision", "claude_models.json")


def _id_label(name: str) -> str:
    """'claude-opus-5-5' → 'Opus 5.5'; 'claude-haiku-4-5-20251001' → 'Haiku 4.5'."""
    parts = name.removeprefix("claude-").split("-")
    if len(parts) > 1 and len(parts[-1]) == 8 and parts[-1].isdigit():
        parts.pop()
    family = [p for p in parts if not p.replace(".", "").isdigit()]
    version = ".".join(p for p in parts if p.replace(".", "").isdigit())
    return " ".join(w.capitalize() for w in family) + (f" {version}" if version else "")


def claude_models_from_catalogue(rows) -> list[ModelInfo]:
    """ModelInfo rows from Claude Code's `initialize` models list. A known family keeps its short alias
    ("fable", even when Claude Code lists "claude-fable-5-1[1m]") and Vision's own description; a new one
    is named by whatever Claude Code's picker would pass as --model. Any model taking xhigh gets ultracode."""
    known = {m.alias: m for m in _CLAUDE_FALLBACK}
    out: list[ModelInfo] = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        value = str(r.get("value") or "").split("[")[0]
        resolved = str(r.get("resolvedModel") or "").split("[")[0]
        if not value or value == "default" or not resolved.startswith("claude-"):
            continue
        family = resolved.removeprefix("claude-").split("-")[0]
        alias = family if family in known or value == family else value
        if any(m.alias == alias for m in out):
            continue
        levels = (r.get("supportedEffortLevels") or []) if r.get("supportsEffort") else []
        efforts = tuple(lv for lv in EFFORT_ORDER if lv in levels)
        desc = str(r.get("description") or "").partition(" · ")[2].strip().rstrip(".")
        desc = known[alias].description if alias in known else (desc[:1].lower() + desc[1:] if desc else "Claude model")
        out.append(ModelInfo("claude", alias, _id_label(resolved), desc, efforts, "ultracode" if "xhigh" in efforts else None, model_id=resolved))
    order = [m.alias for m in _CLAUDE_FALLBACK]
    out.sort(key=lambda m: order.index(m.alias) if m.alias in order else len(order))
    return out


def _claude_models_from_cache() -> list[ModelInfo] | None:
    if not CATALOGUES_ON:
        return None
    try:
        with open(CLAUDE_MODELS_CACHE, encoding="utf-8") as f:
            return claude_models_from_catalogue(json.load(f).get("models")) or None
    except (OSError, ValueError, AttributeError):
        return None


_cached = _claude_models_from_cache()
CLAUDE_MODELS = _cached or list(_CLAUDE_FALLBACK)
# Providers whose list came from the provider itself (a CLI cache or a fetch), not a fallback above: only
# these can say a saved model is gone (see usable_model).
LIVE: set[str] = {"claude"} if _cached else set()


# Fallback if ~/.codex/models_cache.json is missing; normally the cache overrides this list.
_CODEX_FALLBACK = [
    ModelInfo("codex", "gpt-6-astra", "GPT-6 Astra", "most capable for complex, demanding work", EFFORT_ORDER, "ultra", "low"),
    ModelInfo("codex", "gpt-5.6-sol", "GPT-5.6 Sol", "reliable agentic workhorse for everyday tasks", EFFORT_ORDER, "ultra", "low"),
    ModelInfo("codex", "gpt-5.6-terra", "GPT-5.6 Terra", "balanced agentic model for everyday work", EFFORT_ORDER, "ultra", "medium"),
    ModelInfo("codex", "gpt-5.6-luna", "GPT-5.6 Luna", "fast and affordable", EFFORT_ORDER, None, "medium"),
    ModelInfo("codex", "gpt-5.5", "GPT-5.5", "proven previous-generation model", EFFORT_ORDER[:4], None, "medium"),
]


def _codex_label(display_name: str, slug: str) -> str:
    parts = (display_name or slug).split("-")
    if len(parts) > 1 and parts[-1].isalpha():
        return "-".join(parts[:-1]) + " " + parts[-1]
    return display_name or slug


def _codex_models_from_cache() -> list[ModelInfo] | None:
    """The models Codex itself lists, with their reasoning levels, from its local cache."""
    if not CATALOGUES_ON:
        return None
    try:
        with open(CODEX_MODELS_CACHE, encoding="utf-8") as f:
            data = json.load(f)
        rows = [m for m in data.get("models", []) if m.get("visibility", "list") == "list" and m.get("slug")]
    except (OSError, ValueError, AttributeError):
        return None
    if not rows:
        return None
    rows.sort(key=lambda m: m.get("priority") if isinstance(m.get("priority"), (int, float)) else 999)
    out = []
    for m in rows:
        levels = [r.get("effort") if isinstance(r, dict) else r for r in (m.get("supported_reasoning_levels") or [])]
        efforts = tuple(lv for lv in EFFORT_ORDER if lv in levels)
        top = "ultra" if "ultra" in levels else None
        desc = (m.get("description") or "").strip().rstrip(".")
        desc = desc[:1].lower() + desc[1:] if desc else "OpenAI model"
        out.append(ModelInfo("codex", m["slug"], _codex_label(m.get("display_name", ""), m["slug"]), desc, efforts, top, m.get("default_reasoning_level", "") or ""))
    return out


_cached = _codex_models_from_cache()
CODEX_MODELS = _cached or list(_CODEX_FALLBACK)
LIVE |= {"codex"} if _cached else set()

# Fallback if ~/.grok/models_cache.json is missing; normally the cache overrides this list.
_GROK_FALLBACK = [
    ModelInfo("grok", "grok-4.6", "Grok 4.6", "SpaceXAI's latest frontier model", EFFORT_ORDER[:4], None, "high"),
    ModelInfo("grok", "grok-4.5", "Grok 4.5", "previous-generation Grok", ("low", "medium", "high"), None, "high"),
]


def _grok_models_from_cache() -> list[ModelInfo] | None:
    """The models Grok itself lists, with their reasoning levels, from its local cache."""
    if not CATALOGUES_ON:
        return None
    try:
        with open(GROK_MODELS_CACHE, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("models") or {}
        rows = []
        for slug, entry in raw.items() if isinstance(raw, dict) else []:
            info = (entry or {}).get("info") or entry or {}
            if not isinstance(info, dict) or info.get("hidden") or not (info.get("id") or slug):
                continue
            rows.append((str(info.get("id") or slug), info))
    except (OSError, ValueError, AttributeError):
        return None
    if not rows:
        return None
    rows.sort(key=lambda r: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", r[0])], reverse=True)  # newest first: 4.10 above 4.9
    out = []
    for slug, info in rows:
        efforts_raw = info.get("reasoning_efforts") or []
        values, default = [], ""
        for item in efforts_raw:
            if isinstance(item, dict):
                val = item.get("value") or item.get("id") or ""
                if item.get("default"):
                    default = val
            else:
                val = str(item)
            if val:
                values.append(val)
        efforts = tuple(lv for lv in EFFORT_ORDER if lv in values)
        desc = (info.get("description") or "").strip().rstrip(".")
        desc = desc[:1].lower() + desc[1:] if desc else "xAI model"
        out.append(ModelInfo("grok", slug, info.get("name") or slug, desc, efforts, None, default or info.get("reasoning_effort") or ""))
    return out or None


_cached = _grok_models_from_cache()
GROK_MODELS = _cached or list(_GROK_FALLBACK)
LIVE |= {"grok"} if _cached else set()

# Self-hosted models on the llama-server in `[local]` (deploy/local-model). Talked to directly over HTTP:
# no CLI, nothing leaves the LAN and no subscription is used. Vision runs the model's tool calls itself
# (Bash, Read, Write, Edit, WebSearch, WebFetch; see vision/localtools.py). The two effort levels are
# the model's thinking switch: off answers straight away, high reasons first (20-40 s on a base M4 Mac mini).
# llama-server has no per-request budget (`reasoning_budget` in the body is ignored, verified b11064),
# so there is nothing in between; low/medium arriving from another model land on off.
_LOCAL_FALLBACK = [
    ModelInfo("local", "qwen3.6", "Qwen 3.6 35B-A3B", "self-hosted; free; shell, files and web search", (THINKING_OFF, "high"), None, THINKING_OFF),
]
LOCAL_MODELS = list(_LOCAL_FALLBACK)
MODELS: list[ModelInfo] = CLAUDE_MODELS + CODEX_MODELS + GROK_MODELS + LOCAL_MODELS

# (name, rows, note) tabs for the /model pickers, one per provider (←/→ switches); rows are
# (value, label, description) and the note is shown beside the tab bar when that tab is active.
_LISTS = {"claude": CLAUDE_MODELS, "codex": CODEX_MODELS, "grok": GROK_MODELS, "local": LOCAL_MODELS}
assert tuple(_LISTS) == tuple(_REGISTRY)
MODEL_TABS = [(p.label, [(m.alias, m.label, m.description) for m in _LISTS[p.name]], p.note) for p in _REGISTRY.values()]
MODEL_CHOICES = [(m.alias, m.label, f"{PROVIDER_LABELS.get(m.provider, m.provider)} · {m.description}") for m in MODELS]


def _swap(provider: str, new: list[ModelInfo]) -> None:
    """Replace one provider's models everywhere, in place (the pickers and routing hold these lists)."""
    _LISTS[provider][:] = new
    MODELS[:] = [m for p in _LISTS for m in _LISTS[p]]
    for name, rows, _note in MODEL_TABS:
        if name == PROVIDER_LABELS[provider]:
            rows[:] = [(m.alias, m.label, m.description) for m in new]
    MODEL_CHOICES[:] = [(m.alias, m.label, f"{PROVIDER_LABELS.get(m.provider, m.provider)} · {m.description}") for m in MODELS]


def set_claude_catalogue(rows) -> bool:
    """Swap in Claude Code's model list and save it for the next launch. False when it gave nothing
    usable (the current list stays)."""
    new = claude_models_from_catalogue(rows)
    if not new:
        return False
    _swap("claude", new)
    LIVE.add("claude")
    if CATALOGUES_ON:
        try:
            os.makedirs(os.path.dirname(CLAUDE_MODELS_CACHE), exist_ok=True)
            tmp = CLAUDE_MODELS_CACHE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"models": rows}, f)
            os.replace(tmp, CLAUDE_MODELS_CACHE)
        except OSError:
            pass
    return True


def reload_cli_cache(provider: str) -> bool:
    """Re-read the Codex or Grok CLI's models cache (after the CLI refreshed it). False if unreadable."""
    read = _REGISTRY[provider].hook("models_cache") if provider in _REGISTRY else None
    new = read() if read else None
    if not new:
        return False
    _swap(provider, new)
    LIVE.add(provider)
    return True


def set_local_models(ids) -> bool:
    """The models llama-server serves (its /v1/models ids). A known one keeps its label and levels; a new
    one gets the same thinking switch (llama-server's enable_thinking; a model without it just ignores it)."""
    known = {m.alias: m for m in _LOCAL_FALLBACK + LOCAL_MODELS}
    new = [known.get(i) or ModelInfo("local", i, i, "self-hosted; free; shell, files and web search", (THINKING_OFF, "high"), None, THINKING_OFF)
           for i in dict.fromkeys(str(i) for i in ids if i)]
    if not new:
        return False
    _swap("local", new)
    LIVE.add("local")
    return True


def provider_default(provider: str) -> str:
    """The model a provider means when none is named: Opus for Claude, else the provider's first listed
    model (Codex's own priority order, Grok's newest, whatever llama-server serves)."""
    models = _LISTS.get(provider) or []
    preferred = _REGISTRY[provider].default_model if provider in _REGISTRY else ""
    if preferred and any(m.alias == preferred for m in models):
        return preferred
    return models[0].alias if models else ""


def _resolve(alias: str) -> str:
    a = alias or ""
    return provider_default(NICKNAMES[a.lower()]) or a if a.lower() in NICKNAMES else a


# The conversation model (what answers spoken input, `[conversation].model`) talks through Claude Code,
# `codex app-server` (vision/codex_voice.py) or straight to llama-server: Grok has no tool-free
# structured-output mode, so only these three can talk.
CONVERSATION_PROVIDERS = tuple(p.name for p in _REGISTRY.values() if p.conversation)


def find(alias: str) -> ModelInfo | None:
    a = _resolve(alias)
    for m in MODELS:
        if m.alias == a:
            return m
    return None


def usable_model(alias: str) -> tuple[str, str]:
    """A saved model name, or its provider's current default when the provider has stopped listing it
    (retired, renamed). Returns (model, note); note is "" when the name stands. Only a list the provider
    itself gave counts: under a fallback list an unlisted name may be fine, so it is left alone."""
    if not alias or find(alias):
        return alias, ""
    provider = provider_for(alias)
    base = alias.split("[")[0]  # Claude's "opus[1m]" style suffix
    if provider not in LIVE or any(base in (m.alias, m.model_id) for m in _LISTS[provider]):
        return alias, ""
    replacement = provider_default(provider)
    if not replacement:
        return alias, ""
    return replacement, (f"{alias} is no longer offered by {provider_label(provider, cli=True)}; "
                         f"using {model_label(replacement)} for now (config.toml unchanged)")


def replace_retired_models(cfg, *, brain: bool = True) -> list[str]:
    """Swap retired models in a loaded Config for their provider's default, in memory only: the agent
    ([brain]), the voice model ([conversation]) and the /agent targets ([router] agents). Returns the notes
    to show. brain=False leaves [brain] alone (a --model given on the command line)."""
    notes = []
    if brain:
        cfg.brain.model, note = usable_model(cfg.brain.model)
        if note:
            cfg.brain.effort, _ = coerce_effort(cfg.brain.model, cfg.brain.effort)
            notes.append(note)
    cfg.conversation.model, note = usable_model(cfg.conversation.model)
    if note:
        cfg.conversation.effort, _ = coerce_effort(cfg.conversation.model, cfg.conversation.effort)
        notes.append(f"voice model: {note}")
    for name, alias in list(cfg.router.agents.items()):
        cfg.router.agents[name], note = usable_model(alias)
        if note:
            notes.append(f"/agent {name}: {note}")
    return notes


def provider_label(provider: str, *, cli: bool = False) -> str:
    """Short name for UI copy (`Claude`) or the CLI product (`Claude Code`)."""
    return (PROVIDER_CLIS if cli else PROVIDER_LABELS).get(provider, provider)


def provider_for(alias: str) -> str:
    """Which CLI runs this model. Unknown names: grok* → Grok, gpt-*/o-series → Codex, else Claude."""
    m = find(alias or "")
    if m:
        return m.provider
    a = _resolve(alias).lower()
    for m in _CLAUDE_FALLBACK + _CODEX_FALLBACK + _GROK_FALLBACK + _LOCAL_FALLBACK:  # dropped from a fetched list since
        if m.alias == a:
            return m.provider
    if a.startswith("grok"):
        return "grok"
    return "codex" if a.startswith(("gpt-", "codex-")) or (len(a) > 1 and a[0] == "o" and a[1].isdigit()) else "claude"


def model_label(alias: str) -> str:
    m = find(alias)
    if m:
        return m.label
    return _id_label(alias) if alias else "default"


def efforts_for(alias: str) -> tuple[str, ...]:
    """Every level the model accepts, top tier last. Unknown model: the base levels."""
    m = find(alias)
    return m.levels() if m else EFFORT_ORDER


def top_tier_for(alias: str) -> str | None:
    m = find(alias)
    return m.top if m else None


def supports_effort(alias: str, effort: str) -> bool:
    return not effort or effort in efforts_for(alias)


def effort_choices(alias: str) -> list[tuple[str, str, str]]:
    """(value, label, description) rows for the /effort picker, limited to what this model supports."""
    m = find(alias)
    levels = efforts_for(alias)
    rows = []
    for lv in levels:
        desc = EFFORT_DESCRIPTIONS.get(lv, "")
        note = _REGISTRY[m.provider].default_effort_note if m and m.provider in _REGISTRY else ""
        if note and lv == m.default_effort:
            desc += f"  ({note})"
        rows.append((lv, lv, desc))
    if not rows:
        rows.append(("", "n/a", f"{model_label(alias)} has no effort setting"))
    return rows


def coerce_effort(alias: str, effort: str) -> tuple[str, str]:
    """Fit an effort to a model. Returns (effort_to_use, note); note is "" when nothing changed.

    A top tier moves to the other provider's top tier when the model has one; otherwise the
    level is clamped down to the highest supported base level. Models with no effort setting get "";
    an empty effort arriving at a model that has levels (Haiku → Opus) becomes that model's resting level.
    """
    m = find(alias)
    name = model_label(alias)
    if not effort and m and m.efforts:
        return m.resting_effort(), f"effort → {m.resting_effort()} ({name} default)"
    if supports_effort(alias, effort):
        return effort, ""
    levels = efforts_for(alias)
    if not levels:
        return "", f"effort {effort} → off ({name} has no effort setting)"
    if effort in TOP_TIERS and m and m.top:
        return m.top, f"effort {effort} → {m.top} ({name}'s top tier)"
    rank = _rank(effort)
    base = [lv for lv in levels if lv not in TOP_TIERS]
    fitted = next((lv for lv in reversed(base) if _rank(lv) <= rank), base[0] if base else "")
    return fitted, f"effort {effort} → {fitted} ({name} supports {', '.join(levels)})"


def _rank(level: str) -> int:
    """Where a level sits on the one scale: off < low … max < the top tiers. Unknown reads as low."""
    if level == THINKING_OFF:
        return -1
    if level in TOP_TIERS:
        return len(EFFORT_ORDER)
    return EFFORT_ORDER.index(level) if level in EFFORT_ORDER else 0

"""Live weather from Apple WeatherKit, so a spoken "what's the weather?" needs no web search.

The REST API is signed with an ES256 JWT from a WeatherKit key in the user's Apple Developer account
(500,000 calls a month come with the membership). Place names are geocoded once through Open-Meteo's
free geocoder and cached; the answer is a short, speakable report the voice model reads back.

Apple requires attribution wherever its weather is shown: every report ends with "Apple Weather".
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from vision.config import CONFIG_DIR, STATE_DIR, WeatherConfig

WEATHERKIT_URL = "https://weatherkit.apple.com/api/v1/weather"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODE_CACHE = STATE_DIR / "weather-places.json"
TOKEN_LIFETIME_S = 50 * 60
REPORT_CACHE_S = 120
ATTRIBUTION = "Apple Weather"

# Words that make a spoken request about the weather. Kept deliberately narrow: "it's boiling in
# here" is not a forecast request, "is it going to rain" is.
_WEATHER_WORDS = re.compile(
    r"\b((?<!under the )weather|forecast|temperature|rain(?:ing|y|fall)?|snow(?:ing|y)?|sunny|sunshine|cloudy|overcast|"
    r"humid(?:ity)?|windy|wind speed|storm(?:y|s)?|thunder|hail|sleet|fog(?:gy)?|drizzle|umbrella|"
    r"degrees|celsius|fahrenheit|how (?:hot|cold|warm|chilly) (?:is it|will it be|is it going to be)|"
    r"(?:hot|cold|warm|chilly|freezing|nice) (?:out|outside|today|tomorrow|tonight|this (?:morning|afternoon|evening|week|weekend)))\b",
    re.IGNORECASE,
)
_PLACE = re.compile(
    r"\b(?:in|for|at|around|over in|up in|down in)\s+"
    r"((?:[A-Z][\w'.-]*|St\.?|de|del|la|le|of|on|the|upon)(?:[ -](?:[A-Z][\w'.-]*|de|del|la|le|of|on|the|upon))*)"
)
_NOT_PLACES = {"The", "Vision", "I", "My", "Our", "This", "That", "A", "An", "It", "There", "Today", "Tomorrow", "Tonight",
               "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March",
               "April", "May", "June", "July", "August", "September", "October", "November", "December", "Celsius", "Fahrenheit"}

_CONDITIONS = {
    "Clear": "clear", "MostlyClear": "mostly clear", "PartlyCloudy": "partly cloudy", "MostlyCloudy": "mostly cloudy",
    "Cloudy": "cloudy", "Haze": "hazy", "Smoky": "smoky", "Foggy": "foggy", "Breezy": "breezy", "Windy": "windy",
    "Drizzle": "drizzle", "Rain": "rain", "HeavyRain": "heavy rain", "Showers": "showers", "ScatteredShowers": "scattered showers",
    "IsolatedThunderstorms": "isolated thunderstorms", "ScatteredThunderstorms": "scattered thunderstorms",
    "Thunderstorms": "thunderstorms", "StrongStorms": "strong storms", "Hail": "hail", "Sleet": "sleet", "Snow": "snow",
    "HeavySnow": "heavy snow", "Flurries": "flurries", "SunFlurries": "sun and flurries", "SunShowers": "sun and showers",
    "Blizzard": "blizzard", "BlowingSnow": "blowing snow", "BlowingDust": "blowing dust", "FreezingDrizzle": "freezing drizzle",
    "FreezingRain": "freezing rain", "WintryMix": "wintry mix", "Frigid": "frigid", "Hot": "hot", "Hurricane": "hurricane",
    "TropicalStorm": "tropical storm", "Tornado": "tornado",
}


class WeatherError(Exception):
    pass


@dataclass
class Place:
    name: str
    latitude: float
    longitude: float
    timezone: str = "UTC"
    country: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}" if self.country and self.country not in self.name else self.name


def is_weather_request(text: str) -> bool:
    return bool(_WEATHER_WORDS.search(text or ""))


_WEEK = re.compile(r"\b(week|weekend|next (few|couple of|\d+) days|coming days|days ahead|"
                   r"mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\btomorrow\b", re.IGNORECASE)


def requested_scope(text: str) -> str:
    """How far ahead the question looks: "today" (the default: now, the next hours, today),
    "tomorrow" (adds tomorrow) or "week" (the whole outlook). A plain "what's the weather?" is
    today; the report never volunteers days nobody asked about."""
    text = text or ""
    if _WEEK.search(text):
        return "week"
    if _TOMORROW.search(text):
        return "tomorrow"
    return "today"


_HOME = re.compile(r"\b(here|outside|out|at home|back home|where I am)\b", re.IGNORECASE)


def place_in_context(prompt: str, recent: "list[str] | tuple[str, ...]" = ()) -> str | None:
    """The place a weather question is about: one named in the prompt, else the last one named in
    the most recent turns ("what time is it in NYC?" then "and the weather?" means NYC), unless the
    prompt points home ("what's it like outside?")."""
    place = requested_place(prompt)
    if place or _HOME.search(prompt or ""):
        return place
    for earlier in reversed(list(recent)[-3:]):
        place = requested_place(earlier)
        if place:
            return place
    return None


def requested_place(text: str) -> str | None:
    """A place named in the request ("weather in Lisbon tomorrow" → "Lisbon"), else None."""
    for m in _PLACE.finditer(text or ""):
        words = [w for w in re.split(r"[ -]", m.group(1)) if w]
        while words and (words[-1] in _NOT_PLACES or words[-1].lower() in ("the", "of", "on", "upon", "de", "la", "le", "del")):
            words.pop()
        if words and words[0] not in _NOT_PLACES:
            return " ".join(words).rstrip(".,?!")
    return None


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _configured(cfg: WeatherConfig) -> None:
    missing = [k for k in ("team_id", "service_id", "key_id") if not getattr(cfg, k, "").strip()]
    if missing:
        raise WeatherError("WeatherKit is not set up: fill in weather." + ", weather.".join(missing) + " in the config.")
    if not key_path(cfg).is_file():
        raise WeatherError(f"WeatherKit key not found at {key_path(cfg)} (download the .p8 from developer.apple.com).")


def key_path(cfg: WeatherConfig) -> Path:
    p = Path(cfg.key_file or "weatherkit.p8").expanduser()
    return p if p.is_absolute() else CONFIG_DIR / p


def make_token(cfg: WeatherConfig, now: float | None = None) -> str:
    """A signed WeatherKit bearer token (ES256; valid for TOKEN_LIFETIME_S)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    _configured(cfg)
    try:
        key = serialization.load_pem_private_key(key_path(cfg).read_bytes(), password=None)
    except (ValueError, TypeError) as e:
        raise WeatherError(f"WeatherKit key at {key_path(cfg)} is not a readable .p8 private key: {e}") from e
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise WeatherError("WeatherKit keys must be EC (P-256) keys from developer.apple.com.")
    now = int(now if now is not None else time.time())
    header = {"alg": "ES256", "kid": cfg.key_id.strip(), "id": f"{cfg.team_id.strip()}.{cfg.service_id.strip()}"}
    claims = {"iss": cfg.team_id.strip(), "iat": now, "exp": now + TOKEN_LIFETIME_S, "sub": cfg.service_id.strip()}
    signing_input = _b64url(json.dumps(header, separators=(",", ":")).encode()) + "." + \
        _b64url(json.dumps(claims, separators=(",", ":")).encode())
    r, s = decode_dss_signature(key.sign(signing_input.encode(), ec.ECDSA(hashes.SHA256())))
    return signing_input + "." + _b64url(r.to_bytes(32, "big") + s.to_bytes(32, "big"))


def _get_json(url: str, headers: dict | None = None, timeout: float = 8) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Vision/0.1", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        raise WeatherError(f"{e.code} from {urllib.parse.urlsplit(url).netloc}: {body or e.reason}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WeatherError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {getattr(e, 'reason', e)}") from e
    except ValueError as e:
        raise WeatherError("weather service returned something that is not JSON") from e


class WeatherKit:
    """Fetches and summarises WeatherKit reports. One instance per process; safe from any thread."""

    def __init__(self, cfg: WeatherConfig, fetch=_get_json):
        self.cfg = cfg
        self._fetch = fetch
        self._token: tuple[str, float] | None = None
        self._lock = threading.Lock()
        self._places: dict[str, dict] | None = None
        self._reports: dict[tuple, tuple[float, dict]] = {}
        self._report_lock = threading.Lock()

    # --- places -----------------------------------------------------------------------------
    def _place_cache(self) -> dict[str, dict]:
        if self._places is None:
            try:
                self._places = json.loads(GEOCODE_CACHE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._places = {}
        return self._places

    def default_place(self) -> Place:
        cfg = self.cfg
        if cfg.latitude or cfg.longitude:
            return Place(cfg.location or "home", float(cfg.latitude), float(cfg.longitude),
                         cfg.timezone or "UTC", cfg.country_code)
        if cfg.location.strip():
            return self.geocode(cfg.location)
        raise WeatherError("no home location: set weather.location (a place name) or latitude/longitude in the config.")

    def geocode(self, name: str) -> Place:
        key = " ".join(name.lower().split())
        cache = self._place_cache()
        if key in cache:
            return Place(**cache[key])
        query = urllib.parse.urlencode({"name": name, "count": 1, "language": "en", "format": "json"})
        data = self._fetch(f"{GEOCODE_URL}?{query}")
        results = data.get("results") or []
        if not results:
            raise WeatherError(f"I couldn't find a place called {name!r}.")
        r = results[0]
        parts = [r.get("name", name)]
        if r.get("admin1") and r["admin1"] != r.get("name"):
            parts.append(r["admin1"])
        place = Place(", ".join(parts), float(r["latitude"]), float(r["longitude"]),
                      r.get("timezone") or "UTC", r.get("country_code") or "")
        with self._lock:
            cache[key] = place.__dict__
            try:
                GEOCODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
                GEOCODE_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            except OSError:
                pass
        return place

    # --- fetching ---------------------------------------------------------------------------
    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._token[1]:
                self._token = (make_token(self.cfg), time.time() + TOKEN_LIFETIME_S - 5 * 60)
            return self._token[0]

    def raw(self, place: Place, refresh: bool = False) -> dict:
        _configured(self.cfg)
        params = {"dataSets": "currentWeather,forecastDaily,forecastHourly", "timezone": place.timezone}
        if place.country:
            params["dataSets"] += ",weatherAlerts"
            params["countryCode"] = place.country
        lang = (self.cfg.language or "en").strip()
        url = f"{WEATHERKIT_URL}/{lang}/{place.latitude:.4f}/{place.longitude:.4f}?{urllib.parse.urlencode(params)}"
        # Share startup prefetch with the first question. Keep only a short in-memory cache;
        # expired data is never served after a failed refresh.
        key = (url, self.cfg.team_id, self.cfg.service_id, self.cfg.key_id)
        with self._report_lock:
            cached = self._reports.get(key)
            if not refresh and cached and time.monotonic() - cached[0] < REPORT_CACHE_S:
                return cached[1]
            data = self._fetch(url, {"Authorization": f"Bearer {self.token()}"})
            if key not in self._reports and len(self._reports) >= 8:
                self._reports.pop(next(iter(self._reports)))
            self._reports[key] = (time.monotonic(), data)
            return data

    def report(self, place_name: str | None = None, refresh: bool = False, scope: str = "week") -> str:
        _configured(self.cfg)
        place = self.geocode(place_name) if place_name else self.default_place()
        return summarise(self.raw(place, refresh=refresh), place, self.cfg.units, scope)


# --- summarising ---------------------------------------------------------------------------
def _temp(c: float | None, units: str) -> str:
    if c is None:
        return "?"
    return f"{round(c * 9 / 5 + 32)}°F" if units == "imperial" else f"{round(c)}°C"


def _speed(kmh: float | None, units: str) -> str:
    if kmh is None:
        return "?"
    return f"{round(kmh / 1.609)} mph" if units == "imperial" else f"{round(kmh)} km/h"


def _condition(code: str | None) -> str:
    if not code:
        return "unknown"
    return _CONDITIONS.get(code) or re.sub(r"(?<!^)(?=[A-Z])", " ", code).lower()


def _when(iso: str, tz: ZoneInfo) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(tz)


def _rain_window(upcoming: list[dict], ahead: list[dict], now: datetime, tz: ZoneInfo) -> str:
    """When rain starts and, just as important, when it stops: "from about 07:00 until about 14:00",
    "lasting the rest of the day", or "now, easing off about 11:00". `upcoming` is the next 12 hours
    (rain must begin in them to be mentioned), `ahead` the next 24 (where the end is looked for)."""
    def chance(h: dict) -> float:
        return h.get("precipitationChance") or 0

    peak = max(chance(h) for h in upcoming)
    if peak < 0.2:
        return "Rain: none expected in the next 12 hours."
    first_i = next(i for i, h in enumerate(upcoming) if chance(h) >= 0.2)
    kind = _condition(upcoming[first_i].get("conditionCode"))
    if kind in ("unknown", "cloudy", "mostly cloudy", "partly cloudy", "clear", "mostly clear"):
        kind = "rain"
    end = next((h for h in ahead[first_i + 1:] if chance(h) < 0.2), None)
    stops = ""
    if end is not None:
        at = _when(end["forecastStart"], tz)
        stops = "about " + at.strftime("%H:%M") + (" tomorrow" if at.date() != now.date() else "")
    if first_i == 0:
        tail = f"easing off {stops}." if end is not None else "not letting up in the next 24 hours."
        return f"Rain: {kind} now ({round(peak * 100)}% chance over the next hours), {tail}"
    start = _when(upcoming[first_i]["forecastStart"], tz).strftime("%H:%M")
    tail = f" until {stops}." if end is not None else ", lasting the rest of the day."
    return f"Rain: {round(peak * 100)}% chance, {kind} likely from about {start}{tail}"


def summarise(data: dict, place: Place, units: str = "metric", scope: str = "week") -> str:
    """A compact report the voice model can read back verbatim or paraphrase. `scope` is how far
    ahead it goes (see requested_scope): "today" stops after today's line, "tomorrow" adds
    tomorrow, "week" adds the days after. Alerts are always included."""
    try:
        tz = ZoneInfo(place.timezone)
    except (KeyError, ValueError):
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    lines = [f"Live weather for {place.label} at {now.strftime('%H:%M %a %d %b')} (local time):"]
    cur = data.get("currentWeather") or {}
    if cur:
        # Only what is worth saying: the usual humidity, breeze and UV are left out so the model
        # does not narrate them ("muggy with barely any wind").
        bits = [f"{_condition(cur.get('conditionCode'))}, {_temp(cur.get('temperature'), units)}"]
        if cur.get("temperatureApparent") is not None and abs(cur["temperatureApparent"] - cur.get("temperature", 0)) >= 2:
            bits.append(f"feels like {_temp(cur['temperatureApparent'], units)}")
        if (cur.get("humidity") or 0) >= 0.85 and (cur.get("temperature") or 0) >= 20:
            bits.append("humid")
        if (cur.get("windSpeed") or 0) >= 30:
            bits.append(f"windy, {_speed(cur['windSpeed'], units)}")
        if (cur.get("uvIndex") or 0) >= 6:
            bits.append(f"UV {cur['uvIndex']}, strong sun")
        lines.append("- Current: " + ", ".join(bits) + ".")
    hours = (data.get("forecastHourly") or {}).get("hours") or []
    ahead = [h for h in hours if h.get("forecastStart") and _when(h["forecastStart"], tz) >= now][:24]
    upcoming = ahead[:12]
    if upcoming:
        lines.append("- " + _rain_window(upcoming, ahead, now, tz))
    days = (data.get("forecastDaily") or {}).get("days") or []
    shown = {"today": 1, "tomorrow": 2}.get(scope, 2)
    for label, day in zip(("Today", "Tomorrow"), days[:shown]):
        if not day:
            continue
        bits = [f"{label}: {_condition(day.get('conditionCode'))}, high {_temp(day.get('temperatureMax'), units)}, "
                f"low {_temp(day.get('temperatureMin'), units)}"]
        if day.get("precipitationChance") is not None:
            bits.append(f"{round(day['precipitationChance'] * 100)}% chance of {day.get('precipitationType') or 'precipitation'}"
                        if day.get("precipitationType") not in (None, "clear") else "dry")
        if day.get("sunrise") and day.get("sunset") and label == "Today":
            bits.append(f"sun {_when(day['sunrise'], tz).strftime('%H:%M')} to {_when(day['sunset'], tz).strftime('%H:%M')}")
        lines.append("- " + ", ".join(bits) + ".")
    later = days[2:6] if scope == "week" else []
    if later:
        lines.append("- Then: " + "; ".join(
            f"{_when(d['forecastStart'], tz).strftime('%a')} {_condition(d.get('conditionCode'))} "
            f"{_temp(d.get('temperatureMax'), units)}/{_temp(d.get('temperatureMin'), units)}"
            for d in later if d.get("forecastStart")) + ".")
    alerts = (data.get("weatherAlerts") or {}).get("alerts") or []
    for a in alerts[:3]:
        lines.append(f"- ALERT: {a.get('description') or a.get('summary') or 'weather alert'} "
                     f"({a.get('severity', 'unknown')}, {a.get('source', 'official source')}).")
    lines.append(f"Source: {ATTRIBUTION}.")
    return "\n".join(lines)


def prefetch(cfg: WeatherConfig, prompt: str, timeout: float = 6,
             recent: "list[str] | tuple[str, ...]" = ()) -> "threading.Thread | None":
    """Start fetching a report for a weather request. Returns a thread whose .result holds the
    text (or .error a short reason); None when the request is not about the weather or the
    integration is off. `recent` is the user's last few utterances, oldest first, so a place
    named a turn ago carries over. The caller joins it just before it needs the packet."""
    if not cfg.enabled or not is_weather_request(prompt):
        return None
    place = place_in_context(prompt, recent)
    scope = requested_scope(prompt)
    refresh = bool(re.search(r"\b(refresh|check again|fresh lookup)\b", prompt, re.IGNORECASE))

    def run():
        try:
            t.result = shared(cfg).report(place, scope=scope, **({"refresh": True} if refresh else {}))
        except WeatherError as e:
            if place:
                try:  # "weather in Mordor": fall back to home rather than say nothing
                    t.result = shared(cfg).report(None, scope=scope)
                    t.error = str(e)
                    return
                except WeatherError:
                    pass
            t.error = str(e)
        except Exception as e:  # never let the weather break the conversation
            t.error = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=run, daemon=True, name="weather")
    t.result = None
    t.error = None
    t.timeout = timeout
    t.start()
    return t


_shared: dict[int, WeatherKit] = {}


def shared(cfg: WeatherConfig) -> WeatherKit:
    with _shared_lock:
        wk = _shared.get(id(cfg))
        if wk is None:
            _shared.clear()
            wk = _shared[id(cfg)] = WeatherKit(cfg)
        return wk


_shared_lock = threading.Lock()

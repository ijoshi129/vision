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
    r"\b((?<!under the )(?:weather|wheather|weahter|weathr)|forecast|forcast|temperature|rain(?:ing|y|fall)?|snow(?:ing|y)?|"
    r"sunny|sunshine|cloudy|overcast|humid(?:ity)?|muggy|windy|breezy|wind speed|wind ?chill|storm(?:y|s)?|thunder|hail|sleet|"
    r"fog(?:gy)?|drizzle|umbrella|brolly|frost(?:y)?|heat ?wave|heat index|precipitation|uv index|sunrise|sunset|"
    r"hurricane|tornado|blizzard|"
    r"degrees|celsius|fahrenheit|how (?:hot|cold|warm|chilly) (?:is it|will it be|is it going to be)|"
    r"(?:what(?:'s| is) the |the )temps?|temps? (?:out|outside|today|tonight|tomorrow)|"
    r"(?:like|looking|doing) (?:out|outside|out there)|"
    r"(?:need|bring|take|wear|grab|pack) (?:a |an |my |some )?(?:coat|jacket|raincoat|sunscreen|sun cream|gloves|scarf|layers)|"
    r"(?:hot|cold|warm|chilly|freezing|nice|dry|wet|mild|icy|grey|gray|lovely|miserable) "
    r"(?:out|outside|today|tomorrow|tonight|later|this (?:morning|afternoon|evening|week|weekend)))\b",
    re.IGNORECASE,
)
# "temperature" and "degrees" on their own are just as often about a model, an oven or a fever.
_NOT_WEATHER = re.compile(
    r"\b(?:model|llm|gpt|claude|sampling|top[_ -]?[pk]|api|param(?:eter)?s?|config|gpu|cpu|oven|fridge|freezer|body|"
    r"fever|water|bath|room|coffee|tea|cook(?:ing)?|bake|baking|roast|university|college|angle|triangle|rotate|radians?|convert(?:ing)?|latitude)\b",
    re.IGNORECASE,
)
_WEAK = ("temperature", "degrees", "celsius", "fahrenheit")
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
    hits = [m.group(0).lower() for m in _WEATHER_WORDS.finditer(text or "")]
    if not hits:
        return False
    return not (all(h.startswith(_WEAK) for h in hits) and _NOT_WEATHER.search(text))


# --- the web is never the weather source ---------------------------------------------------------
# The conversation model has web tools for everything else; a weather search or a weather site is
# turned back to WeatherKit in code (the model is told why and given the report). These checks are
# stricter than is_weather_request: they judge a terse search query or a URL, and a false hit would
# block an ordinary search.
WEATHER_HOSTS = (
    "weather.com", "accuweather.com", "wunderground.com", "weather.gov", "nws.noaa.gov", "metoffice.gov.uk",
    "weatherbug.com", "windy.com", "meteoblue.com", "yr.no", "openweathermap.org", "weatherapi.com", "open-meteo.com",
    "wttr.in", "weather.us", "theweathernetwork.com", "weatherzone.com.au", "weatheravenue.com", "foreca.com",
    "weather-forecast.com", "timeanddate.com/weather", "bbc.co.uk/weather", "bbc.com/weather", "weather.apple.com",
    "msn.com/en-us/weather", "weathernews.jp", "tomorrow.io", "ventusky.com", "zoom.earth", "weather.metoffice.gov.uk",
)
_CODE_HOSTS = ("github.com", "gitlab.com", "pypi.org", "npmjs.com", "stackoverflow.com", "developer.apple.com")
_SEARCH_WEATHER = re.compile(
    r"\b(?:(?<!under the )weather(?!\s+(?:the|a|an|out|through)\b)|wheather|weahter|"
    r"(?:weather|rain|snow|temperature|pollen|uv|surf|ski|marine|hourly|daily|extended|local|\d+[- ]day|"
    r"today'?s|tonight'?s|tomorrow'?s|weekend) forecasts?|forecasts? (?:for )?(?:today|tonight|tomorrow|this week(?:end)?)|"
    r"(?:will|is|does) it (?:be )?(?:going to )?(?:rain|raining|snow|snowing|hot|cold|sunny|windy|freezing)|"
    r"how (?:hot|cold|warm|chilly) is it|chance of (?:rain|snow|showers|storms|thunderstorms)|"
    r"wind ?chill|heat index|feels like temperature|accuweather|wunderground|weatherbug|wttr)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?:temperature|temps|rain(?:ing|fall)?|snow(?:ing|fall)?|sunny|humidity|windy|wind speed|thunderstorms?|hail|sleet|"
    r"foggy|drizzle|heat ?wave|frost|precipitation|uv index)\b",
    re.IGNORECASE,
)
_LIVE = re.compile(  # case matters only for the place: "rain in Paris", not "rain in the garden"
    r"(?i:\b(?:today|tonight|tomorrow|now|right now|currently|current|this (?:morning|afternoon|evening|week|weekend)|"
    r"next (?:few|\d+) days|hourly|outside|near me|local|mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b)|\b(?:in|at|for|near)\s+[A-Z]",
)
_NOT_WEATHER_SEARCH = re.compile(
    r"\b(?:api|sdk|library|package|npm|pip|github|python|javascript|swift|json|dataset|icon|emoji|stock|shares|ticker|"
    r"recession|company|jobs?|lyrics|song|album|band|movie|film|sales|revenue|earnings|economic|"
    r"economy|market|gdp|inflation|budget|demand|financial|election|polls?|llm|gpt|model|sampling|gpu|cpu|oven|body|fever)\b",
    re.IGNORECASE,
)


def is_weather_search(query: str) -> bool:
    """True when a web search query is after the weather: "London weather tomorrow", "will it rain
    in Paris", "temperature in NYC now". "Weather API python", "Purple Rain lyrics" and "sales
    forecast 2026" are ordinary searches."""
    query = query or ""
    if _NOT_WEATHER_SEARCH.search(query):
        return False
    if any(h in query.lower() for h in WEATHER_HOSTS):
        return True
    return bool(_SEARCH_WEATHER.search(query) or (_CONDITION.search(query) and _LIVE.search(query)))


def is_weather_url(url: str) -> bool:
    """True for a weather site or page: a known forecast host, a /weather or /forecast page, or a
    search-engine URL whose query is a weather search."""
    try:
        parts = urllib.parse.urlsplit(url or "")
    except ValueError:
        return False
    host = (parts.hostname or "").lower().removeprefix("www.")
    where = host + parts.path.lower()
    if any(where == h or where.startswith(h + "/") or host.endswith("." + h) or host == h for h in WEATHER_HOSTS):
        return True
    for value in urllib.parse.parse_qs(parts.query).get("q", []) + urllib.parse.parse_qs(parts.query).get("query", []):
        if is_weather_search(value):
            return True
    if any(host == h or host.endswith("." + h) for h in _CODE_HOSTS):
        return False
    return bool(re.search(r"/(?:weather|forecast)(?:[/?#.-]|$)", parts.path.lower()))


def web_weather(tool: str, tool_input: dict) -> str | None:
    """What a web tool call is really asking the weather for, or None when it is an ordinary lookup:
    the search query or fetched URL that gives it away. `tool` is WebSearch/WebFetch (Claude's names)
    or web_search."""
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    if tool in ("WebFetch", "web_fetch"):
        url = str(tool_input.get("url") or "")
        return url if is_weather_url(url) else None
    query = str(tool_input.get("query") or "")
    domains = [str(d) for d in (tool_input.get("allowed_domains") or tool_input.get("domains") or [])]
    if is_weather_search(query) or any(is_weather_url("https://" + d.split("://")[-1]) for d in domains):
        return query or ", ".join(domains)
    return None


_SEARCH_FILLER = {
    "what", "whats", "what's", "is", "it", "the", "a", "an", "like", "in", "for", "at", "near", "me", "now", "right", "current",
    "currently", "today", "tonight", "tomorrow", "this", "week", "weekend", "morning", "afternoon", "evening", "next", "days",
    "day", "hourly", "daily", "local", "forecast", "forecasts", "weather", "conditions", "temperature", "temps", "will", "be",
    "going", "to", "rain", "raining", "snow", "snowing", "sunny", "humidity", "wind", "windy", "speed", "how", "hot", "cold",
    "warm", "chance", "of", "report", "outside", "later", "and", "or", "degrees", "fahrenheit", "celsius", "live", "update",
    "does", "do", "i", "need", "umbrella", "on", "extended", "radar", "precipitation", "uv", "index", "high", "low",
    "accuweather", "wunderground", "weatherbug", "wttr",
}


def search_place(query: str) -> str | None:
    """The place a weather search is about ("London weather tomorrow" → "London"), else None."""
    place = requested_place(query)
    if place:
        return place
    words = [w for w in re.findall(r"[A-Za-z][\w'.-]*", query or "") if w.lower().rstrip("'s") not in _SEARCH_FILLER
             and w.lower() not in _SEARCH_FILLER and w not in _NOT_PLACES and not _WEEK.fullmatch(w)]
    if words and len(words) <= 5 and any(w[0].isupper() for w in words):
        return " ".join(words).rstrip(".,?!")
    return None


def lookup(cfg: WeatherConfig, place: str | None = None, scope: str = "today") -> str:
    """A WeatherKit report as data for the conversation model, whatever happens: an unknown place
    falls back to home (and says so), and a failure says to tell the user, never to try the web."""
    if not cfg.enabled:
        return "Weather is switched off in Vision's config ([weather] enabled = false): say so. Never look it up on the web."
    scope = scope if scope in ("today", "tomorrow", "week") else "today"
    try:
        try:
            return shared(cfg).report(place or None, scope=scope)
        except WeatherError as e:
            short = re.sub(r"(?:[ ,]+[A-Z]{2,3})+$", "", place or "")  # "Springfield IL" → "Springfield"
            if short and short != place:
                try:
                    return shared(cfg).report(short, scope=scope)
                except WeatherError:
                    pass
            if not place:
                raise
            return shared(cfg).report(None, scope=scope) + f"\n(The place asked for was not available: {e} This is the home report.)"
    except WeatherError as e:
        return f"WeatherKit could not answer ({e}): say so. Never look the weather up on the web."
    except Exception as e:  # never let the weather break the conversation
        return f"WeatherKit could not answer ({type(e).__name__}: {e}): say so. Never look the weather up on the web."


def lookup_many(cfg: WeatherConfig, places: "list[str | None]", scope: str = "today") -> str:
    """One WeatherKit report per place, fetched side by side, in the order asked: "ten cities in Texas"
    is one request. A place that can't be found says so in its slot instead of falling back to home."""
    places = list(dict.fromkeys(places or [None]))[:MAX_PLACES]
    if len(places) == 1 or not cfg.enabled:
        return lookup(cfg, places[0], scope)
    reports: list[str] = [""] * len(places)

    def one(i: int, place: str | None) -> None:
        try:
            reports[i] = shared(cfg).report(place, scope=scope if scope in ("today", "tomorrow", "week") else "today")
        except Exception as e:  # noqa: BLE001  (one bad place doesn't sink the rest)
            reports[i] = f"{place or 'home'}: WeatherKit could not answer ({e})."

    threads = [threading.Thread(target=one, args=(i, p), daemon=True) for i, p in enumerate(places)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    return "\n\n".join(r or f"{p or 'home'}: WeatherKit timed out." for p, r in zip(places, reports))


_WEEK = re.compile(r"\b(week|weekend|next (few|couple of|\d+) days|coming days|days ahead|"
                   r"mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\btomorrow\b", re.IGNORECASE)
_STATES = {  # "Austin, TX": the geocoder knows states by name only
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
# States whose names are not also a big city ("New York", "Washington" are), so asked on their own they
# mean the region.
_REGIONS = {s.lower() for s in _STATES.values()} - {"new york", "washington", "district of columbia"}
MAX_PLACES = 12  # places in one weather request


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
        """A town by name. "Austin, TX" or "Paris, France" picks the match in that state or country
        (Open-Meteo's search takes only the bare name); a bare "Texas" is a region, not a town, and is
        refused rather than answered with a village that shares its name."""
        key = " ".join(name.lower().split())
        town, *where = [p.strip() for p in name.split(",") if p.strip()] or [name.strip()]
        if not where and town.lower() in _REGIONS:
            raise WeatherError(f"{town} is a whole state, not a town: ask for its towns (\"Newark, NJ\").")
        cache = self._place_cache()
        if key in cache:
            return Place(**cache[key])
        query = urllib.parse.urlencode({"name": town, "count": 10, "language": "en", "format": "json"})
        data = self._fetch(f"{GEOCODE_URL}?{query}")
        results = [r for r in data.get("results") or [] if str(r.get("feature_code", "PPL")).startswith("PPL")]
        wanted = [_STATES.get(w.upper().replace(".", ""), w).lower() for w in where]
        if wanted:
            results = [r for r in results if all(
                w in {str(r.get(k) or "").lower() for k in ("admin1", "admin2", "country", "country_code")} for w in wanted)]
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
            if key not in self._reports and len(self._reports) >= MAX_PLACES + 4:
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

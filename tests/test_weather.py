from __future__ import annotations

import base64
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vision.config import Config, WeatherConfig
from vision import weather
from vision.weather import Place, WeatherError, WeatherKit, is_weather_request, make_token, place_in_context, requested_place, requested_scope, summarise


def _b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sample(now: datetime) -> dict:
    hour = now.replace(minute=0, second=0, microsecond=0)
    hours = []
    for i in range(14):
        start = hour + timedelta(hours=i)
        hours.append({"forecastStart": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                      "conditionCode": "Rain" if i in (3, 4) else "Cloudy",
                      "precipitationChance": 0.7 if i in (3, 4) else 0.05, "temperature": 15})
    days = []
    for i, (code, hi, lo) in enumerate([("PartlyCloudy", 19.6, 11.2), ("Rain", 16, 9), ("Clear", 21, 10), ("Showers", 18, 9)]):
        day = (now + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        days.append({"forecastStart": day.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                     "conditionCode": code, "temperatureMax": hi, "temperatureMin": lo,
                     "precipitationChance": 0.6 if code in ("Rain", "Showers") else 0.1,
                     "precipitationType": "rain" if code in ("Rain", "Showers") else "clear",
                     "sunrise": day.replace(hour=6, minute=41).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                     "sunset": day.replace(hour=19, minute=8).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})
    return {"currentWeather": {"conditionCode": "MostlyCloudy", "temperature": 14.4, "temperatureApparent": 11.0,
                               "humidity": 0.82, "windSpeed": 21.3, "uvIndex": 2},
            "forecastHourly": {"hours": hours}, "forecastDaily": {"days": days},
            "weatherAlerts": {"alerts": [{"description": "Yellow warning of wind", "severity": "moderate", "source": "Met Office"}]}}


class RequestParsingTests(unittest.TestCase):
    def test_weather_questions_are_recognised(self):
        for text in ("What's the weather like?", "is it going to rain later", "Do I need an umbrella today?",
                     "how cold is it outside", "give me the forecast for tomorrow", "will it be windy this weekend"):
            self.assertTrue(is_weather_request(text), text)

    def test_other_requests_are_not(self):
        for text in ("fix the bug in Main Street", "I feel under the weather today", "what's the time", ""):
            self.assertFalse(is_weather_request(text), text)

    def test_named_place(self):
        self.assertEqual(requested_place("Is it going to rain in Lisbon tomorrow?"), "Lisbon")
        self.assertEqual(requested_place("weather for New York City please"), "New York City")
        self.assertEqual(requested_place("what's it like in Saint Albans on Monday"), "Saint Albans")
        self.assertIsNone(requested_place("what's the weather like today"))
        self.assertIsNone(requested_place("will it rain in the morning"))

    def test_place_carries_over_from_recent_turns(self):
        recent = ["what time is it in NYC", "cheers"]
        self.assertEqual(place_in_context("and what's the weather", recent), "NYC")
        self.assertEqual(place_in_context("weather in Lisbon", recent), "Lisbon")  # the prompt wins
        self.assertIsNone(place_in_context("what's it like outside", recent))  # pointing home
        self.assertIsNone(place_in_context("what's the weather here", recent))
        self.assertIsNone(place_in_context("what's the weather", ["what time is it in NYC", "a", "b", "c"]))  # too long ago
        self.assertIsNone(place_in_context("what's the weather", []))

    def test_scope_follows_the_question(self):
        for text in ("what's the weather like?", "is it going to rain later", "how cold is it outside", "weather in Lisbon"):
            self.assertEqual(requested_scope(text), "today", text)
        for text in ("give me the forecast for tomorrow", "will it rain tomorrow morning in Leeds"):
            self.assertEqual(requested_scope(text), "tomorrow", text)
        for text in ("will it be windy this weekend", "what's the week looking like", "weather for the next few days",
                     "is Friday going to be dry", "any rain on Saturday"):
            self.assertEqual(requested_scope(text), "week", text)


class TokenTests(unittest.TestCase):
    def setUp(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.key = ec.generate_private_key(ec.SECP256R1())
        pem = self.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        self.key_file = Path(self.tmp.name) / "AuthKey_ABC123.p8"
        self.key_file.write_bytes(pem)
        self.cfg = WeatherConfig(team_id="TEAM123456", service_id="com.example.vision.weather", key_id="ABC123",
                                 key_file=str(self.key_file))

    def test_token_is_a_valid_es256_jwt_with_apple_claims(self):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        token = make_token(self.cfg, now=1_700_000_000)
        header, claims, sig = token.split(".")
        self.assertEqual(json.loads(_b64(header)), {"alg": "ES256", "kid": "ABC123", "id": "TEAM123456.com.example.vision.weather"})
        self.assertEqual(json.loads(_b64(claims)), {"iss": "TEAM123456", "iat": 1_700_000_000,
                                                    "exp": 1_700_000_000 + 3000, "sub": "com.example.vision.weather"})
        raw = _b64(sig)
        self.assertEqual(len(raw), 64)
        der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        self.key.public_key().verify(der, f"{header}.{claims}".encode(), ec.ECDSA(hashes.SHA256()))

    def test_missing_setup_is_explained(self):
        with self.assertRaises(WeatherError) as ctx:
            make_token(WeatherConfig(team_id="T"))
        self.assertIn("service_id", str(ctx.exception))
        with self.assertRaises(WeatherError) as ctx:
            make_token(WeatherConfig(team_id="T", service_id="s", key_id="k", key_file=str(Path(self.tmp.name) / "nope.p8")))
        self.assertIn("nope.p8", str(ctx.exception))

    def test_token_is_cached_and_sent_as_bearer(self):
        calls = []

        def fetch(url, headers=None, timeout=8):
            calls.append((url, headers))
            return {"currentWeather": {"conditionCode": "Clear", "temperature": 20}}

        wk = WeatherKit(WeatherConfig(**{**self.cfg.__dict__, "latitude": 51.5, "longitude": -0.12,
                                         "timezone": "Europe/London", "country_code": "GB", "location": "London"}))
        wk._fetch = fetch
        wk.report()
        wk.report(refresh=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertTrue(calls[0][1]["Authorization"].startswith("Bearer ey"))
        self.assertIn("weatherkit.apple.com/api/v1/weather/en/51.5000/-0.1200?", calls[0][0])
        self.assertIn("weatherAlerts", calls[0][0])
        self.assertIn("countryCode=GB", calls[0][0])


class ReportCacheTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("vision.weather._configured"))
        self.fetch = Mock(return_value={"currentWeather": {"temperature": 20}})
        self.wk = WeatherKit(WeatherConfig(team_id="T", service_id="s", key_id="k"), fetch=self.fetch)
        self.wk.token = Mock(return_value="token")
        self.place = Place("London", 51.5, -0.12)

    def test_reports_are_reused_until_expired(self):
        with patch("vision.weather.time.monotonic", return_value=10):
            self.wk.raw(self.place)
            self.wk.raw(self.place)
        self.assertEqual(self.fetch.call_count, 1)
        with patch("vision.weather.time.monotonic", return_value=131):
            self.wk.raw(self.place)
        self.assertEqual(self.fetch.call_count, 2)

    def test_refresh_and_other_places_do_not_reuse_cached_data(self):
        self.wk.raw(self.place)
        self.wk.raw(self.place, refresh=True)
        self.wk.raw(Place("Paris", 48.86, 2.35))
        self.assertEqual(self.fetch.call_count, 3)

    def test_expired_report_is_not_used_after_network_failure(self):
        with patch("vision.weather.time.monotonic", return_value=10):
            self.wk.raw(self.place)
        self.fetch.side_effect = WeatherError("offline")
        with patch("vision.weather.time.monotonic", return_value=131):
            with self.assertRaises(WeatherError):
                self.wk.raw(self.place)

    def test_explicit_spoken_refresh_bypasses_cache(self):
        wk = Mock()
        wk.report.return_value = "fresh report"
        with patch.object(weather, "shared", return_value=wk):
            thread = weather.prefetch(WeatherConfig(), "refresh the weather")
            thread.join(1)
        wk.report.assert_called_once_with(None, scope="today", refresh=True)


class SummaryTests(unittest.TestCase):
    def test_report_reads_naturally_and_credits_apple(self):
        from zoneinfo import ZoneInfo

        place = Place("London, England", 51.5, -0.12, "Europe/London", "GB")
        text = summarise(_sample(datetime.now(ZoneInfo("Europe/London"))), place, "metric")
        self.assertIn("Live weather for London, England, GB at", text)
        self.assertIn("Current: mostly cloudy, 14°C, feels like 11°C.", text)  # mild humidity, breeze and UV go unsaid
        self.assertRegex(text, r"Rain: 70% chance, rain likely from about \d\d:00 until about \d\d:00")
        self.assertIn("Today: partly cloudy, high 20°C, low 11°C, dry, sun 06:41 to 19:08.", text)
        self.assertIn("Tomorrow: rain, high 16°C, low 9°C, 60% chance of rain.", text)
        self.assertIn("ALERT: Yellow warning of wind (moderate, Met Office).", text)
        self.assertTrue(text.endswith("Source: Apple Weather."))

    def test_scope_trims_the_days_nobody_asked_about(self):
        from zoneinfo import ZoneInfo

        place = Place("London", 51.5, -0.12, "Europe/London", "GB")
        data = _sample(datetime.now(ZoneInfo("Europe/London")))
        today = summarise(data, place, "metric", "today")
        self.assertIn("Today:", today)
        self.assertNotIn("Tomorrow:", today)
        self.assertNotIn("Then:", today)
        self.assertIn("ALERT:", today)  # alerts always ride along
        tomorrow = summarise(data, place, "metric", "tomorrow")
        self.assertIn("Tomorrow:", tomorrow)
        self.assertNotIn("Then:", tomorrow)
        self.assertIn("Then:", summarise(data, place, "metric", "week"))

    def test_imperial_units(self):
        from zoneinfo import ZoneInfo

        place = Place("Boston", 42.3, -71.0, "America/New_York", "US")
        text = summarise(_sample(datetime.now(ZoneInfo("America/New_York"))), place, "imperial")
        self.assertIn("Current: mostly cloudy, 58°F, feels like 52°F.", text)

    def test_rain_window_says_when_it_stops(self):
        from zoneinfo import ZoneInfo

        place = Place("London", 51.5, -0.12, "Europe/London", "GB")
        now = datetime.now(ZoneInfo("Europe/London"))
        data = _sample(now)
        for h in data["forecastHourly"]["hours"][2:]:  # wet from two hours out, right through the sample
            h["precipitationChance"] = 0.8
        text = summarise(data, place, "metric", "today")
        self.assertIn("rain likely from about", text)
        self.assertIn(", lasting the rest of the day.", text)
        for h in data["forecastHourly"]["hours"]:
            h["precipitationChance"] = 0.8
        data["forecastHourly"]["hours"][5]["precipitationChance"] = 0.0
        text = summarise(data, place, "metric", "today")
        self.assertRegex(text, r"Rain: rain now \(80% chance over the next hours\), easing off about \d\d:00")
        for h in data["forecastHourly"]["hours"]:
            h["precipitationChance"] = 0.0
        self.assertIn("Rain: none expected in the next 12 hours.", summarise(data, place, "metric", "today"))

    def test_notable_conditions_are_mentioned(self):
        from zoneinfo import ZoneInfo

        place = Place("London", 51.5, -0.12, "Europe/London", "GB")
        data = _sample(datetime.now(ZoneInfo("Europe/London")))
        data["currentWeather"].update({"temperature": 27, "temperatureApparent": 27, "humidity": 0.9, "windSpeed": 45, "uvIndex": 8})
        text = summarise(data, place, "metric", "today")
        self.assertIn("Current: mostly cloudy, 27°C, humid, windy, 45 km/h, UV 8, strong sun.", text)


class GeocodeTests(unittest.TestCase):
    def test_places_are_geocoded_once_and_cached(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fetch = Mock(return_value={"results": [{"name": "Lisbon", "admin1": "Lisbon", "latitude": 38.7, "longitude": -9.1,
                                                "timezone": "Europe/Lisbon", "country_code": "PT"}]})
        with patch.object(weather, "GEOCODE_CACHE", Path(tmp.name) / "places.json"):
            wk = WeatherKit(WeatherConfig(), fetch=fetch)
            place = wk.geocode("Lisbon")
            self.assertEqual((place.name, place.timezone, place.country), ("Lisbon", "Europe/Lisbon", "PT"))
            again = WeatherKit(WeatherConfig(), fetch=fetch).geocode("  lisbon ")
        self.assertEqual(again, place)
        self.assertEqual(fetch.call_count, 1)

    def test_unknown_place_and_no_default(self):
        with patch.object(weather, "GEOCODE_CACHE", Path(tempfile.mkdtemp()) / "places.json"):
            wk = WeatherKit(WeatherConfig(), fetch=Mock(return_value={"results": []}))
            with self.assertRaises(WeatherError):
                wk.geocode("Mordor")
            with self.assertRaises(WeatherError) as ctx:
                wk.default_place()
        self.assertIn("weather.location", str(ctx.exception))
        with self.assertRaises(WeatherError) as ctx:
            WeatherKit(WeatherConfig(location="London"), fetch=Mock()).report()
        self.assertIn("weather.team_id", str(ctx.exception))


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        weather._shared.clear()
        self.cfg = WeatherConfig(team_id="T", service_id="s", key_id="k", location="London")

    def test_only_weather_requests_fetch(self):
        self.assertIsNone(weather.prefetch(self.cfg, "fix the tests"))
        self.assertIsNone(weather.prefetch(WeatherConfig(enabled=False), "what's the weather"))

    def test_report_and_fallback_to_home(self):
        wk = Mock()
        wk.report.side_effect = lambda place, **kw: "home report" if place is None else (_ for _ in ()).throw(WeatherError("no such place"))
        with patch.object(weather, "shared", return_value=wk):
            t = weather.prefetch(self.cfg, "weather in Mordor")
            t.join(5)
        self.assertEqual(t.result, "home report")
        self.assertEqual(t.error, "no such place")
        wk.report.side_effect = lambda place, **kw: "the report"
        with patch.object(weather, "shared", return_value=wk):
            t = weather.prefetch(self.cfg, "will it rain in Lisbon")
            t.join(5)
        self.assertEqual((t.result, t.error), ("the report", None))
        wk.report.assert_called_with("Lisbon", scope="today")
        with patch.object(weather, "shared", return_value=wk):
            weather.prefetch(self.cfg, "will it rain this weekend").join(5)
        wk.report.assert_called_with(None, scope="week")
        with patch.object(weather, "shared", return_value=wk):
            weather.prefetch(self.cfg, "what's the weather", recent=["what time is it in New York"]).join(5)
        wk.report.assert_called_with("New York", scope="today")

    def test_unexpected_errors_never_escape(self):
        wk = Mock()
        wk.report.side_effect = RuntimeError("boom")
        with patch.object(weather, "shared", return_value=wk):
            t = weather.prefetch(self.cfg, "what's the weather")
            t.join(5)
        self.assertIsNone(t.result)
        self.assertEqual(t.error, "RuntimeError: boom")


class ConversationInjectionTests(unittest.TestCase):
    def setUp(self):
        from vision.conversation import VoiceConversation

        self.cfg = Config()
        self.cfg.weather.team_id = self.cfg.weather.service_id = self.cfg.weather.key_id = "x"
        self.cfg.weather.location = "London"
        self.agent = SimpleNamespace(cfg=self.cfg.brain, provider="claude", session_id="s", workdir="/tmp", ask=Mock())
        self.voice = VoiceConversation(self.cfg, self.agent)
        for target, value in (("vision.conversation.local_handoff", ""), ("vision.memory.facts", [])):
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def _model(self):
        model = Mock()
        model.usage = None
        model.complete.return_value = {"speech": "Grey and mild.", "task": None}
        self.voice.model = model
        return model

    def test_weather_report_rides_along_in_the_packet(self):
        model = self._model()
        wk = Mock()
        wk.report.return_value = "Live weather for London: now cloudy, 14°C.\nSource: Apple Weather."
        with patch.object(weather, "shared", return_value=wk):
            turn = self.voice.ask("what's the weather like?")
        self.assertFalse(turn.is_error)
        packet = model.complete.call_args[0][0]
        self.assertEqual(packet["weather"], wk.report.return_value)
        wk.report.assert_called_once_with(None, scope="today")

    def test_no_weather_field_for_ordinary_chat(self):
        model = self._model()
        with patch.object(weather, "shared") as shared:
            self.voice.ask("how's it going")
        shared.assert_not_called()
        self.assertNotIn("weather", model.complete.call_args[0][0])

    def test_failure_is_reported_as_data_not_speech(self):
        model = self._model()
        wk = Mock()
        wk.report.side_effect = WeatherError("401 from weatherkit.apple.com")
        with patch.object(weather, "shared", return_value=wk):
            turn = self.voice.ask("is it raining?")
        self.assertFalse(turn.is_error)
        self.assertIn("WeatherKit could not answer (401 from weatherkit.apple.com)", model.complete.call_args[0][0]["weather"])
        self.assertEqual(turn.text, "Grey and mild.")


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
from unittest import mock

from vision import cachettl
from vision.sessions import _parse_iso


def user(stamp, **kw):
    return json.dumps({"type": "user", "timestamp": stamp, "message": {"role": "user", "content": "hi"}, **kw}).encode()


def reply(stamp, mid="m1", five=0, hour=0, created=None, model="claude-opus-5-5", **kw):
    usage = {"cache_read_input_tokens": 1000, "cache_creation": {"ephemeral_5m_input_tokens": five, "ephemeral_1h_input_tokens": hour}}
    usage["cache_creation_input_tokens"] = five + hour if created is None else created
    return json.dumps({"type": "assistant", "timestamp": stamp, "message": {"id": mid, "model": model, "usage": usage}, **kw}).encode()


T0 = "2026-09-25T20:00:00.000Z"


class ParseTest(unittest.TestCase):
    def test_hour_tier_anchored_before_the_reply(self):
        # the reply lands 30 s after the prompt (thinking); the clock started with the request
        s = cachettl.parse([user(T0), reply("2026-09-25T20:00:30.000Z", hour=500), reply("2026-09-25T20:00:31.000Z", hour=500)])
        self.assertEqual(s.ttl, 3600)
        self.assertEqual(s.expires, _parse_iso(T0) + 3600)

    def test_five_minute_tier(self):
        s = cachettl.parse([user(T0), reply("2026-09-25T20:00:05.000Z", five=500)])
        self.assertEqual(s.ttl, 300)

    def test_both_tiers_take_the_shorter(self):
        s = cachettl.parse([user(T0), reply("2026-09-25T20:00:05.000Z", five=10, hour=500)])
        self.assertEqual(s.ttl, 300)

    def test_pure_read_keeps_the_last_tier_and_moves_the_anchor(self):
        s = cachettl.parse([user(T0), reply("2026-09-25T20:00:05.000Z", hour=500),
                            user("2026-09-25T20:10:00.000Z"), reply("2026-09-25T20:10:04.000Z", mid="m2")])
        self.assertEqual(s.ttl, 3600)
        self.assertEqual(s.expires, _parse_iso("2026-09-25T20:10:00.000Z") + 3600)

    def test_write_without_split_is_five_minutes(self):
        line = json.dumps({"type": "assistant", "timestamp": "2026-09-25T20:00:05.000Z",
                           "message": {"id": "m1", "model": "claude-opus-5-5", "usage": {"cache_creation_input_tokens": 50}}}).encode()
        self.assertEqual(cachettl.parse([user(T0), line]).ttl, 300)

    def test_sidechain_and_synthetic_replies_ignored(self):
        s = cachettl.parse([user(T0), reply("2026-09-25T20:00:05.000Z", hour=500),
                            user("2026-09-25T20:30:00.000Z", isSidechain=True),
                            reply("2026-09-25T20:30:05.000Z", mid="sub", five=500, isSidechain=True),
                            reply("2026-09-25T20:31:00.000Z", mid="err", model="<synthetic>", five=1)])
        self.assertEqual((s.ttl, s.expires), (3600, _parse_iso(T0) + 3600))

    def test_nothing_before_the_first_reply(self):
        self.assertIsNone(cachettl.parse([user(T0)]))
        self.assertIsNone(cachettl.parse([b"not json", b""]))


class LeftTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.dir.name, "proj"))
        with open(os.path.join(self.dir.name, "proj", "sid.jsonl"), "wb") as f:
            f.write(b"\n".join([user(T0), reply("2026-09-25T20:00:05.000Z", hour=500)]) + b"\n")
        self.patch = mock.patch.object(cachettl, "CLAUDE_PROJECTS", self.dir.name)
        self.patch.start()
        cachettl._memo.clear()

    def tearDown(self):
        self.patch.stop()
        self.dir.cleanup()

    def test_counts_down_and_goes_cold(self):
        t0 = _parse_iso(T0)
        self.assertEqual(cachettl.seconds_left("sid", "claude-opus-5-5[1m]", t0 + 60), 3540)
        self.assertEqual(cachettl.label(cachettl.seconds_left("sid", "opus", t0 + 60)), "cache 59:00")
        self.assertEqual(cachettl.label(cachettl.seconds_left("sid", None, t0 + 4000)), "cache cold")

    def test_other_model_is_cold(self):
        self.assertEqual(cachettl.seconds_left("sid", "claude-sonnet-5", _parse_iso(T0)), 0.0)

    def test_unknown_session(self):
        self.assertIsNone(cachettl.seconds_left("nope", None, 0))
        self.assertEqual(cachettl.label(None), "")


if __name__ == "__main__":
    unittest.main()

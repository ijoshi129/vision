import datetime as dt
import tempfile
import unittest
from pathlib import Path

from vision.scheduled import CATCH_UP, Schedule, clean, next_run


def at(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm).timestamp()


class NextRunTest(unittest.TestCase):
    def test_daily_rolls_to_tomorrow(self):
        t = {"repeat": "daily", "time": "08:30"}
        self.assertEqual(next_run(t, at(2026, 9, 25, 9, 0)), at(2026, 9, 26, 8, 30))
        self.assertEqual(next_run(t, at(2026, 9, 25, 8, 0)), at(2026, 9, 25, 8, 30))

    def test_weekdays_skip_weekend(self):
        t = {"repeat": "weekdays", "time": "07:00"}
        # Friday 25 Sep 2026 after 7 → Monday 28th
        self.assertEqual(next_run(t, at(2026, 9, 25, 9, 0)), at(2026, 9, 28, 7, 0))

    def test_weekly(self):
        t = {"repeat": "weekly", "time": "18:00", "weekday": 6}
        self.assertEqual(next_run(t, at(2026, 9, 25, 9, 0)), at(2026, 9, 27, 18, 0))

    def test_once_in_the_past_is_none(self):
        t = {"repeat": "once", "time": "07:00", "date": "2026-09-25"}
        self.assertIsNone(next_run(t, at(2026, 9, 25, 9, 0)))

    def test_clean_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            clean({"prompt": "", "repeat": "daily", "time": "08:00"})
        with self.assertRaises(ValueError):
            clean({"prompt": "x", "repeat": "hourly", "time": "08:00"})
        with self.assertRaises(ValueError):
            clean({"prompt": "x", "repeat": "daily", "time": "25:00"})


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "scheduled.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_due_moves_on_and_persists(self):
        s = Schedule(self.path)
        task = s.upsert({"prompt": "weather", "repeat": "daily", "time": "08:00"})
        self.assertEqual(task["title"], "weather")
        when = task["next_run"]
        self.assertEqual([t["id"] for t in s.due(when + 1)], [task["id"]])
        self.assertEqual(s.due(when + 2), [])  # not twice
        again = Schedule(self.path).get(task["id"])
        self.assertGreater(again["next_run"], when)

    def test_missed_run_is_skipped(self):
        s = Schedule(self.path)
        task = s.upsert({"prompt": "x", "repeat": "daily", "time": "08:00"})
        self.assertEqual(s.due(task["next_run"] + CATCH_UP + 60), [])
        self.assertGreater(s.get(task["id"])["next_run"], task["next_run"])

    def test_once_switches_off_after_running(self):
        s = Schedule(self.path)
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
        task = s.upsert({"prompt": "x", "repeat": "once", "time": "08:00", "date": tomorrow})
        self.assertEqual(len(s.due(task["next_run"])), 1)
        self.assertFalse(s.get(task["id"])["enabled"])

    def test_edit_and_delete(self):
        s = Schedule(self.path)
        task = s.upsert({"prompt": "x", "repeat": "daily", "time": "08:00"})
        s.upsert({"id": task["id"], "enabled": False})
        self.assertFalse(s.get(task["id"])["enabled"])
        self.assertIsNone(s.get(task["id"])["next_run"])
        self.assertTrue(s.delete(task["id"]))
        self.assertEqual(s.list(), [])


if __name__ == "__main__":
    unittest.main()


class EndpointTest(unittest.TestCase):
    def test_library_thumb_and_scheduled_routes(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from fastapi.testclient import TestClient
        from PIL import Image

        from vision import library, server

        with tempfile.TemporaryDirectory() as d:
            uploads = Path(d) / "uploads"
            uploads.mkdir()
            Image.new("RGB", (800, 600), (200, 30, 30)).save(uploads / "20260925-101010-abcdef.jpg")
            (uploads / "20260925-101011-abcdef-notes.txt").write_text("hi")
            posted = []
            hub = SimpleNamespace(authorized=lambda t: t == "t", log=lambda _: None, post=posted.append,
                                  schedule=Schedule(Path(d) / "s.json"))
            client = TestClient(server.create_app(hub))
            auth = {"Authorization": "Bearer t"}
            with patch.object(server, "UPLOAD_DIR", uploads), patch.object(library, "THUMB_DIR", Path(d) / "thumbs"), \
                    patch.object(library, "_session_logs", lambda: []):
                items = client.get("/library", headers=auth).json()
                self.assertEqual({(i["kind"], i["name"]) for i in items},
                                 {("image", "20260925-101010-abcdef.jpg"), ("file", "notes.txt")})
                pic = next(i for i in items if i["kind"] == "image")["path"]
                r = client.get("/thumb", params={"path": pic, "size": 100}, headers=auth)
                self.assertEqual(r.status_code, 200)
                import io
                self.assertEqual(max(Image.open(io.BytesIO(r.content)).size), 100)
                self.assertEqual(client.get("/thumb", params={"path": "/etc/hostname"}, headers=auth).status_code, 404)

            r = client.post("/scheduled", json={"prompt": "news", "repeat": "daily", "time": "07:15"}, headers=auth)
            self.assertEqual(r.status_code, 200)
            tid = r.json()["id"]
            self.assertEqual(client.post("/scheduled", json={"prompt": "x", "repeat": "daily", "time": "7"},
                                         headers=auth).status_code, 400)
            self.assertEqual([t["id"] for t in client.get("/scheduled", headers=auth).json()], [tid])
            self.assertEqual(client.delete(f"/scheduled/{tid}", headers=auth).status_code, 200)
            self.assertEqual(client.get("/scheduled", headers=auth).json(), [])
            self.assertIn({"type": "scheduled"}, posted)

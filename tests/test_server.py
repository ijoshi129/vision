from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vision import server


class UploadTests(unittest.TestCase):
    def test_uploaded_image_reaches_the_message_as_a_readable_file(self):
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi.testclient import TestClient

        hub = SimpleNamespace(authorized=lambda token: token == "upload-test", log=lambda _: None)
        client = TestClient(server.create_app(hub))
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "UPLOAD_DIR", Path(directory)):
            payload = b"test image bytes"
            response = client.post("/upload", files={"file": ("photo.jpg", payload, "image/jpeg")},
                                   headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(response.status_code, 200)
            path = response.json()["path"]
            self.assertEqual(Path(path).read_bytes(), payload)
            self.assertIn(f"[Attached image: {path}]", server.with_attachments("Look at this", [path]))
            self.assertIn("Look at the attached image", server.with_attachments("", [path]))

    @unittest.skipIf(sys.platform == "win32", "`vision serve` is not ported to Windows yet")
    def test_reply_image_is_served_from_home_or_tmp_only(self):
        from types import SimpleNamespace
        from fastapi.testclient import TestClient

        hub = SimpleNamespace(authorized=lambda token: token == "image-test", log=lambda _: None)
        client = TestClient(server.create_app(hub))
        auth = {"Authorization": "Bearer image-test"}
        with tempfile.NamedTemporaryFile(suffix=".png", dir="/tmp") as f:
            f.write(b"png bytes")
            f.flush()
            ok = client.get("/image", params={"path": f.name}, headers=auth)
            self.assertEqual((ok.status_code, ok.content, ok.headers["content-type"]), (200, b"png bytes", "image/png"))
            self.assertEqual(client.get("/image", params={"path": f.name}).status_code, 401)
        with tempfile.NamedTemporaryFile(suffix=".txt", dir="/tmp") as f:
            self.assertEqual(client.get("/image", params={"path": f.name}, headers=auth).status_code, 404)
        self.assertEqual(client.get("/image", params={"path": "/tmp/../etc/passwd"}, headers=auth).status_code, 404)
        self.assertFalse(server.image_path_allowed(__import__("pathlib").Path("/usr/share/pixmaps/x.png")))

    def test_uploaded_file_keeps_its_name_and_reaches_the_brain(self):
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi.testclient import TestClient

        hub = SimpleNamespace(authorized=lambda token: token == "upload-test", log=lambda _: None)
        client = TestClient(server.create_app(hub))
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "UPLOAD_DIR", Path(directory)):
            payload = b"%PDF-1.4 test"
            response = client.post("/upload", files={"file": ("Q3 report (final).pdf", payload, "application/pdf")},
                                   headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(response.status_code, 200, response.text)
            path = Path(response.json()["path"])
            self.assertEqual(path.parent, Path(directory))
            self.assertTrue(path.name.endswith("-Q3 report _final_.pdf"), path.name)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(server.with_attachments("", [str(path)]), f"[Attached file: {path}]\n\nRead the attached file(s).")
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"jpg")
            note = server.with_attachments("", [str(image), str(path)])
            self.assertTrue(note.endswith("Look at the attached image(s).\nRead the attached file(s)."), note)
            escape = client.post("/upload", files={"file": ("../../evil.sh", b"echo", "text/plain")},
                                 headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(Path(escape.json()["path"]).parent, Path(directory))

    def test_failed_upload_returns_an_error_instead_of_a_path(self):
        from types import SimpleNamespace
        from fastapi.testclient import TestClient

        hub = SimpleNamespace(authorized=lambda token: token == "upload-test", log=lambda _: None)
        client = TestClient(server.create_app(hub))
        self.assertEqual(client.post("/upload", files={"file": ("photo.jpg", b"x")}).status_code, 401)
        response = client.post("/upload", files={"file": ("photo.jpg", b"", "image/jpeg")},
                               headers={"Authorization": "Bearer upload-test"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("path", response.json())


    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_uploaded_video_reaches_the_brain_as_frames_and_words(self):
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi.testclient import TestClient

        heard = []
        hub = SimpleNamespace(authorized=lambda token: token == "upload-test", log=lambda _: None,
                              transcribe_bytes=lambda wav: heard.append(wav) or "hello from the clip")
        client = TestClient(server.create_app(hub))
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "UPLOAD_DIR", Path(directory)):
            clip = Path(directory) / "source.mp4"
            # Portrait, 12 s, a hard cut from red to blue at 6 s, with sound.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=360x640:d=6:r=10",
                            "-f", "lavfi", "-i", "color=blue:s=360x640:d=6:r=10", "-f", "lavfi", "-i", "sine=d=12",
                            "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2",
                            "-c:a", "aac", "-shortest", str(clip)], check=True)
            response = client.post("/upload", files={"file": ("video.mp4", clip.read_bytes(), "video/mp4")},
                                   headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(response.status_code, 200, response.text)
            path = response.json()["path"]
            note = server.with_attachments("", [path])
            self.assertIn(f"[Attached video: {path}]", note)
            frames = [line.split(": ", 1)[1].rstrip("]") for line in note.splitlines() if line.startswith("[Frame ")]
            self.assertGreaterEqual(len(frames), 2)
            self.assertLessEqual(len(frames), server.MAX_VIDEO_FRAMES)
            self.assertTrue(all(os.path.isfile(f) for f in frames))
            self.assertIn("[Frame 0:06:", note)  # the cut is a frame of its own
            self.assertIn("[Video audio, transcribed: hello from the clip]", note)
            self.assertTrue(heard and heard[0][:4] == b"RIFF")
            self.assertTrue(note.endswith("Watch the attached video(s) through their frames."))
            thumb = client.get(f"/uploads/{Path(path).name}", headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(thumb.headers["content-type"], "image/jpeg")
            auth = {"Authorization": "Bearer upload-test"}
            whole = client.get(f"/videos/{Path(path).name}", headers=auth)
            self.assertEqual(whole.content, Path(path).read_bytes())
            part = client.get(f"/videos/{Path(path).name}", headers={**auth, "Range": "bytes=0-99"})
            self.assertEqual((part.status_code, len(part.content)), (206, 100))
            self.assertEqual(client.get(f"/videos/{Path(path).name}").status_code, 401)
            self.assertEqual(client.get(f"/videos/{Path(path).with_suffix('.json').name}", headers=auth).status_code, 404)

            broken = client.post("/upload", files={"file": ("video.mp4", b"not a video", "video/mp4")},
                                 headers={"Authorization": "Bearer upload-test"})
            self.assertEqual(broken.status_code, 422)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir() if p.name != "source.mp4"),
                             sorted([Path(path).name, Path(path).with_suffix(".json").name, Path(path).stem + "-frames"]))

class TokenTests(unittest.TestCase):
    def test_token_is_created_once_and_private(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "remote_token")
            with patch.object(server, "TOKEN_FILE", server.TOKEN_FILE.__class__(path)), patch.object(server, "CONFIG_DIR", server.CONFIG_DIR.__class__(d)):
                first = server.load_token()
                self.assertGreaterEqual(len(first), 30)
                if os.name == "posix":  # Windows has no mode bits; the file inherits the folder's ACL
                    self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
                self.assertEqual(server.load_token(), first)
                self.assertNotEqual(server.load_token(regenerate=True), first)


class PairingTests(unittest.TestCase):
    def test_payload_round_trips(self):
        payload = server.pairing_payload("https://laptop.tail.ts.net", "abc")
        self.assertEqual(json.loads(payload), {"url": "https://laptop.tail.ts.net", "token": "abc"})

    def test_qr_is_square_ish_and_uses_half_blocks(self):
        lines = server.qr_lines(server.pairing_payload("https://example.ts.net", "x" * 32))
        self.assertTrue(lines)
        self.assertEqual(len({len(l) for l in lines}), 1)
        self.assertTrue(set("".join(lines)) <= set("█▀▄ "))


class HistoryTests(unittest.TestCase):
    def test_reads_user_and_assistant_text_and_merges_streamed_assistant_entries(self):
        from vision import sessions

        sid = "11111111-2222-3333-4444-555555555555"
        with tempfile.TemporaryDirectory() as d:
            proj = os.path.join(d, "-home-x")
            os.makedirs(proj)
            lines = [
                {"type": "user", "message": {"role": "user", "content": "hello"}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "name": "Bash"}]}},
                {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "..."}]}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
                {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<local-command-stdout>x</local-command-stdout>"}},
                {"type": "user", "message": {"role": "user", "content": "You are taking over an ongoing conversation from another model. Its handoff note:\n\nnote\n\n---\n\nContinue naturally from here; do not mention the handoff unless asked. The user's next message:\n\nwhat next?"}},
            ]
            with open(os.path.join(proj, f"{sid}.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(l) for l in lines) + "\n")
            with patch.object(sessions, "CLAUDE_PROJECTS", d):
                hist = server.claude_history(sid)
        self.assertEqual(hist, [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi\n\ndone"},
            {"role": "user", "text": "what next?"},
        ])

    def test_unknown_session_is_empty(self):
        self.assertEqual(server.claude_history("nope"), [])


class WavTests(unittest.TestCase):
    def test_wav_bytes_is_riff_pcm16_mono(self):
        import numpy as np

        data = server.wav_bytes(np.zeros(2400, dtype=np.float32))
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertIn(b"WAVE", data[:16])
        self.assertEqual(len(data), 44 + 2400 * 2)


if __name__ == "__main__":
    unittest.main()


class _Turn:
    def __init__(self, text, sid):
        self.text, self.session_id, self.model, self.error, self.is_error = text, sid, "stub", "", False


class _StubBrain:
    """Answers "echo: <text>" after a short pause, streaming one delta; independent per instance."""

    provider = "claude"
    workdir = "/tmp"

    def __init__(self, cfg, delay=0.2):
        self.cfg, self.delay = cfg, delay
        self.session_id = None
        self.calls = 0

    def resolved_model(self):
        return self.cfg.model

    def cancel(self):
        pass

    def ask(self, text, on_text=None, on_status=None, on_question=None, on_agent=None, on_tool=None):
        import time as _t

        self.calls += 1
        self.session_id = self.session_id or f"sess-{id(self) % 1000}"
        _t.sleep(self.delay)
        reply = f"echo: {text}"
        if on_text:
            on_text(reply)
        return _Turn(reply, self.session_id)


class MultiChatTests(unittest.TestCase):
    """Several chats run at the same time and every frame names its chat."""

    def _hub(self):
        from vision.config import Config

        cfg = Config()
        cfg.conversation.model = "haiku"
        brain = _StubBrain(cfg.brain)
        with patch.object(server, "load_token", return_value="tok"):
            hub = server.Hub(cfg, brain, "tok", log=lambda *_: None)
        return hub

    def _drain(self, ws, until, limit=60):
        seen = []
        for _ in range(limit):
            ev = ws.receive_json()
            seen.append(ev)
            if until(ev, seen):
                return seen
        raise AssertionError(f"never saw the frame; got {[e['type'] for e in seen]}")

    def test_two_chats_run_concurrently_and_frames_carry_chat_ids(self):
        from fastapi.testclient import TestClient

        hub = self._hub()
        with tempfile.TemporaryDirectory() as live, patch("vision.link.LIVE_DIR", __import__("pathlib").Path(live)), \
             patch.object(server.Hub, "warm_up", lambda self: None), \
             patch("vision.sessions.session_history", lambda provider, sid, limit=0, **kw: []), \
             patch("vision.brain.create_brain", side_effect=lambda cfg, **kw: _StubBrain(cfg, delay=0.3)), \
             patch("vision.cli._turn_brain", lambda brain, conv, voice, text="", **kw: brain):
            app = server.create_app(hub)
            with TestClient(app) as tc, tc.websocket_connect("/ws?token=tok") as ws:
                hello = ws.receive_json()
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(len(hello["chats"]), 1)
                first = hello["chats"][0]["chat"]

                ws.send_json({"type": "new"})
                opened = self._drain(ws, lambda ev, _: ev["type"] == "chat" and ev.get("opened"))[-1]
                second = opened["chat"]
                self.assertNotEqual(first, second)

                ws.send_json({"type": "message", "chat": first, "text": "one"})
                ws.send_json({"type": "message", "chat": second, "text": "two"})
                seen = self._drain(ws, lambda ev, all_: sum(e["type"] == "done" for e in all_) == 2)
                starts = [e for e in seen if e["type"] == "start"]
                dones = {e["chat"]: e for e in seen if e["type"] == "done"}
                self.assertEqual({e["chat"] for e in starts}, {first, second})
                # both started before either finished: they ran in parallel, not queued
                first_done = next(i for i, e in enumerate(seen) if e["type"] == "done")
                self.assertTrue(all(i < first_done for i, e in enumerate(seen) if e["type"] == "start"))
                self.assertEqual(dones[first]["text"], "echo: one")
                self.assertEqual(dones[second]["text"], "echo: two")
                self.assertEqual(hub.chats[first].title, "one")
                self.assertEqual(hub.chats[second].title, "two")

                # closing one chat leaves the other untouched
                ws.send_json({"type": "close", "chat": second})
                self._drain(ws, lambda ev, _: ev["type"] == "chat_closed" and ev["chat"] == second)
                self.assertEqual(set(hub.chats), {first})

                r = tc.get("/chats", headers={"Authorization": "Bearer tok"})
                self.assertEqual([c["chat"] for c in r.json()], [first])
                r = tc.get("/history", params={"chat": first}, headers={"Authorization": "Bearer tok"})
                self.assertEqual(r.status_code, 200)

                # closing the last chat leaves none open: no replacement is forced on the phone
                ws.send_json({"type": "close", "chat": first})
                self._drain(ws, lambda ev, _: ev["type"] == "chat_closed" and ev["chat"] == first)
                self.assertEqual(hub.chats, {})
                r = tc.get("/chats", headers={"Authorization": "Bearer tok"})
                self.assertEqual(r.json(), [])
                ws.send_json({"type": "new"})
                self._drain(ws, lambda ev, _: ev["type"] == "chat" and ev.get("opened"))
                self.assertEqual(len(hub.chats), 1)


class FollowDefaultsTests(unittest.TestCase):
    """A new chat picks up /default saved while `vision serve` runs, unless flags pinned the model."""

    def _new_chat_model(self, config_text, follow=True):
        from pathlib import Path
        from vision import config
        from vision.config import Config

        cfg = Config()
        cfg.brain.model, cfg.brain.effort = "opus", "high"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text(config_text)
            with patch.object(config, "CONFIG_DIR", Path(d)), patch.object(config, "CONFIG_PATH", path), \
                 patch("vision.brain.create_brain", side_effect=lambda c, **kw: _StubBrain(c)):
                hub = server.Hub(cfg, _StubBrain(cfg.brain), "tok", log=lambda *_: None, follow_defaults=follow)
                chat = hub.open_chat()
        return chat.brain.cfg.model, chat.brain.cfg.effort

    def test_new_chat_uses_the_default_saved_after_start_up(self):
        self.assertEqual(self._new_chat_model('[brain]\nmodel = "sonnet"\neffort = "low"\n'), ("sonnet", "low"))

    def test_broken_config_falls_back_to_the_start_up_defaults(self):
        self.assertEqual(self._new_chat_model('[brain\nmodel = "sonnet"\n'), ("opus", "high"))

    def test_model_flag_pins_new_chats(self):
        self.assertEqual(self._new_chat_model('[brain]\nmodel = "sonnet"\neffort = "low"\n', follow=False), ("opus", "high"))


class DefaultsEndpointTests(unittest.TestCase):
    """The phone reads and saves the default model for new chats, the same config.toml lines /default writes."""

    def test_saves_the_default_and_new_chats_follow_it(self):
        from pathlib import Path
        from fastapi.testclient import TestClient
        from vision import config
        from vision.config import Config

        cfg = Config()
        cfg.brain.model, cfg.brain.effort = "opus", "high"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text('# mine\n[brain]\nmodel = "opus"\neffort = "high"\n')
            with patch.object(config, "CONFIG_DIR", Path(d)), patch.object(config, "CONFIG_PATH", path), \
                 patch("vision.brain.create_brain", side_effect=lambda c, **kw: _StubBrain(c)):
                hub = server.Hub(cfg, _StubBrain(cfg.brain), "tok", log=lambda *_: None, follow_defaults=False)
                tc = TestClient(server.create_app(hub))
                auth = {"Authorization": "Bearer tok"}
                self.assertEqual(tc.get("/defaults", headers=auth).json()["current"], "opus")
                saved = tc.post("/defaults", json={"model": "sonnet", "effort": "low"}, headers=auth).json()
                self.assertEqual((saved["current"], saved["effort"]), ("sonnet", "low"))
                self.assertIn('model = "sonnet"', path.read_text())
                self.assertIn("# mine", path.read_text())
                chat = hub.open_chat()
                self.assertEqual((chat.brain.cfg.model, chat.brain.cfg.effort), ("sonnet", "low"))
                self.assertEqual(tc.post("/defaults", json={"model": "nope"}, headers=auth).status_code, 400)
                self.assertEqual(tc.get("/defaults").status_code, 401)

from __future__ import annotations

import asyncio
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from vision.brain import Turn
from vision.cli import _next_input, _talk_turns, _turn_brain
from vision.config import Config
from vision.server import Hub


def make_brain(cfg):
    brain = SimpleNamespace(provider="claude", cfg=cfg.brain, workdir="/tmp", session_id=None,
                            voice_mode=False, cancel=Mock(), new_session=Mock(), resume=Mock(), resolved_model=lambda: cfg.brain.model)
    def answer(text, **callbacks):
        if callbacks.get("on_text"):
            callbacks["on_text"]("Typed reply")
        return Turn(text="Typed reply")
    brain.ask = Mock(side_effect=answer)
    return brain


class TerminalRoutingTests(unittest.TestCase):
    def test_keyboard_during_push_to_talk_keeps_text_origin(self):
        cfg = Config()
        kb = SimpleNamespace(q=queue.Queue())
        kb.q.put("  Fix it  ")
        with patch("vision.config.resolve_device", return_value=None), patch("vision.cli.console"):
            self.assertEqual(_next_input(cfg, Mock(), Mock(), kb, ptt=True), ("Fix it", False))

    def test_push_to_talk_transcription_keeps_voice_origin(self):
        cfg = Config()
        kb = SimpleNamespace(q=queue.Queue())
        kb.q.put("")
        kb.q.put("")
        mic = Mock()
        mic.record_until_enter.return_value = SimpleNamespace(size=10)
        stt = Mock()
        stt.transcribe.return_value = "Hello"
        with patch("vision.config.resolve_device", return_value=None), patch("vision.cli.console"):
            self.assertEqual(_next_input(cfg, mic, stt, kb, ptt=True), ("Hello", True))

    def test_typing_while_hands_free_capture_runs_stays_typed(self):
        cfg = Config()
        cfg.listen.chime = False
        kb = Mock()
        kb.poll.return_value = "typed while listening"
        mic = Mock()
        def record(**kw):
            kw["cancel"].wait(1)
            return SimpleNamespace(size=0)
        mic.record_utterance.side_effect = record
        with patch("vision.config.resolve_device", return_value=None), patch("vision.cli.console"):
            self.assertEqual(_next_input(cfg, mic, Mock(), kb, ptt=False), ("typed while listening", False))

    def test_talk_loop_sends_both_sources_to_the_conversation_and_updates_worker_after_switch(self):
        cfg = Config()
        brain = make_brain(cfg)
        replacement = make_brain(cfg)
        voice = Mock()
        with patch("vision.conversation.VoiceConversation", return_value=voice), \
             patch("vision.cli._next_input", side_effect=[("Hi", True), ("Fix it", False), ("/model sonnet", False), ("Did it work?", True), None]), \
             patch("vision.cli._switch_spoken", return_value=(replacement, "")), \
             patch("vision.cli._run_turn") as run, patch("vision.cli.console"), patch("vision.cli.show_user"):
            result = _talk_turns(cfg, brain, Mock(), Mock(), Mock(), Mock(), ptt=False)
        self.assertEqual([c.args[0] for c in run.call_args_list], [voice, voice, voice])  # one model for the chat
        self.assertEqual([c.kwargs["markdown"] for c in run.call_args_list], [False, True, False])
        self.assertIs(voice.agent, replacement)
        self.assertIs(result, replacement)
        self.assertFalse(brain.voice_mode)

    def test_barge_in_transcription_returns_to_conversation(self):
        cfg = Config()
        cfg.listen.barge_in = "speech"
        brain = make_brain(cfg)
        voice, barge = Mock(), Mock()
        barge.error = None
        barge.cut.is_set.return_value = False
        barge.stop.side_effect = [("audio", ""), None]
        stt = Mock()
        stt.transcribe.return_value = "Wait, what about the other one?"
        with patch("vision.conversation.VoiceConversation", return_value=voice), \
             patch("vision.cli._next_input", side_effect=[("typed", False), None]), \
             patch("vision.wake.BargeIn") as barge_type, patch("vision.cli._run_turn") as run, \
             patch("vision.cli.console"), patch("vision.cli.show_user"):
            barge_type.return_value.start.return_value = barge
            _talk_turns(cfg, brain, Mock(), Mock(), stt, Mock(), ptt=False, listener=Mock())
        self.assertEqual([c.args[0] for c in run.call_args_list], [voice, voice])


class RemoteRoutingTests(unittest.IsolatedAsyncioTestCase):
    """The phone's side: each chat on the hub routes its own turns, like the terminal's talk loop."""

    def setUp(self):
        self.cfg = Config()
        self.brain = make_brain(self.cfg)
        self.hub = Hub(self.cfg, self.brain, "token", log=lambda _: None)
        self.events = []
        self.hub.post = self.events.append
        (self.chat,) = self.hub.chats.values()
        self.voice = Mock()
        def answer(text, **callbacks):
            callbacks["on_text"]("Spoken reply")
            return Turn(text="Spoken reply", model="sonnet")
        self.voice.ask.side_effect = answer
        self.chat.conversation = self.voice
        # A fresh chat has no saved session to seed its transcript from.
        history = patch("vision.sessions.session_history", return_value=[])
        history.start()
        self.addCleanup(history.stop)

    def socket(self):
        ws = Mock()
        ws.send_json = AsyncMock()
        return ws

    async def test_playback_does_not_send_typed_input_to_conversation(self):
        wire = Mock()
        with patch.object(self.hub, "speaker", return_value=Mock()), patch("vision.server.WireSpeaker", return_value=wire):
            await self.chat.run_turn("typed", speak=True, voice=False)
        self.brain.ask.assert_called_once()
        self.voice.ask.assert_not_called()
        wire.feed.assert_called_once_with("Typed reply")
        self.assertFalse(self.brain.voice_mode)

    async def test_voice_routes_to_conversation_even_without_playback(self):
        await self.chat.run_turn("spoken", speak=False, voice=True)
        self.voice.ask.assert_called_once()
        self.brain.ask.assert_not_called()
        self.assertEqual(self.chat.history(), [{"role": "user", "text": "spoken"}, {"role": "assistant", "text": "Spoken reply"}])
        self.assertIsNotNone(self.hub.hello()["chats"][0]["history_id"])
        self.assertFalse(self.chat.busy)

    async def test_voice_then_text_preserves_mixed_history_on_reconnect(self):
        await self.chat.run_turn("spoken", speak=False, voice=True)
        await self.chat.run_turn("typed", speak=False, voice=False)
        self.assertEqual([row["text"] for row in self.chat.history()], ["spoken", "Spoken reply", "typed", "Typed reply"])
        self.assertIs(self.voice.agent, self.brain)

    async def test_new_and_resume_open_their_own_chats_and_leave_this_one_alone(self):
        from vision.sessions import SessionInfo

        ws = self.socket()
        old = SessionInfo(id="old-typed", provider=self.brain.provider, title="earlier", last_active=0.0)
        with patch("vision.brain.create_brain", side_effect=lambda *a, **k: make_brain(self.cfg)), \
             patch("vision.sessions.find_any_session", return_value=old), \
             patch("vision.cli._apply_session", return_value="resumed old-typed"):
            await self.hub.handle(ws, {"type": "new"})
            await self.hub.handle(ws, {"type": "resume", "session_id": "old-typed"})
        opened = [c.args[0] for c in ws.send_json.call_args_list]
        self.assertEqual([frame.get("opened") for frame in opened], [True, True])
        self.assertEqual(len(self.hub.chats), 3)
        ids = {frame["chat"] for frame in opened} | {self.chat.id}
        self.assertEqual(len(ids), 3)
        self.assertEqual(len({c.history_id for c in self.hub.chats.values()}), 3)
        self.assertEqual(self.hub.chats[opened[1]["chat"]].title, "earlier")
        self.voice.new_session.assert_not_called()

    async def test_resume_of_an_unknown_id_changes_nothing(self):
        ws = self.socket()
        history_id = self.chat.history_id
        with patch("vision.sessions.find_any_session", return_value=None):
            await self.hub.handle(ws, {"type": "resume", "session_id": "nope"})
        self.assertEqual(list(self.hub.chats), [self.chat.id])
        self.assertEqual(self.chat.history_id, history_id)
        self.voice.new_session.assert_not_called()
        self.assertIn("no conversation starts with", ws.send_json.call_args.args[0]["text"])

    async def test_cancel_reaches_both_models_and_audio(self):
        wire = self.chat._wire = Mock()
        self.hub.cancel()
        self.voice.cancel.assert_called_once()
        self.brain.cancel.assert_called_once()
        wire.stop.assert_called_once()

    async def test_messages_received_during_a_turn_run_in_order(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def run_one(text, speak, voice, talk=False):
            calls.append(text)
            if text == "first":
                started.set()
                await release.wait()
            self.events.append({"type": "done", "text": f"reply to {text}", "busy": bool(self.chat._pending)})

        self.chat._run_one_turn = run_one
        ws = self.socket()
        await self.hub.handle(ws, {"type": "message", "text": "first", "chat": self.chat.id})
        await asyncio.wait_for(started.wait(), 1)

        await self.hub.handle(ws, {"type": "message", "text": "second", "chat": self.chat.id})
        self.assertEqual(calls, ["first"])
        self.assertEqual(len(self.chat._pending), 1)
        task = self.chat._turn_task
        self.assertIsNotNone(task)
        release.set()
        await asyncio.wait_for(task, 3)

        self.assertFalse(self.chat.busy)
        self.assertEqual(calls, ["first", "second"])
        done = [event for event in self.events if event["type"] == "done"]
        self.assertEqual([event["busy"] for event in done], [True, False])


if __name__ == "__main__":
    unittest.main()

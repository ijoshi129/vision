"""The spoken filler ("One sec.") that covers a late first sentence."""
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_streaming_speaker import FakeSpeaker
from vision.timing import VoiceTiming
from vision.tts import SAMPLE_RATE, Speaker, StreamingSpeaker


def clip(seconds: float = 0.3) -> np.ndarray:
    return np.full(int(SAMPLE_RATE * seconds), 0.2, dtype=np.float32)


class FillerSpeaker(FakeSpeaker):
    def __init__(self):
        super().__init__()
        self.last_filler = ""
        self.pieces: list[float] = []  # seconds per piece handed to the device, in order

    def play(self, audio, stream=None):
        self.pieces.append(len(audio) / SAMPLE_RATE)
        super().play(audio, stream)


CLIPS = [("One sec.", clip()), ("Let me see.", clip())]
LATER = [("Still on it.", clip(0.2))]


class FillerTimingTests(unittest.TestCase):
    def test_late_first_sentence_gets_a_filler_first(self):
        sp = FillerSpeaker()
        trace = VoiceTiming()
        ss = StreamingSpeaker(sp, timing=trace)
        ss.arm_filler(CLIPS, 0.1)
        time.sleep(0.35)
        ss.feed("Here is the answer. ")
        ss.finish()
        self.assertEqual(ss._chunks[0][0], 0, "the filler is a chunk of no source text")
        self.assertGreater(ss._chunks[0][1], 0.2)
        self.assertEqual(ss._chunks[1][0], len("Here is the answer. "))
        self.assertIn(sp.last_filler, ("One sec.", "Let me see."))
        stages = [e["stage"] for e in trace.events]
        self.assertLess(stages.index("filler"), stages.index("speech_queued"))
        self.assertGreaterEqual(sp.played, 0.3 + len("Here is the answer.") / sp.CPS - 0.05)

    def test_prompt_reply_cancels_the_filler(self):
        sp = FillerSpeaker()
        ss = StreamingSpeaker(sp)
        ss.arm_filler(CLIPS, 0.3)
        ss.feed("Quick answer. ")
        time.sleep(0.5)
        ss.finish()
        self.assertEqual([c[0] for c in ss._chunks], [len("Quick answer. ")])
        self.assertEqual(sp.last_filler, "")

    def test_silent_turn_does_not_fire_after_finish(self):
        sp = FillerSpeaker()
        ss = StreamingSpeaker(sp)
        ss.arm_filler(CLIPS, 0.2)
        ss.finish()
        time.sleep(0.35)
        self.assertEqual(ss._chunks, [])
        self.assertEqual(sp.played, 0.0)

    def test_stop_cancels_the_filler(self):
        sp = FillerSpeaker()
        ss = StreamingSpeaker(sp)
        ss.arm_filler(CLIPS, 0.2)
        ss.stop()
        time.sleep(0.35)
        ss.finish()
        self.assertEqual(ss._chunks, [])

    def test_later_filler_once_when_the_reply_is_still_coming(self):
        sp = FillerSpeaker()
        ss = StreamingSpeaker(sp)
        ss.arm_filler(CLIPS, 0.05, later=LATER, again_s=0.2)
        time.sleep(0.8)
        ss.feed("Done. ")
        ss.finish()
        self.assertEqual([c[0] for c in ss._chunks], [0, 0, len("Done. ")])
        self.assertEqual(sp.last_filler, "Still on it.")

    def test_never_the_same_phrase_twice_running(self):
        sp = FillerSpeaker()
        seen = []
        for _ in range(6):
            ss = StreamingSpeaker(sp)
            ss.arm_filler(CLIPS, 0.01)
            time.sleep(0.1)
            seen.append(sp.last_filler)
            ss.feed("Right. ")
            ss.finish()
        self.assertTrue(all(a != b for a, b in zip(seen, seen[1:])), seen)

    def test_reveal_waits_on_the_filler(self):
        sp = FillerSpeaker()
        ss = StreamingSpeaker(sp)
        ss.arm_filler([("One sec.", clip(0.6))], 0.01)
        time.sleep(0.15)
        ss.feed("Answer. ")
        time.sleep(0.15)
        self.assertEqual(ss.spoken, 0, "no reply text is revealed while the filler plays")
        ss.finish()


class FillerCacheTests(unittest.TestCase):
    def make_speaker(self, synth):
        sp = Speaker.__new__(Speaker)
        sp.cfg = types.SimpleNamespace(engine="qwen3", voice="jarvis", language="English", clone_mode="embedding", rate=1.0)
        sp.engine = types.SimpleNamespace(voice="jarvis")
        sp._filler_cache = {}
        sp.last_filler = ""
        sp.synth = synth
        sp._voice_rate = lambda: 1.0
        return sp

    def test_clips_are_made_once_and_read_back_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Speaker, "FILLER_DIR", Path(tmp)):
            synth = mock.Mock(return_value=clip())
            sp = self.make_speaker(synth)
            self.assertEqual(sp.fillers(["One sec."]), [], "nothing is synthesised on the turn")
            made = sp.prepare_fillers(["One sec.", "Let me see."])
            self.assertEqual([p for p, _ in made], ["One sec.", "Let me see."])
            self.assertEqual(synth.call_count, 2)
            sp.prepare_fillers(["One sec."])
            self.assertEqual(synth.call_count, 2, "a clip in memory is not made again")
            self.assertEqual(len(list(Path(tmp).glob("*.npy"))), 2)
            fresh = self.make_speaker(mock.Mock(side_effect=AssertionError("must read the cache")))
            ready = fresh.prepare_fillers(["One sec."])
            self.assertEqual(len(ready), 1)
            np.testing.assert_array_equal(ready[0][1], clip())
            self.assertEqual([p for p, _ in fresh.fillers(["One sec.", "Let me see."])], ["One sec."])

    def test_a_new_voice_gets_its_own_clips(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Speaker, "FILLER_DIR", Path(tmp)):
            synth = mock.Mock(return_value=clip())
            sp = self.make_speaker(synth)
            sp.prepare_fillers(["One sec."])
            sp.engine.voice = "narrator"
            self.assertEqual(sp.fillers(["One sec."]), [])
            sp.prepare_fillers(["One sec."])
            self.assertEqual(synth.call_count, 2)

    def test_phoenix_fillers_stay_in_the_configured_style(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Speaker, "FILLER_DIR", Path(tmp)):
            synth = mock.Mock(return_value=clip())
            sp = self.make_speaker(synth)
            sp.cfg.voice = sp.engine.voice = "phoenix-conversational"
            sp.set_voice = lambda spec: setattr(sp.engine, "voice", spec)
            sp.prepare_fillers(["One sec."])
            sp.engine.voice = "phoenix-expressive"  # where an excited reply left the engine
            self.assertEqual(len(sp.fillers(["One sec."])), 1, "the clip is filed under the configured style")
            sp.prepare_fillers(["Let me see."])
            self.assertEqual(sp.engine.voice, "phoenix-conversational", "a new clip is made in that style")

    def test_a_failed_phrase_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Speaker, "FILLER_DIR", Path(tmp)):
            sp = self.make_speaker(mock.Mock(side_effect=[RuntimeError("no gpu"), clip()]))
            made = sp.prepare_fillers(["One sec.", "Let me see."])
            self.assertEqual([p for p, _ in made], ["Let me see."])


if __name__ == "__main__":
    unittest.main()

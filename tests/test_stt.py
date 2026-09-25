"""Live speech-preview safeguards."""
import threading
import time
import unittest

import numpy as np

from vision.config import ListenConfig
from vision.stt import LiveTranscript, Transcriber


class LiveTranscriptTests(unittest.TestCase):
    def test_live_preview_is_only_enabled_for_cuda(self):
        stt = Transcriber(ListenConfig())
        stt.device = "cpu/small.en"
        self.assertFalse(stt.can_preview_live)
        stt.device = "cuda/large-v3-turbo"
        self.assertTrue(stt.can_preview_live)

    def test_stop_times_out_instead_of_freezing_on_stuck_preview(self):
        entered = threading.Event()
        release = threading.Event()

        class StuckTranscriber:
            def transcribe(self, _audio):
                entered.set()
                release.wait()
                return "done"

        live = LiveTranscript(StuckTranscriber(), lambda _text: None)
        live.feed(np.ones(16000, dtype=np.float32))
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        self.assertFalse(live.stop(timeout=0.05))
        self.assertLess(time.monotonic() - started, 0.5)
        release.set()
        live._thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()

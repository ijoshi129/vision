import sys
import unittest

import numpy as np

from vision import echo


class _Frame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        assert (sample_rate, num_channels, samples_per_channel) == (echo.RATE, 1, echo.FRAME)
        self.data = bytearray(data)


class _Apm:
    """Stands in for livekit's module: records the far end and zeroes whatever it is asked to clean."""

    def __init__(self):
        self.far_frames = 0
        self.delays = []

    def process_reverse_stream(self, frame):
        self.far_frames += 1

    def set_stream_delay_ms(self, ms):
        self.delays.append(ms)

    def process_stream(self, frame):
        frame.data[:] = bytes(len(frame.data))


class EchoCancellerTests(unittest.TestCase):
    def test_mode_is_on_for_windows_only_unless_set(self):
        self.assertEqual(echo.wanted("auto"), sys.platform == "win32")
        self.assertTrue(echo.wanted("on"))
        self.assertTrue(echo.wanted(" On "))
        self.assertFalse(echo.wanted("off"))

    def test_mic_frames_come_back_the_same_length_ten_ms_late(self):
        apm = _Apm()
        ec = echo.EchoCanceller(apm, frame_type=_Frame)
        mic = (np.arange(512 * 3) % 1000 + 1).astype(np.int16)  # Silero-sized frames: not a multiple of 10 ms
        out = b"".join(ec.cleaned(mic[i:i + 512].tobytes()) for i in range(0, len(mic), 512))
        self.assertEqual(len(out), mic.nbytes)
        got = np.frombuffer(out, dtype=np.int16)
        self.assertTrue((got == 0).all())  # the first 10 ms is the lead-in, the rest what the fake zeroed
        self.assertEqual(len(apm.delays), len(mic) // echo.FRAME)

    def test_played_audio_is_resampled_and_framed(self):
        apm = _Apm()
        ec = echo.EchoCanceller(apm, frame_type=_Frame)
        ec.output_latency_s, ec.input_latency_s = 0.2, 0.03
        for _ in range(10):
            ec.played(np.zeros(2400, np.float32), 24000)  # ten of the speaker's 100 ms blocks
        self.assertEqual(apm.far_frames, 100)  # one second of 10 ms frames at 16 kHz
        ec.cleaned(bytes(2 * echo.FRAME))
        self.assertEqual(apm.delays[-1], 230)


if __name__ == "__main__":
    unittest.main()

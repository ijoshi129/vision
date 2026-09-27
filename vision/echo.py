"""Take Vision's own voice back out of the microphone: WebRTC's echo canceller (with its noise
suppressor and high-pass filter), through livekit's audio processing module.

Speaker.play() hands every block it plays to played(), the far end; Microphone passes what it
captures through cleaned() before the VAD, the wake word or Whisper see it. The canceller learns the
path from the speakers to the microphone, delay included, and subtracts what it predicts.

On Linux, PipeWire's echo-cancel module does this for every program, so Vision leaves it off there;
Windows has nothing Vision can switch on, so "auto" turns it on there when livekit is installed
(the voice extra brings it). [listen] echo_cancel = "on" / "off" overrides either way.
"""
from __future__ import annotations

import sys
import threading

import numpy as np

RATE = 16000  # the microphone's rate; the far end is resampled to it
FRAME = RATE // 100  # the module takes exactly 10 ms at a time

_instance: EchoCanceller | None = None
_instance_lock = threading.Lock()


def wanted(mode: str) -> bool:
    """Whether [listen] echo_cancel asks for the canceller on this platform."""
    mode = str(mode or "auto").strip().lower()
    return mode == "on" or (mode == "auto" and sys.platform == "win32")


def get(mode: str) -> EchoCanceller | None:
    """The process's canceller, created on first use when `mode` wants one and livekit is there;
    None otherwise (then the microphone passes audio through untouched, as before)."""
    global _instance
    if not wanted(mode):
        return None
    with _instance_lock:
        if _instance is None:
            try:
                from livekit import rtc
            except ImportError:
                return None
            _instance = EchoCanceller(rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True,
                                                                high_pass_filter=True))
        return _instance


def current() -> EchoCanceller | None:
    """The canceller if a microphone set one up: what the speaker feeds. Nothing to cancel otherwise."""
    return _instance


def _to_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return audio.astype(np.float32, copy=False)
    n = int(round(len(audio) * RATE / rate))
    if n <= 0:
        return np.zeros(0, np.float32)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


class EchoCanceller:
    """Thread-safe wrapper: the speaker thread calls played(), the microphone's reader cleaned()."""

    def __init__(self, apm, frame_type=None):
        if frame_type is None:
            from livekit import rtc

            frame_type = rtc.AudioFrame
        self._apm = apm
        self._frame = frame_type
        self._lock = threading.Lock()
        self._far = np.zeros(0, np.float32)
        self._near = np.zeros(0, np.int16)
        # One frame of silence up front, so cleaned() can always hand back as many samples as it got
        # (10 ms of constant delay) while it works through whole 10 ms frames.
        self._out = np.zeros(FRAME, np.int16)
        self.output_latency_s = 0.2  # Speaker.OUTPUT_LATENCY_S until play() reports the stream's own
        self.input_latency_s = 0.05

    @property
    def delay_ms(self) -> int:
        """The module's hint for how long played audio takes to come back in: the time it sits in the
        output buffer plus the input's. It adapts to the real path from there."""
        return int((self.output_latency_s + self.input_latency_s) * 1000)

    def played(self, audio: np.ndarray, rate: int) -> None:
        """Audio (float32 mono at `rate`) that is about to go to the speakers."""
        with self._lock:
            self._far = np.concatenate([self._far, _to_rate(np.asarray(audio, np.float32).reshape(-1), rate)])
            while len(self._far) >= FRAME:
                chunk, self._far = self._far[:FRAME], self._far[FRAME:]
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                self._apm.process_reverse_stream(self._frame(pcm, RATE, 1, FRAME))

    def cleaned(self, pcm: bytes) -> bytes:
        """The microphone's int16 16 kHz audio with Vision's voice taken out; always the same length."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        with self._lock:
            self._near = np.concatenate([self._near, samples])
            done = []
            while len(self._near) >= FRAME:
                chunk, self._near = self._near[:FRAME], self._near[FRAME:]
                frame = self._frame(chunk.tobytes(), RATE, 1, FRAME)
                self._apm.set_stream_delay_ms(self.delay_ms)
                self._apm.process_stream(frame)
                done.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
            if done:
                self._out = np.concatenate([self._out, *done])
            out, self._out = self._out[:len(samples)], self._out[len(samples):]
        return out.tobytes()

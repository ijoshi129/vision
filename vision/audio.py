"""Microphone capture with voice-activity end-pointing (hands-free) or push-to-talk."""
from __future__ import annotations

import collections
import sys
import threading
import time

import numpy as np

from vision.config import ListenConfig, resolve_device

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000  # samples per VAD frame


class Microphone:
    def __init__(self, cfg: ListenConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.input_device, "input")
        import webrtcvad

        self._vad = webrtcvad.Vad(int(cfg.vad_aggressiveness))

    def _stream(self):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_LEN, device=self.device
        )

    def record_utterance(
        self,
        on_speech_start=None,
        cancel: threading.Event | None = None,
        start_timeout_s: float | None = None,
    ) -> np.ndarray | None:
        """Wait for speech, record until end_silence_ms of quiet. Returns float32 16 kHz or None."""
        end_silence_frames = max(1, int(self.cfg.end_silence_ms / FRAME_MS))
        max_frames = int(self.cfg.max_utterance_s * 1000 / FRAME_MS)
        timeout = self.cfg.start_timeout_s if start_timeout_s is None else start_timeout_s
        preroll = collections.deque(maxlen=int(400 / FRAME_MS))  # keep 400 ms before speech
        frames: list[bytes] = []
        voiced_run = 0
        silence_run = 0
        started = False
        t0 = time.monotonic()
        # Adaptive noise floor so a loud room doesn't trigger on hiss.
        noise = 0.0
        with self._stream() as stream:
            while True:
                if cancel is not None and cancel.is_set():
                    return None
                data, _ = stream.read(FRAME_LEN)
                pcm = bytes(data)
                samples = np.frombuffer(pcm, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
                if not started:
                    noise = rms if noise == 0 else 0.95 * noise + 0.05 * rms
                loud = rms > max(0.006, noise * 2.5) if not started else rms > max(0.004, noise * 1.5)
                is_speech = loud and self._vad.is_speech(pcm, SAMPLE_RATE)
                if not started:
                    preroll.append(pcm)
                    if is_speech:
                        voiced_run += 1
                        if voiced_run >= 3:  # ~90 ms of speech
                            started = True
                            frames.extend(preroll)
                            if on_speech_start:
                                on_speech_start()
                    else:
                        voiced_run = 0
                    if timeout and time.monotonic() - t0 > timeout:
                        return None
                    continue
                frames.append(pcm)
                if is_speech:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= end_silence_frames:
                        break
                if len(frames) >= max_frames:
                    break
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def record_until_enter(self, prompt: str = "") -> np.ndarray:
        """Push-to-talk: record until the user presses Enter."""
        stop = threading.Event()

        def waiter():
            try:
                sys.stdin.readline()
            finally:
                stop.set()

        threading.Thread(target=waiter, daemon=True).start()
        frames: list[bytes] = []
        with self._stream() as stream:
            while not stop.is_set():
                data, _ = stream.read(FRAME_LEN)
                frames.append(bytes(data))
        return np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

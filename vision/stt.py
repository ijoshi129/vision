"""Speech-to-text with faster-whisper (CTranslate2), CUDA when available."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from vision.config import ListenConfig

SAMPLE_RATE = 16000
# Whisper's well-known hallucinations on silence / noise.
_HALLUCINATIONS = {
    "thank you.", "thanks for watching.", "thank you for watching.", "you", "you.", "bye.", "thank you", ".", "",
    "subtitles by the amara.org community", "thanks for watching!",
}


from vision.cuda import preload as _preload_cuda_libs


class Transcriber:
    def __init__(self, cfg: ListenConfig):
        self.cfg = cfg
        self._model = None
        self.device = "?"
        self._lock = threading.Lock()

    def load(self, quiet: bool = False) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1" if quiet else "0")
            from faster_whisper import WhisperModel

            want = self.cfg.device
            attempts = []
            if want in ("auto", "cuda"):
                attempts.append(("cuda", "float16", self.cfg.whisper_model))
            if want in ("auto", "cpu"):
                cpu_model = self.cfg.whisper_model
                if want == "auto" and "large" in cpu_model:
                    cpu_model = "small.en"  # large models are too slow on CPU for live use
                attempts.append(("cpu", "int8", cpu_model))
            last = None
            for dev, ct, name in attempts:
                try:
                    if dev == "cuda":
                        _preload_cuda_libs()
                    self._model = WhisperModel(name, device=dev, compute_type=ct)
                    # Touch the encoder once so missing CUDA libs fail here, not mid-conversation.
                    self._model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1, language="en")
                    self.device = f"{dev}/{name}"
                    return
                except Exception as e:  # noqa: BLE001
                    last = e
                    self._model = None
            raise RuntimeError(f"Could not load Whisper model: {last}")

    def warm_up(self) -> None:
        self.load(quiet=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        self.load(quiet=True)
        if sample_rate != SAMPLE_RATE:
            idx = np.arange(0, len(audio), sample_rate / SAMPLE_RATE)
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        if audio.size < SAMPLE_RATE * 0.25:
            return ""
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            beam_size=2,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        pieces = []
        for s in segments:
            if s.no_speech_prob > 0.8 and s.avg_logprob < -1.0:
                continue
            pieces.append(s.text.strip())
        text = " ".join(p for p in pieces if p).strip()
        if text.lower() in _HALLUCINATIONS:
            return ""
        return text

    def transcribe_file(self, path: str) -> str:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe(audio, sr)

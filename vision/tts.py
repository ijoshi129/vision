"""Text-to-speech with Kokoro (ONNX). Streams sentences to the speaker as they arrive."""
from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

from vision.config import MODELS_DIR, VOICE_PRESETS, VoiceConfig, resolve_device

SAMPLE_RATE = 24000
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]?\s+|\n+")
_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|St|No|approx)\.$", re.I)


class TTSError(RuntimeError):
    pass


def model_files() -> tuple[Path, Path]:
    return MODELS_DIR / "kokoro-v1.0.onnx", MODELS_DIR / "voices-v1.0.bin"


def models_present() -> bool:
    m, v = model_files()
    return m.exists() and v.exists()


# ---------------------------------------------------------------- text cleanup
_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_EMPH = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF⬀-⯿️]+"
)


def speechify(text: str) -> str:
    """Convert markdown-ish text into something that reads well aloud."""
    text = _FENCE.sub(" I've put the code on screen. ", text)
    text = _TABLE_ROW.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _HEADING.sub("", text)
    text = _EMPH.sub(r"\2", text)
    text = _EMOJI.sub("", text)
    # bullets become sentences
    lines = []
    for line in text.splitlines():
        if _BULLET.match(line):
            line = _BULLET.sub("", line).strip()
            if line and line[-1] not in ".!?:":
                line += "."
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _lang_for(voice_name: str) -> str:
    return "en-gb" if voice_name.startswith("b") else "en-us"


# ---------------------------------------------------------------- engine
class Speaker:
    """Lazy-loading Kokoro wrapper with an output stream and a sentence queue."""

    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        self._kokoro = None
        self._style = None
        self._lang = "en-gb"
        self._voice_desc = cfg.voice
        self.device = "?"
        self._out_device = resolve_device(cfg.output_device, "output")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._playing = threading.Event()

    # -- loading
    def _load(self):
        if self._kokoro is not None:
            return
        with self._lock:
            if self._kokoro is not None:
                return
            if not models_present():
                raise TTSError("Kokoro model files missing. Run `vision setup`.")
            import logging

            logging.getLogger("kokoro_onnx").setLevel(logging.WARNING)
            import onnxruntime as rt
            from kokoro_onnx import Kokoro

            m, v = model_files()
            self._kokoro = Kokoro.from_session(self._session(rt, m), str(v))
            self.set_voice(self.cfg.voice)
            # First inference on CUDA compiles kernels; do it now, not on the first reply.
            self._kokoro.create("Ready.", voice=self._style, speed=1.0, lang=self._lang)

    def _session(self, rt, model_path):
        """Build an ONNX Runtime session on the GPU when possible, else CPU."""
        rt.set_default_logger_severity(3)  # silence benign ScatterND / conv warnings
        so = rt.SessionOptions()
        so.log_severity_level = 3
        want = self.cfg.device or "auto"
        if want in ("auto", "cuda") and "CUDAExecutionProvider" in rt.get_available_providers():
            try:
                from vision.cuda import preload

                preload()
                # HEURISTIC avoids cuDNN's exhaustive autotune (40 s warm-up on first run).
                sess = rt.InferenceSession(
                    str(model_path), so,
                    providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
                )
                if sess.get_providers()[0] == "CUDAExecutionProvider":
                    self.device = "cuda"
                    return sess
            except Exception:  # noqa: BLE001
                if want == "cuda":
                    raise
        self.device = "cpu"
        return rt.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

    @property
    def voice(self) -> str:
        return self._voice_desc

    def available_voices(self) -> list[str]:
        self._load()
        return sorted(self._kokoro.get_voices())

    def set_voice(self, spec: str) -> None:
        """spec: preset name, a voice name, or 'name:weight,name:weight' blend."""
        self._load()
        spec = VOICE_PRESETS.get(spec.strip().lower(), spec).strip()
        parts = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, _, w = chunk.partition(":")
            name = name.strip()
            if name not in self._kokoro.get_voices():
                raise TTSError(f"Unknown voice {name!r}. Run `vision voices`.")
            parts.append((name, float(w) if w else 1.0))
        if not parts:
            raise TTSError("Empty voice spec.")
        total = sum(w for _, w in parts)
        style = sum(self._kokoro.get_voice_style(n) * (w / total) for n, w in parts)
        self._style = style.astype(np.float32)
        self._lang = _lang_for(parts[0][0])
        self._voice_desc = spec

    # -- synthesis
    def synth(self, text: str, speed: float | None = None) -> np.ndarray:
        self._load()
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        audio, sr = self._kokoro.create(
            text, voice=self._style, speed=speed or self.cfg.speed, lang=self._lang, trim=True
        )
        if sr != SAMPLE_RATE:
            raise TTSError(f"Unexpected sample rate {sr}")
        return audio.astype(np.float32)

    def save(self, text: str, path: str | Path, speed: float | None = None) -> Path:
        import soundfile as sf

        audio = self.synth(speechify(text), speed)
        sf.write(str(path), audio, SAMPLE_RATE)
        return Path(path)

    # -- playback
    def stop(self) -> None:
        self._stop.set()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def open_stream(self):
        import sounddevice as sd

        return sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=self._out_device)

    def play(self, audio: np.ndarray, stream=None) -> None:
        """Blocking playback that honours stop(). Pass an open stream to avoid per-chunk gaps."""
        if audio.size == 0:
            return
        self._playing.set()
        try:
            own = stream is None
            if own:
                stream = self.open_stream()
                stream.start()
            step = 2400  # 100 ms
            for i in range(0, len(audio), step):
                if self._stop.is_set():
                    stream.abort()
                    break
                stream.write(audio[i : i + step].reshape(-1, 1))
            if own:
                stream.stop()
                stream.close()
        finally:
            self._playing.clear()

    def say(self, text: str, speed: float | None = None) -> None:
        """Speak a full text now: chunk by sentence so playback starts quickly."""
        self._stop.clear()
        s = StreamingSpeaker(self, speed=speed)
        s.feed(text)
        s.finish()


class StreamingSpeaker:
    """Feed text deltas in; sentences are synthesised and played in order, overlapped."""

    def __init__(self, speaker: Speaker, speed: float | None = None, min_chars: int = 12):
        self.speaker = speaker
        self.speed = speed
        self.min_chars = min_chars
        self._buf = ""
        self._text_q: queue.Queue[str | None] = queue.Queue()
        self._audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        self._synth_thread = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.speaker._stop.clear()
        self._synth_thread.start()
        self._play_thread.start()

    def feed(self, delta: str) -> None:
        self._buf += delta
        self._flush(final=False)

    def _flush(self, final: bool) -> None:
        while True:
            found = None
            for m in _SENTENCE_END.finditer(self._buf):
                head = self._buf[: m.start()].rstrip()
                if _ABBREV.search(head):
                    continue  # "Dr." / "e.g." are not sentence ends
                if len(head) < self.min_chars and not final:
                    continue  # too short to be worth a separate chunk ("OK.") - merge with the next
                found = m
                break
            if found is None:
                # Long run without punctuation: cut at a clause boundary to keep latency low.
                if len(self._buf) > 220:
                    cut = max(self._buf.rfind(", "), self._buf.rfind("; "), self._buf.rfind(" - "))
                    if cut > 60:
                        self._emit(self._buf[: cut + 1])
                        self._buf = self._buf[cut + 1 :]
                        continue
                break
            self._emit(self._buf[: found.end()])
            self._buf = self._buf[found.end() :]
        if final and self._buf.strip():
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, text: str) -> None:
        clean = speechify(text)
        if clean:
            self._text_q.put(clean)

    def _synth_loop(self) -> None:
        while True:
            item = self._text_q.get()
            if item is None:
                self._audio_q.put(None)
                return
            if self.speaker._stop.is_set():
                continue
            try:
                audio = self.speaker.synth(item, self.speed)
            except Exception as e:  # keep the pipeline alive on a bad chunk
                print(f"[tts error: {e}]")
                continue
            if audio.size:
                pad = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
                self._audio_q.put(np.concatenate([audio, pad]))

    def _play_loop(self) -> None:
        stream = None
        try:
            while True:
                audio = self._audio_q.get()
                if audio is None:
                    return
                if self.speaker._stop.is_set():
                    continue
                if stream is None:
                    stream = self.speaker.open_stream()
                    stream.start()
                self.speaker.play(audio, stream)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def finish(self) -> None:
        """Flush remaining text and wait until playback completes (or stop() was called)."""
        self._flush(final=True)
        self._text_q.put(None)
        self._synth_thread.join()
        self._play_thread.join()

    def stop(self) -> None:
        self.speaker.stop()
        # Drain so threads exit promptly.
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass


def chime(kind: str = "listen", device=None) -> None:
    """Short UI tones so you know when Vision is listening / done, without looking."""
    import sounddevice as sd

    sr = SAMPLE_RATE
    if kind == "listen":
        notes = [(660, 0.07), (880, 0.09)]
    elif kind == "done":
        notes = [(880, 0.06), (660, 0.08)]
    else:
        notes = [(440, 0.12)]
    parts = []
    for f, d in notes:
        t = np.linspace(0, d, int(sr * d), endpoint=False)
        env = np.minimum(1.0, np.minimum(t / 0.01, (d - t) / 0.02))
        parts.append(0.18 * env * np.sin(2 * np.pi * f * t))
    audio = np.concatenate(parts).astype(np.float32)
    try:
        sd.play(audio, sr, device=device)
        sd.wait()
    except Exception:
        pass
    time.sleep(0.02)

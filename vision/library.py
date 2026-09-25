"""The phone's Library: every picture a reply showed, and every photo, video and file sent from the phone.

Reply pictures are found by scanning the brains' session logs for `![caption](/abs/path.png)`; each log
is rescanned only when it changes (cache keyed on size + mtime), so a refresh after the first is cheap.
Uploads are the files in UPLOAD_DIR (a video's digest folder and json are not items of their own).
"""
from __future__ import annotations

import glob
import hashlib
import io
import os
import re
import shutil
import threading
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
THUMB_DIR = Path.home() / ".cache" / "vision" / "thumbs"

_PICTURE = re.compile(rb'!\[([^\]\n"]{0,200})\]\((/[^)\s"\\]+\.(?:png|jpe?g|webp|gif))\)', re.I)
_logs: dict[str, tuple[int, float, list[tuple[str, str]]]] = {}
_lock = threading.Lock()


def _session_logs() -> list[str]:
    from vision.codex import CODEX_SESSIONS
    from vision.grok import GROK_SESSIONS
    from vision.local import SESSIONS_DIR
    from vision.sessions import CLAUDE_PROJECTS

    paths = glob.glob(os.path.join(CLAUDE_PROJECTS, "*", "*.jsonl"))
    paths += glob.glob(os.path.join(CODEX_SESSIONS, "*", "*", "*", "rollout-*.jsonl"))
    paths += glob.glob(os.path.join(GROK_SESSIONS, "*", "*", "chat_history.jsonl"))
    paths += glob.glob(os.path.join(str(SESSIONS_DIR), "*.json"))
    return paths


def _pictures_in(path: str) -> list[tuple[str, str]]:
    try:
        st = os.stat(path)
    except OSError:
        return []
    cached = _logs.get(path)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime:
        return cached[2]
    found: list[tuple[str, str]] = []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        data = b""
    if b"![" in data:
        seen = set()
        for m in _PICTURE.finditer(data):
            pic = m.group(2).decode("utf-8", "replace")
            if pic not in seen:
                seen.add(pic)
                found.append((pic, m.group(1).decode("utf-8", "replace").replace("\\n", " ").strip()))
    _logs[path] = (st.st_size, st.st_mtime, found)
    return found


def reply_pictures(allowed) -> list[dict]:
    """Pictures replies showed that still exist on disk (and that `allowed(Path)` lets the phone see)."""
    items: dict[str, dict] = {}
    with _lock:
        for log in _session_logs():
            for pic, caption in _pictures_in(log):
                if pic in items:
                    continue
                p = Path(pic)
                try:
                    st = p.stat()
                except OSError:
                    continue
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or not allowed(p.resolve()):
                    continue
                items[pic] = {"kind": "image", "source": "vision", "path": pic, "name": p.name,
                              "caption": caption, "time": st.st_mtime, "size": st.st_size}
    return list(items.values())


def uploads(upload_dir: Path) -> list[dict]:
    out = []
    try:
        entries = list(os.scandir(upload_dir))
    except OSError:
        return out
    for e in entries:
        if not e.is_file() or e.name.endswith((".json", ".tmp")):
            continue
        ext = os.path.splitext(e.name)[1].lower()
        kind = "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS else "file"
        # "20260925-113458-e5d17a-report.pdf" → "report.pdf"; photos keep their stamped name
        name = re.sub(r"^\d{8}-\d{6}-[0-9a-f]{6}-", "", e.name) if kind == "file" else e.name
        st = e.stat()
        out.append({"kind": kind, "source": "you", "path": e.path, "name": name, "caption": "",
                    "time": st.st_mtime, "size": st.st_size})
    return out


def library(upload_dir: Path, allowed) -> list[dict]:
    items = reply_pictures(allowed) + uploads(upload_dir)
    items.sort(key=lambda i: i["time"], reverse=True)
    return items


def delete(path: str, upload_dir: Path, allowed) -> bool:
    """Delete a Library item from disk; only something the Library lists, so the phone can't reach any other
    file. A clip's digest (its json and frames folder) goes with it. False when the Library has no such item."""
    item = next((i for i in library(upload_dir, allowed) if i["path"] == path), None)
    if item is None:
        return False
    p = Path(path)
    p.unlink()
    if item["kind"] == "video":
        p.with_suffix(".json").unlink(missing_ok=True)
        shutil.rmtree(p.with_name(p.stem + "-frames"), ignore_errors=True)
    return True


def thumbnail(path: Path, size: int = 360) -> bytes:
    """A JPEG no bigger than `size` on its long edge, cached on disk by path + mtime."""
    from PIL import Image, ImageOps

    st = path.stat()
    key = hashlib.sha1(f"{path}|{st.st_mtime}|{st.st_size}|{size}".encode()).hexdigest()
    cached = THUMB_DIR / f"{key}.jpg"
    if cached.is_file():
        return cached.read_bytes()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((size, size))
        if im.mode not in ("RGB", "L"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.convert("RGBA").split()[-1])
            im = bg
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
    data = buf.getvalue()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data

"""The few places Windows needs something other than the POSIX call the rest of Vision makes.

Every helper here does exactly what the code did before on Linux and macOS; only the Windows branch
is new. Keep it that way: a behaviour change for everyone belongs at the call site, not here.
"""
from __future__ import annotations

import os
import subprocess
import sys

WINDOWS = sys.platform == "win32"


def _taskkill(proc: subprocess.Popen) -> None:
    """End a process and everything it started. TerminateProcess (Popen.terminate/kill on Windows)
    stops only the direct child, so a brain's tool shells and MCP servers would outlive the turn."""
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    if proc.poll() is None:
        proc.kill()


def terminate(proc: subprocess.Popen) -> None:
    """Popen.terminate(), which on Windows also takes the process's children with it."""
    if WINDOWS:
        _taskkill(proc)
    else:
        proc.terminate()


def kill(proc: subprocess.Popen) -> None:
    """Popen.kill(), which on Windows also takes the process's children with it."""
    if WINDOWS:
        _taskkill(proc)
    else:
        proc.kill()


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists. os.kill(pid, 0) is the POSIX probe; on Windows signal 0
    is CTRL_C_EVENT, so it would interrupt the process instead of asking about it."""
    if not WINDOWS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE, ERROR_ACCESS_DENIED = 0x1000, 259, 5
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED  # exists, just not ours to query
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def lock_nonblocking(fd: int) -> None:
    """An exclusive, non-blocking lock on an open file, released when the fd closes (or the process
    dies). Raises OSError (BlockingIOError on POSIX) when another process holds it."""
    if not WINDOWS:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    import msvcrt

    # Windows byte-range locks are mandatory: lock a byte far past the contents so other processes
    # can still read the file (the holder's pid) while it is held.
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 1 << 30, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    finally:
        os.lseek(fd, pos, os.SEEK_SET)


def copy_text_windows(text: str) -> bool:
    """Put text on the Windows clipboard (CF_UNICODETEXT). False if the clipboard is busy or absent."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
    CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002

    data = (text.replace("\r\n", "\n").replace("\n", "\r\n") + "\0").encode("utf-16-le")
    if not user32.OpenClipboard(None):
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            return False
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)  # still ours when SetClipboardData fails
            return False
        return True
    finally:
        user32.CloseClipboard()


def is_batch_file(path: str) -> bool:
    """A .cmd/.bat launcher (what `npm i -g` installs on Windows). Popen runs those through cmd.exe,
    which re-parses the whole command line: a multi-line argument ends it at the first newline, and
    quotes, % and & in a prompt become cmd syntax. Vision passes the system prompt as an argument."""
    return WINDOWS and path.lower().endswith((".cmd", ".bat"))

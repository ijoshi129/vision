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
    if WINDOWS and getattr(proc, "pid", None) is not None:
        _taskkill(proc)
    else:
        proc.terminate()


def kill(proc: subprocess.Popen) -> None:
    """Popen.kill(), which on Windows also takes the process's children with it."""
    if WINDOWS and getattr(proc, "pid", None) is not None:
        _taskkill(proc)
    else:
        proc.kill()


_jobs: list[int] = []  # job handles, deliberately never closed: Windows closes them when Vision exits


def end_with_this_process(proc: subprocess.Popen) -> None:
    """Windows: put proc in a job object that kills it when this process exits, crash included (the
    job's last handle closes with us). The POSIX callers get the same from a watchdog shell instead."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    JobObjectExtendedLimitInformation, KILL_ON_JOB_CLOSE = 9, 0x2000
    PROCESS_TERMINATE, PROCESS_SET_QUOTA = 0x0001, 0x0100

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    handle = kernel32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, proc.pid)
    if not handle:
        kernel32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.AssignProcessToJobObject(job, handle):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)
    _jobs.append(job)


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


def process_cmdline(pid: int) -> list[str]:
    """Windows: another process's command line as argv, what /proc/<pid>/cmdline gives on Linux.
    [] if it can't be read (gone, or not ours to query)."""
    import ctypes
    from ctypes import wintypes

    class UNICODE_STRING(ctypes.Structure):
        _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", ctypes.c_void_p)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    shell32 = ctypes.WinDLL("shell32")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    ntdll.NtQueryInformationProcess.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG,
                                                ctypes.POINTER(wintypes.ULONG))
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    shell32.CommandLineToArgvW.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
    PROCESS_QUERY_LIMITED_INFORMATION, ProcessCommandLineInformation = 0x1000, 60  # the latter: Windows 8.1+

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return []
    try:
        size = wintypes.ULONG(0)
        ntdll.NtQueryInformationProcess(handle, ProcessCommandLineInformation, None, 0, ctypes.byref(size))
        if not size.value:
            return []
        buf = ctypes.create_string_buffer(size.value)
        if ntdll.NtQueryInformationProcess(handle, ProcessCommandLineInformation, buf, size, ctypes.byref(size)) != 0:
            return []
        text = UNICODE_STRING.from_buffer(buf)
        line = ctypes.wstring_at(text.Buffer, text.Length // 2) if text.Buffer else ""
    finally:
        kernel32.CloseHandle(handle)
    if not line:
        return []
    argc = ctypes.c_int(0)
    argv = shell32.CommandLineToArgvW(line, ctypes.byref(argc))
    if not argv:
        return []
    try:
        return [argv[i] for i in range(argc.value)]
    finally:
        kernel32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


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


def held_elsewhere(fd: int) -> bool:
    """Whether another process holds lock_nonblocking's lock on this file. Only probes: nothing is
    held afterwards."""
    if not WINDOWS:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    import msvcrt

    pos = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 1 << 30, os.SEEK_SET)  # the byte lock_nonblocking locks
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        os.lseek(fd, 1 << 30, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # right away: Windows may keep a closed handle's lock a while
        return False
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


def bash() -> str | None:
    """The bash that runs a local model's shell commands: plain `bash` on POSIX. On Windows, Git for
    Windows' bash.exe (the one Claude Code uses too), never System32\\bash.exe, which is the WSL
    launcher and would run the command in a Linux VM without Vision's PATH shims. None if not found."""
    import shutil

    if not WINDOWS:
        return "bash"
    configured = os.environ.get("CLAUDE_CODE_GIT_BASH_PATH")
    if configured and os.path.isfile(configured):
        return configured
    git = shutil.which("git")
    if git:
        root = os.path.dirname(os.path.dirname(os.path.realpath(git)))  # <Git>\cmd\git.exe → <Git>
        for rel in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
            cand = os.path.join(root, *rel)
            if os.path.isfile(cand):
                return cand
    for base in (os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA") and os.path.join(os.environ["LOCALAPPDATA"], "Programs")):
        cand = base and os.path.join(base, "Git", "bin", "bash.exe")
        if cand and os.path.isfile(cand):
            return cand
    return None


def check_hf_symlinks(repo_id: str) -> None:
    """Windows: run huggingface_hub's symlink test for a model's cache folder before downloading it.
    The library records "supported" before it tries, so its parallel download threads can race the
    test, try a real symlink, and fail with WinError 1314 when Developer Mode is off. Run once, first,
    the test settles on copies. Does nothing on other platforms or for a local path."""
    if not WINDOWS or not repo_id or os.path.isdir(repo_id):
        return
    try:
        from huggingface_hub import constants
        from huggingface_hub.file_download import are_symlinks_supported, repo_folder_name

        constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True  # copies instead of links is the expected outcome here
        are_symlinks_supported(os.path.join(constants.HF_HUB_CACHE, repo_folder_name(repo_id=repo_id, repo_type="model")))
    except Exception:  # noqa: BLE001 - an older or newer huggingface_hub: download as before
        pass


def is_batch_file(path: str) -> bool:
    """A .cmd/.bat launcher (what `npm i -g` installs on Windows). Popen runs those through cmd.exe,
    which re-parses the whole command line: a multi-line argument ends it at the first newline, and
    quotes, % and & in a prompt become cmd syntax. Vision passes the system prompt as an argument."""
    return WINDOWS and path.lower().endswith((".cmd", ".bat"))

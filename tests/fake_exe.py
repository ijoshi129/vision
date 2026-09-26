"""Stand-in executables for tests that need a real command to run (a fake CLI, a fake app-server)."""
from __future__ import annotations

import stat
import sys
from pathlib import Path


def python_command(path: str | Path, source: str) -> str:
    """Write `source` (Python) as a command at `path` and return what to execute.

    POSIX runs the script itself (#! line, executable bit). Windows runs neither, so the script goes
    beside a .cmd that calls this interpreter on it, and the .cmd is what gets executed."""
    path = Path(path)
    if sys.platform != "win32":
        path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)
    script = path.with_name(path.name + ".py")
    # Piped stdout would otherwise be the ANSI code page; the code under test reads UTF-8.
    script.write_text("import sys\nsys.stdout.reconfigure(encoding='utf-8')\n" + source, encoding="utf-8")
    launcher = path.with_name(path.name + ".cmd")
    launcher.write_text(f'@"{sys.executable}" "%~dp0{script.name}" %*\r\n', encoding="utf-8")
    return str(launcher)

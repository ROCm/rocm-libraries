# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""pytest plugin: run as a Windows CI runner would, on Linux (`-p rkmon_winsim`).

Windows lanes fail tests that pass on Linux for a few recurring reasons; this plugin brings each one to a Linux run, so
a container finds them without a Windows machine. RKMON_WINSIM names the simulations (comma-separated; blank or `all`:
text, crlf, tempfile):
- `text`: a text-mode open without an encoding uses the Windows code page (RKMON_WINSIM_ENCODING, default cp1252), as
  `open()`, `Path.read_text()`, `Path.write_text()`, `os.fdopen()` and `tempfile`'s text files do there: a non-ASCII
  file read or written that way fails or reads wrong. `encoding="locale"` reads the same way.
- `crlf`: a text-mode open for writing (w, a, x; not +) without a newline argument writes "\\r\\n" for "\\n", as Windows
  does: bytes read back are not the bytes a test wrote as text.
- `tempfile`: a NamedTemporaryFile that deletes itself when closed (delete=True, delete_on_close=True: the defaults)
  cannot be opened again by its name while it is open: PermissionError, as on Windows.
- `os` (only when named): `os.name` "nt", `os.sep` "\\\\", `os.altsep` "/", `os.linesep` "\\r\\n" while each test's own
  call runs (not its fixtures, nor a module's import-time code: pytest's path handling needs the POSIX values), for
  code that branches on them. os.path stays POSIX (its functions keep working), and os.pathsep stays ":" (the PATH
  lookups keep working); code that imports a Windows-only module on `os.name == "nt"` fails here for that reason
  alone, so a failure under `os` alone is a lead to read, not a verdict.
Not simulated: a child process's own opens and output (subprocess text mode), the 32,767-character command line,
spawn instead of fork, pytest-timeout's thread method.

The simulations are installed when pytest configures the run (a conftest.py's own import-time opens are not
simulated). At the end the run prints "rkmon-winsim:<TAB><the simulations><TAB><the code page>", so a runner knows it was on.
Used by tools/winsim.py (`rk winsim`, the checks' Windows simulation section).
"""

import io
import os
import tempfile
import weakref

try:
    import pytest

    _hookwrapper = pytest.hookimpl(hookwrapper=True)
except ImportError:  # imported for its names outside pytest (tools/winsim.py)

    def _hookwrapper(f):
        return f


KNOWN = ("text", "crlf", "tempfile", "os")
DEFAULT = ("text", "crlf", "tempfile")


def chosen(value: str | None) -> tuple[str, ...]:
    """The simulations a RKMON_WINSIM value names (blank or `all`: the default set); an unknown name is left out."""
    names = [s.strip() for s in (value or "").split(",") if s.strip()]
    if not names or names == ["all"]:
        return DEFAULT
    return tuple(s for s in KNOWN if s in names)


SIMS = chosen(os.environ.get("RKMON_WINSIM"))
ENCODING = os.environ.get("RKMON_WINSIM_ENCODING") or "cp1252"
_open = io.open
_named = tempfile.NamedTemporaryFile
_OPEN_TEMPS: dict = (
    {}
)  # real path -> a weak reference to the NamedTemporaryFile open there


def _refuse_reopen(file) -> None:
    if isinstance(file, int):
        return
    try:
        path = os.path.realpath(os.fsdecode(os.fspath(file)))
    except TypeError:
        return
    ref = _OPEN_TEMPS.get(path)
    f = ref() if ref is not None else None
    if f is None or getattr(f, "closed", True):
        _OPEN_TEMPS.pop(path, None)
        return
    raise PermissionError(
        13,
        "Permission denied (rkmon winsim: Windows cannot open a NamedTemporaryFile again while it is open)",
        os.fsdecode(os.fspath(file)),
    )


def sim_open(
    file,
    mode="r",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    closefd=True,
    opener=None,
):
    """io.open as Windows runs it, for the simulations that are on."""
    if "tempfile" in SIMS:
        _refuse_reopen(file)
    if "b" not in mode:
        if "text" in SIMS and encoding in (None, "locale"):
            encoding = ENCODING
        if (
            "crlf" in SIMS
            and newline is None
            and "+" not in mode
            and any(c in mode for c in "wax")
        ):
            newline = "\r\n"
    return _open(file, mode, buffering, encoding, errors, newline, closefd, opener)


def sim_named(*args, **kwargs):
    """tempfile.NamedTemporaryFile, its name refused to a second open while it is open (as Windows refuses it)."""
    f = _named(*args, **kwargs)
    delete = kwargs.get("delete", args[7] if len(args) > 7 else True)
    if delete and kwargs.get("delete_on_close", True):
        _OPEN_TEMPS[os.path.realpath(f.name)] = weakref.ref(f)
    return f


def install() -> None:
    import builtins

    if "text" in SIMS or "crlf" in SIMS or "tempfile" in SIMS:
        builtins.open = io.open = sim_open
    if "tempfile" in SIMS:
        tempfile.NamedTemporaryFile = sim_named


def pytest_configure(config):
    """Installed when pytest configures the run, before it collects (imports) the tests; importing this module alone
    changes nothing."""
    install()


WINDOWS_OS = ("nt", "\\", "/", "\r\n")  # os.name, os.sep, os.altsep, os.linesep


@_hookwrapper
def pytest_runtest_call(item):
    """`os`: the Windows values around the test's own call only (pytest's path handling needs the POSIX ones)."""
    if "os" not in SIMS:
        yield
        return
    saved = (os.name, os.sep, os.altsep, os.linesep)
    os.name, os.sep, os.altsep, os.linesep = WINDOWS_OS
    try:
        yield
    finally:
        os.name, os.sep, os.altsep, os.linesep = saved


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"rkmon-winsim:\t{','.join(SIMS)}\t{ENCODING}")

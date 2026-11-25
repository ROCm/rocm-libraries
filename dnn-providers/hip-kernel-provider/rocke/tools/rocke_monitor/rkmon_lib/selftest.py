# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's own suite, run the one way: `rk selftest` and the package build's test step (tools/make_package.py)
run it through `run`; tools/sync_from_live.py and the image's test (tests/container/image.py) run `rk selftest`.

- `run(folder, env=None, capture=False, timeout=None, names=(), jobs=1) -> Run`: the suite of the package
  in FOLDER (COMMAND: unittest's discovery over FOLDER/tests, by this interpreter with -B), or, NAMES given, the tests
  they name, as unittest names them from FOLDER (`tests.test_selftest`, `tests.test_selftest.TestSelftest`), after
  GUARD, which the whole suite runs first too; in a child process in a session of its own, with `environ(ENV, own)`
  (ENV: this process's environment when None): its home folder is a new folder of the run's own, made in this process's
  temporary folder, and its TMPDIR and its monitor home are that folder's `folders`. When the suite ends, runs past
  TIMEOUT or is interrupted, every process it left running in its process group ends too (SIGKILL), so none writes any
  more; then the run's folders are read, and FOLDER. A test removes what it makes in TMPDIR, makes a home folder or a
  monitor home of its own when it writes in one, and never writes in the package's folder: when the suite leaves
  anything in TMPDIR, in the home folder (its `folders` aside) or in the monitor home, or an entry appears in FOLDER
  during the run (an ignored file too, and one in its .git), the run fails, its exit code 1 when the tests passed, and
  its last lines name each entry, a folder with a trailing `/` (in FOLDER, the outermost new one, by its path from
  there; the run's own folder, made there when this process's temporary folder is in FOLDER, is not one). The run's
  folder goes then, with any read-only folder a test left in it; a folder that stays (a process the suite started in a
  session of its own can still write there) fails the run too, and the last line names it. CAPTURE: the output comes
  back in `Run.output` (stdout, then stderr, then those last lines); else it goes to this process's terminal, and those
  lines to its stderr. TIMEOUT, in seconds: subprocess.TimeoutExpired past it, once the suite's processes have ended and
  the folder is removed. SIGTERM and SIGHUP (`timeout`, `kill`, a terminal that closes) stop the run as an interrupt
  does: SystemExit, its code 128 + the signal's number, once the suite's processes have ended and the folder is removed.
  That holds in the main thread, the only one a handler can be set from, and for a signal left at its default: one the
  caller ignores (SIGHUP under nohup) or handles stays so. SIGKILL, which no process can handle, leaves the suite
  running.
- JOBS above 1, with no NAMES, splits the whole suite by module (`modules`): up to JOBS child processes at a time each
  run one module's tests as the discovery loads them (UNIT), after loading GUARD, whose tests run once, in its own
  module's process; each with a home folder of its own, a numbered folder in the run's folder, its TMPDIR and its
  monitor home that folder's `folders`, read as above once it ends, what it leaves named after its module
  (`test_x: left.txt`); FOLDER is read before the first and after the last. The modules start the largest first
  (`order`). Each module's output comes as it ends, but for the last lines unittest writes ("Ran N tests in Xs" and
  the result, in color or not); then come those of the whole run: "Ran N tests in Xs", the tests of every module and
  the run's own time, and the result, "OK" or "FAILED (failures=F, errors=E)" with what was skipped, as unittest
  writes them; a module whose process ends without its result counts as an error and is named. The exit code is the
  first nonzero one a module's process ended with, else as above. Past TIMEOUT, subprocess.TimeoutExpired names the
  command of the module that has run the longest. With NAMES, the tests named run in one process, as with JOBS 1.
- `modules(folder) -> list`: the file names of the suite's modules in FOLDER/tests, in the order unittest's discovery
  finds them (its pattern PATTERN and its rule for a module's name); CannotSplit, a ValueError, for a package in that
  folder, whose tests discovery reads and the split would not, before a split run starts anything.
- `order(folder, names) -> list`: the modules NAMES (file names in FOLDER/tests), the largest file first, files of one
  size in the order of NAMES: a large module tends to run long, and a long one that starts last keeps the run waiting.
- `environ(base, own) -> dict`: the suite's environment from BASE, one for every run (`rk selftest`, the package build
  and the image's test alike): no name TEST_ENV_DROP starts, the test tools' places (TEST_TOOLS)
  aside, so no test sees a login, a key, a gateway or a host's name, a caller's git settings (its GIT_DIR, a git
  hook's or `git rebase --exec`'s, would point the suite's git commands at its repository), the monitor's settings or
  the image's switch (the tests run in the image as on a host; a test of the image's ways sets it itself); no bytecode
  written, by the tests' own child processes either (PYTHONDONTWRITEBYTECODE); HOME the folder OWN, with XDG's folders
  in it (XDG_HOMES), and TMPDIR and RKMON_HOME its `folders`, so no test sees its caller's home folder or the home of
  the monitor that runs it (the image's RKMON_HOME included), and a temporary file is in the home folder (a message
  shows it as `~/…`) wherever the suite runs.
- `folders(own)`: the suite's TMPDIR and its monitor home, OWN's folders TMP and MONITOR, made new and empty.
- `Run`: the run's `returncode`, its `output` ("" when not captured), the entries named, sorted: `left` in TMPDIR,
  `in_home` in the home folder, `in_monitor_home` in the monitor home, `added` in FOLDER; and `stays`, the run's
  folder's path when it could not be removed ("" when it went).

The suite runs in a child process, so what the tests' own exit handlers remove (tests/test_00_no_github.py's fake gh,
tests/test_publish.py's key) is gone before the folders are read. Each child process has no input (/dev/null), so a
test that reads its input never waits for its caller's.
"""

from __future__ import annotations

import contextlib
import fnmatch
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.loader
from dataclasses import dataclass

from . import wrapping

# The suite's folder in the package's folder, and its command after the interpreter, run in the package's folder.
TESTS = "tests"
UNITTEST = ("-B", "-m", "unittest")
COMMAND = (*UNITTEST, "discover", "-s", TESTS)
# The suite's first module, which puts a gh that refuses every call first on PATH: no test reaches GitHub.
GUARD = "tests.test_00_no_github"
# What points git at a repository from the environment, which the pytest path drops (tests/conftest.py); `environ`
# drops every git name (TEST_ENV_DROP).
GIT_NAMES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
)
# What the suite never sees of its caller's environment, by how a name starts: the monitor's own settings (its home,
# the image's switch RKMON_CONTAINER), the logins', the keys' and the gateways' (gh's, GitHub's tokens, the atlassian
# CLI's, Claude's, the Anthropic API's and the Ollama model server's), the Windows host's name (WINHOST), and git's;
# the test tools' places aside (TEST_TOOLS).
TEST_ENV_DROP = (
    "RKMON_",
    "GH_",
    "ATLASSIAN_",
    "CLAUDE",
    "ANTHROPIC_",
    "GIT_",
    "GITHUB_",
    "OLLAMA_",
    "WINHOST",
)
# The test tools' places, which the suite sees: the checks venv (RKMON_VENV), the probes' torch python (RKMON_TORCH_PY)
# and the pinned black (RKMON_BLACK).
TEST_TOOLS = ("RKMON_VENV", "RKMON_TORCH_PY", "RKMON_BLACK")
# How the name of a run's own folder in this process's temporary folder starts: the suite's home folder, or for a split
# run, the folder of its modules' home folders.
FOLDER_PREFIX = "rkmon-tests-"
# The suite's own folders in the run's folder, its home folder (HOME): its TMPDIR and its monitor home (RKMON_HOME).
TMP = "tmp"
MONITOR = "monitor"
# XDG's folders, by the name that points at each, where a home folder has them when no name does (XDG's base
# directories): the suite's are in its home folder, whatever its caller's environment names.
XDG_HOMES = {
    "XDG_CONFIG_HOME": ".config",
    "XDG_CACHE_HOME": ".cache",
    "XDG_DATA_HOME": ".local/share",
    "XDG_STATE_HOME": ".local/state",
}
# What a run names, by the Run field that holds it: where the tests left it, and what a test does instead.
CHECKS = (
    (
        "left",
        "left {} in their temporary folder (TMPDIR); a test removes what it makes there",
    ),
    (
        "in_home",
        "left {} in their home folder (HOME); a test that writes in a home folder makes one of its own",
    ),
    (
        "in_monitor_home",
        "left {} in their monitor home (RKMON_HOME); a test that needs a monitor home makes one of its own",
    ),
    (
        "added",
        "added {} to the package's folder; a test writes only in folders of its own",
    ),
)
# The seconds the processes of the suite's group have to be gone once they are killed; the folders are read after
# that all the same.
GONE_S = 5.0
# The signals that stop a run as an interrupt does (see the module's docstring).
STOPS = (signal.SIGTERM, signal.SIGHUP)
# The file names unittest's discovery takes for a module of tests: its default pattern, and its rule for a module's
# name.
PATTERN = inspect.signature(unittest.TestLoader.discover).parameters["pattern"].default
MODULE_NAME = unittest.loader.VALID_MODULE_NAME
# What each child process of a split run runs (python -B -c UNIT TESTS GUARD_FILE MODULE FILE COUNTS, in the package's
# folder): GUARD loaded first as the discovery loads it, so the gh that refuses is first on PATH, its tests left to its
# own module's process; then the discovery over the suite's folder for the one module MODULE (a file's name), run as
# `python -m unittest discover` runs it; then the tests it ran ("run") and its result's COUNTS (JSON), as JSON, in the
# file FILE. Its exit code is unittest's: 1 when a test failed, else 0 (a module with no test too).
UNIT = """\
import json
import sys
import unittest

tests, guard, module, file, counts = sys.argv[1:]
if module != guard:
    unittest.TestLoader().discover(tests, pattern=guard)
argv = [sys.argv[0], "discover", "-s", tests, "-p", module]
result = unittest.main(module=None, argv=argv, exit=False).result
found = {word: len(getattr(result, name)) for word, name in json.loads(counts).items()}
with open(file, "w", encoding="utf-8") as f:
    json.dump({"run": result.testsRun, **found}, f)
sys.exit(not result.wasSuccessful())
"""
# The counts of a result, in the order unittest writes them, each by its word and the result's list it counts
# (unittest.TestResult); those that make it fail.
COUNTS = {
    "failures": "failures",
    "errors": "errors",
    "skipped": "skipped",
    "expected failures": "expectedFailures",
    "unexpected successes": "unexpectedSuccesses",
}
FAILING = ("failures", "errors", "unexpected successes")
# The exit code and the result of a run in which no test ran (unittest, from Python 3.12).
NO_TESTS = (5, "NO TESTS RAN")
# The last lines unittest writes for a run, after its tests' output: its line of dashes, the tests it ran and their
# time, and its result, which it colors when the colors are on (FORCE_COLOR, Python 3.14 on): SGR, a color's escape.
SGR = r"\x1b\[[0-9;]*m"
SUMMARY = re.compile(
    rf"^-{{70}}\nRan \d+ tests? in \d+\.\d+s\n\n(?:{SGR})*(?:OK|FAILED|NO TESTS RAN)[^\n]*\n",
    re.M,
)
# The seconds between two looks at a split run's processes.
POLL_S = 0.05


@dataclass(frozen=True)
class Run:
    """One run of the suite: see the module's docstring."""

    returncode: int
    output: str
    left: tuple
    in_home: tuple
    in_monitor_home: tuple
    added: tuple
    stays: str


class CannotSplit(ValueError):
    """The suite cannot run split by module: see the module's docstring (`modules`)."""


def environ(base, own) -> dict:
    """The suite's environment from the mapping BASE, its home folder OWN: see the module's docstring."""
    own = os.fspath(own)
    env = {
        k: v
        for k, v in base.items()
        if k in TEST_TOOLS or not k.startswith(TEST_ENV_DROP)
    }
    env.update(
        {name: os.path.join(own, where) for name, where in XDG_HOMES.items()},
        PYTHONDONTWRITEBYTECODE="1",
        HOME=own,
        TMPDIR=os.path.join(own, TMP),
        RKMON_HOME=os.path.join(own, MONITOR),
    )
    return env


def folders(own) -> None:
    """The suite's folders in OWN, new and empty: see the module's docstring."""
    for name in (TMP, MONITOR):
        os.mkdir(os.path.join(own, name))


def modules(folder) -> list:
    """The file names of the suite's modules in FOLDER, as the discovery finds them: see the module's docstring."""
    tests = os.path.join(folder, TESTS)
    out = []
    for name in sorted(os.listdir(tests)):
        path = os.path.join(tests, name)
        if os.path.isfile(os.path.join(path, "__init__.py")):
            raise CannotSplit(
                f"{TESTS}/{name} is a package: the discovery reads its tests, and a run split by module would not"
            )
        if os.path.isfile(path) and fnmatch.fnmatch(name, PATTERN):
            if MODULE_NAME.match(name):
                out.append(name)
    return out


def order(folder, names) -> list:
    """The modules NAMES, the largest file first: see the module's docstring."""
    tests = os.path.join(folder, TESTS)
    return sorted(names, key=lambda name: -os.path.getsize(os.path.join(tests, name)))


def _names(path) -> tuple:
    """The entries in the folder PATH by name, sorted, a folder's with a trailing `/`."""
    with os.scandir(path) as entries:
        return tuple(
            sorted(
                e.name + ("/" if e.is_dir(follow_symlinks=False) else "")
                for e in entries
            )
        )


def _inside(path, folder) -> str:
    """PATH by its path from FOLDER when it is in there, else ""."""
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(folder))
    return "" if rel == os.pardir or rel.startswith(os.pardir + os.sep) else rel


def _entries(folder, aside: str) -> set:
    """Every entry in FOLDER, all the way down, by its path from there, a folder's with a trailing `/`: ASIDE (a path
    from FOLDER; "" for none) aside, and what a folder that cannot be read holds.
    """
    found, todo = set(), [""]
    while todo:
        rel = todo.pop()
        try:
            with os.scandir(os.path.join(folder, rel)) as entries:
                for e in entries:
                    path = rel + e.name
                    if path == aside:
                        continue
                    if e.is_dir(follow_symlinks=False):
                        found.add(path + "/")
                        todo.append(path + "/")
                    else:
                        found.add(path)
        except OSError:
            continue
    return found


def _outermost(paths) -> tuple:
    """PATHS, sorted, but for those a folder among them holds."""
    out = []
    for path in sorted(paths):
        if not (out and out[-1].endswith("/") and path.startswith(out[-1])):
            out.append(path)
    return tuple(out)


def _left(own) -> dict:
    """What the tests left in their home folder OWN, by the Run field that names it (CHECKS, FOLDER's aside)."""
    mine = (TMP + "/", MONITOR + "/")
    return {
        "left": _names(os.path.join(own, TMP)),
        "in_home": tuple(n for n in _names(own) if n not in mine),
        "in_monitor_home": _names(os.path.join(own, MONITOR)),
    }


def _found_lines(found: dict, stays: str) -> str:
    """The lines that name what a run found (CHECKS, by its Run field), then its folder when it stays; "" when it found
    nothing."""
    lines = []
    for field, what in CHECKS:
        if found[field]:
            n = wrapping.counted(len(found[field]), "entry", "entries")
            lines += [
                f"FAILED: the tests {what.format(n)}:",
                *(f"  {name}" for name in found[field]),
            ]
    if stays:
        lines.append(
            "FAILED: the tests left their temporary folder, which could not be removed (a process they started may "
            f"still write there): {stays}"
        )
    return "\n".join(lines)


def _end_group(p: subprocess.Popen) -> None:
    """Every process left in the suite's process group (its session's, `p.pid`) ends (SIGKILL), the suite's own process
    is waited for, and then this waits, GONE_S at most, until no process of the group is left.
    """
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:  # no process of the group is left (or none this user may end)
        return
    p.wait()
    until = time.monotonic() + GONE_S
    while time.monotonic() < until:
        try:
            os.killpg(p.pid, 0)
        except OSError:
            return
        time.sleep(0.05)


@contextlib.contextmanager
def _stops():
    """While a run lasts, each of the STOPS signals left at its default raises SystemExit (128 + its number), so the
    run's own cleanup ends the suite's processes and removes its folder. The first one ignores them all from then on,
    so none cuts that cleanup short; after the run, each is back at its default."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    mine = [s for s in STOPS if signal.getsignal(s) == signal.SIG_DFL]

    def stop(signum, frame):
        for s in mine:
            signal.signal(s, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    for s in mine:
        signal.signal(s, stop)
    try:
        yield
    finally:
        for s in mine:
            signal.signal(s, signal.SIG_DFL)


def _text(f) -> str:
    """What the suite wrote to the file F."""
    f.seek(0)
    return f.read().decode("utf-8", "replace")


@contextlib.contextmanager
def _own_folder(folder):
    """A run's own folder, new in this process's temporary folder (FOLDER_PREFIX), while the run lasts: its path, and a
    function that returns the outermost entries added to FOLDER since, the run's own folder aside when it is in there.
    The folder is removed when the run ends (one that cannot be stays), and the STOPS signals are back at their
    defaults after that (`_stops`)."""
    with _stops(), tempfile.TemporaryDirectory(
        prefix=FOLDER_PREFIX, ignore_cleanup_errors=True
    ) as own:
        aside = _inside(own, folder)
        before = _entries(folder, aside)
        yield own, lambda: _outermost(_entries(folder, aside) - before)


def _end(own, found: dict, code: int, output: str, capture: bool) -> Run:
    """A run once its own folder OWN is removed or stays: the lines that name what it FOUND (by Run field) and OWN when
    it stays, after OUTPUT when CAPTURE, else on stderr; its exit code CODE, else 1 when there are such lines.
    """
    stays = own if os.path.lexists(own) else ""
    lines = _found_lines(found, stays)
    if lines and not capture:
        print(lines, file=sys.stderr, flush=True)
    output += lines + "\n" if lines and capture else ""
    return Run(code or (1 if lines else 0), output, stays=stays, **found)


def run(
    folder,
    env=None,
    capture: bool = False,
    timeout=None,
    names=(),
    jobs: int = 1,
) -> Run:
    """The suite of the package in FOLDER, or the tests NAMES names, with a home folder, a TMPDIR and a monitor home of
    its own, FOLDER as it was, split by module in JOBS processes at a time, the largest first: see the module's
    docstring.
    """
    base = os.environ if env is None else env
    if jobs > 1 and not names:
        return _split(folder, base, capture, timeout, jobs)
    # GUARD first, once, then the tests named.
    argv = (*UNITTEST, *dict.fromkeys((GUARD, *names))) if names else COMMAND
    with contextlib.ExitStack() as stack:
        own, added = stack.enter_context(_own_folder(folder))
        folders(own)
        # The captured output goes to files, not pipes: a process the suite leaves running cannot keep the run waiting
        # for the end of its output.
        out, err = (
            [stack.enter_context(tempfile.TemporaryFile()) for _ in range(2)]
            if capture
            else (None, None)
        )
        p = subprocess.Popen(
            [sys.executable, *argv],
            cwd=folder,
            env=environ(base, own),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,  # a process group of its own, to end with it
        )
        try:
            p.wait(timeout=timeout)
        finally:
            _end_group(p)
        found = {**_left(own), "added": added()}
        output = _text(out) + _text(err) if capture else ""
    return _end(own, found, p.returncode, output, capture)


class _Module:
    """One module's child process in a split run: its file name, its home folder (the run's folder ROOT's I-th: a
    number, not the module's name, which a path a test sees would hold), the file its counts go to, its output files
    and its process."""

    def __init__(self, folder, base, root: str, i: int, name: str, guard: str):
        self.name = name
        self.home = os.path.join(root, str(i))
        os.mkdir(self.home)
        folders(self.home)
        self.counts = self.home + ".json"
        self.out, self.err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        try:
            self.p = subprocess.Popen(
                [sys.executable, "-B", "-c", UNIT, TESTS, guard, name, self.counts]
                + [json.dumps(COUNTS)],
                cwd=folder,
                env=environ(base, self.home),
                stdin=subprocess.DEVNULL,
                stdout=self.out,
                stderr=self.err,
                start_new_session=True,  # a process group of its own, to end with it
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.out.close()
        self.err.close()

    def result(self) -> dict:
        """Its result's counts (COUNTS, and "run"), None when its process ended without them."""
        try:
            with open(self.counts, encoding="utf-8") as f:
                counts = json.load(f)
        except (OSError, ValueError):
            return None
        return counts if isinstance(counts, dict) else None


def _result(totals: dict) -> tuple:
    """A run's exit code and result from its counts TOTALS, as `python -m unittest` ends: 1 and "FAILED", or 5 and "NO
    TESTS RAN" when no test ran and none was skipped (Python 3.12 on), or 0 and "OK"; then the counts that are not 0,
    failures and errors for a run that failed only."""
    words = tuple(COUNTS)
    shown = words[2:]
    if any(totals[k] for k in FAILING):
        code, word, shown = 1, "FAILED", words
    elif not totals["run"] and not totals["skipped"] and sys.version_info >= (3, 12):
        code, word = NO_TESTS
    else:
        code, word = 0, "OK"
    infos = [f"{k}={totals[k]}" for k in shown if totals[k]]
    return code, word + (f" ({', '.join(infos)})" if infos else "")


def _split(folder, base, capture: bool, timeout, jobs: int) -> Run:
    """The whole suite split by module, JOBS processes at a time, the largest first: see the module's docstring."""
    guard = GUARD.rsplit(".", 1)[1] + ".py"
    # Each module with its number, its home folder's name in the run's folder.
    waiting = list(enumerate(order(folder, modules(folder)), 1))
    totals = dict.fromkeys(("run", *COUNTS), 0)
    found = {field: [] for field, _ in CHECKS[:3]}
    outs, errs, codes = [], [], []
    with _own_folder(folder) as (root, added):
        start = time.perf_counter()
        until = None if timeout is None else time.monotonic() + timeout
        running = []
        try:
            while waiting or running:
                while waiting and len(running) < jobs:
                    running.append(_Module(folder, base, root, *waiting.pop(0), guard))
                ended = [m for m in running if m.p.poll() is not None]
                if not ended:
                    if until is not None and time.monotonic() > until:
                        # The command of the module that has run the longest.
                        raise subprocess.TimeoutExpired(running[0].p.args, timeout)
                    time.sleep(POLL_S)
                for m in ended:
                    running.remove(m)
                    _end_group(m.p)
                    out, err = _text(m.out), _text(m.err)
                    m.close()
                    counts = m.result()
                    if counts is None:
                        totals["errors"] += 1
                        err += (
                            f"FAILED: {TESTS}/{m.name}'s process ended with exit code {m.p.returncode} before its "
                            "result\n"
                        )
                    else:
                        for k in totals:
                            totals[k] += int(counts.get(k) or 0)
                        # unittest's last lines for the module: the run's own come at its end.
                        ends = list(SUMMARY.finditer(err))
                        if ends:
                            err = err[: ends[-1].start()] + err[ends[-1].end() :]
                    codes.append(m.p.returncode)
                    module = m.name[: -len(".py")]
                    for field, entries in _left(m.home).items():
                        found[field] += [f"{module}: {e}" for e in entries]
                    if capture:
                        outs.append(out)
                        errs.append(err)
                    else:
                        sys.stdout.write(out)
                        sys.stdout.flush()
                        sys.stderr.write(err)
                        sys.stderr.flush()
        finally:
            for m in running:
                _end_group(m.p)
                m.close()
        found = {**{f: tuple(sorted(e)) for f, e in found.items()}, "added": added()}
        took = time.perf_counter() - start
    code, result = _result(totals)
    n = totals["run"]
    summary = (
        f"{'-' * 70}\nRan {n} test{'s' if n != 1 else ''} in {took:.3f}s\n\n{result}\n"
    )
    if capture:
        output = "".join(outs) + "".join(errs) + summary
    else:
        output = ""
        sys.stderr.write(summary)
        sys.stderr.flush()
    return _end(root, found, next((c for c in codes if c), code), output, capture)

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's own suite as `rk selftest`, the package build and the image's test run it (rkmon_lib/selftest.py),
over a stand-in package of one test, a first module standing in for the suite's guard against GitHub, and one more
module, which write what they see outside the package: the suite's home folder is a new folder of the run's own, with
XDG's folders in it and its TMPDIR and its monitor home new, empty folders in there, all gone after the run, never the
home folder or the monitor home the caller's environment names; a test that leaves anything there, or that adds
anything to the package's folder (an ignored file too), fails the run, which names each entry; a run of the tests named
runs them after the guard, with the same checks. The suite runs with none of its caller's logins, keys, gateways, host
name, git settings or monitor settings, the test tools' places aside, without the image's switch, and writes no
bytecode. A process the suite leaves running ends with the suite, before the folders are read, when the suite runs past
its timeout too, and when SIGTERM or SIGHUP stop the run, unless its caller ignores them; a run in another thread leaves
them as they are. A folder that cannot be removed fails the run, which names it. The image's test runs the suite in a
read-only package folder, as the image's is, in the environment `rk selftest` gives it. A run split by module (JOBS, `rk
selftest --jobs N`) runs each module in a process and a home folder of its own, after the guard, the largest first, and
writes nothing in its caller's home folder; it ends as the suite in one process does, with the same tests, counts,
failures and errors, exit code and checks, what a module leaves named after it.
"""

import contextlib
import errno
import io
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests" / "container"), str(ROOT / "tools"), str(ROOT)]

import image  # noqa: E402
import make_package  # noqa: E402
from rkmon_lib import selftest  # noqa: E402

# The kinds of names, by how a name starts, that a caller's environment never hands the tests, the test tools' places
# aside: the monitor's own, the logins', the keys' and the gateways', the Windows host's name, and git's.
KINDS = (
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
# The stand-in's one test writes, in the folder STANDIN_OUT names, outside the package: `standin` to ran.txt; the
# TMPDIR it runs with and what was in it; its monitor home (RKMON_HOME) and what was in it; its home folder (HOME), what
# was in it and the XDG_ names it has; the names of KINDS it has (RKMON_HOME aside) and PYTHONDONTWRITEBYTECODE; and
# what its input is (stdin.txt: the file its stdin reads, as /proc names it).
# Then it does what STANDIN says: "leave" leaves a file and a folder with a read-only folder in it in TMPDIR; "home"
# leaves a file and a tool's cache (XDG_CACHE_HOME's) in its home folder; "monitor" leaves a file and a folder with a
# folder in it in its monitor home; "litter" adds LITTER to the package's folder: a file, a folder with a folder in it,
# a folder in tests/ and a folder no one can read; "git" adds a file to the package's .git (a repository the test makes
# there); "writer" leaves a process running, on the suite's stdout, that writes w0, w1 and w2 in TMPDIR over and over,
# and a byte to `beat` beside ran.txt for each, until `stop` is there, TMPDIR gone or not (a failed redirection of
# `true` leaves the shell running; one of `:`, a special built-in, would end it); "sleep" sleeps a minute; "exit" ends
# its process, exit code 3, before any result; "fail" fails.
STANDIN = """import json, os, pathlib, subprocess, tempfile, time, unittest

WRITER = 'i=0; while [ ! -e "$3" ]; do true > "$1/w$((i % 3))"; printf . >> "$2"; i=$((i + 1)); done'


class T(unittest.TestCase):
    def test_it(self):
        here, out = pathlib.Path(__file__).parent, pathlib.Path(os.environ["STANDIN_OUT"])
        with open(out / "ran.txt", "a") as f:
            f.write("standin\\n")
        tmp, monitor = pathlib.Path(tempfile.gettempdir()), os.environ.get("RKMON_HOME", "")
        (out / "seen.txt").write_text("\\n".join([str(tmp), *sorted(os.listdir(tmp))]))
        there = sorted(os.listdir(monitor)) if monitor and os.path.isdir(monitor) else None
        (out / "monitor.json").write_text(json.dumps([monitor, there]))
        home = pathlib.Path(os.environ["HOME"])
        xdg = {k: v for k, v in os.environ.items() if k.startswith("XDG_")}
        (out / "home.json").write_text(json.dumps([str(home), sorted(os.listdir(home)), xdg]))
        kinds, bytecode = KINDS, "PYTHONDONTWRITEBYTECODE"
        env = {k: v for k, v in os.environ.items() if k.startswith(kinds) and k != "RKMON_HOME" or k == bytecode}
        (out / "env.json").write_text(json.dumps(env))
        (out / "stdin.txt").write_text(os.readlink("/proc/self/fd/0"))
        does = os.environ.get("STANDIN", "")
        if "leave" in does:
            (tmp / "left.txt").write_text("x")
            (tmp / "left-folder" / "locked").mkdir(parents=True)
            (tmp / "left-folder" / "locked" / "f").write_text("x")
            (tmp / "left-folder" / "locked").chmod(0o500)
        if "home" in does:
            (pathlib.Path(os.environ["XDG_CACHE_HOME"]) / "tool").mkdir(parents=True)
            (home / ".toolrc").write_text("x")
        if "monitor" in does:
            (pathlib.Path(monitor) / "state" / "jobs").mkdir(parents=True)
            (pathlib.Path(monitor) / "config.json").write_text("{}")
        if "litter" in does:
            (here.parent / "notes.txt").write_text("x")
            (here.parent / "state" / "jobs").mkdir(parents=True)
            (here / "__pycache__").mkdir()
            (here / "__pycache__" / "test_standin.pyc").write_bytes(b"")
            (here.parent / "unread").mkdir()
            (here.parent / "unread" / "f").write_text("x")
            (here.parent / "unread").chmod(0o300)
        if "git" in does:
            (here.parent / ".git" / "litter").write_text("x")
        if "writer" in does:
            argv = ["sh", "-c", WRITER, "writer", str(tmp), str(out / "beat"), str(out / "stop")]
            subprocess.Popen(argv, stderr=subprocess.DEVNULL)
            while not (out / "beat").exists():
                time.sleep(0.01)
        if "sleep" in does:
            time.sleep(60)
        if "exit" in does:
            os._exit(3)
        self.assertNotIn("fail", does)
""".replace(
    "KINDS", repr(KINDS)
)
# The stand-in's first module, where the package's guard against GitHub is (selftest.GUARD): it writes `guard` to
# ran.txt when it is imported, and `guard test` when its test runs. The other module writes `other` when it is imported.
STANDIN_GUARD = """import os, pathlib, unittest

RAN = pathlib.Path(os.environ["STANDIN_OUT"]) / "ran.txt"
with open(RAN, "a") as f:
    f.write("guard\\n")


class G(unittest.TestCase):
    def test_g(self):
        with open(RAN, "a") as f:
            f.write("guard test\\n")
"""
STANDIN_OTHER = """import os, pathlib

with open(pathlib.Path(os.environ["STANDIN_OUT"]) / "ran.txt", "a") as f:
    f.write("other\\n")
"""
# A fourth module for the runs split by module (`more`): when it is imported, it writes to more.json whether the guard
# was loaded before it, its home folder and its TMPDIR; its four tests pass, are skipped, fail when STANDIN holds
# "more-fail" and raise an error when it holds "more-error".
STANDIN_MORE = """import json, os, pathlib, sys, tempfile, unittest

DOES = os.environ.get("STANDIN", "")
SEEN = {"guard first": "test_00_no_github" in sys.modules, "home": os.environ["HOME"], "tmp": tempfile.gettempdir()}
(pathlib.Path(os.environ["STANDIN_OUT"]) / "more.json").write_text(json.dumps(SEEN))


class M(unittest.TestCase):
    def test_passes(self):
        pass

    @unittest.skip("on purpose")
    def test_skipped(self):
        pass

    def test_fails_when_told(self):
        self.assertNotIn("more-fail", DOES, "the failure the run reports")

    def test_errs_when_told(self):
        if "more-error" in DOES:
            raise RuntimeError("the error the run reports")
"""
# A stand-in's rkmon.py: this package's, its package folder the stand-in's (ROOT: this package's folder).
LAUNCHER = """import pathlib, sys

sys.path.insert(0, {root!r})
import rkmon

rkmon.ROOT = pathlib.Path(__file__).resolve().parent
sys.exit(rkmon.main(sys.argv[1:]))
"""
# A process that runs selftest.run over the stand-in in argv[2], this package's folder argv[1], argv[3] modules at a
# time; "nohup" after them: with SIGHUP ignored, as nohup ignores it.
RUNNER = """import signal, sys

sys.path.insert(0, sys.argv[1])
from rkmon_lib import selftest

if sys.argv[4:] == ["nohup"]:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
selftest.run(sys.argv[2], capture=True, timeout=300, jobs=int(sys.argv[3]))
"""
LEFT = (
    "FAILED: the tests left 2 entries in their temporary folder (TMPDIR); a test removes what it makes there:\n"
    "  left-folder/\n"
    "  left.txt\n"
)
IN_HOME = (
    "FAILED: the tests left 2 entries in their home folder (HOME); a test that writes in a home folder makes one of "
    "its own:\n"
    "  .cache/\n"
    "  .toolrc\n"
)
IN_MONITOR_HOME = (
    "FAILED: the tests left 2 entries in their monitor home (RKMON_HOME); a test that needs a monitor home makes one "
    "of its own:\n"
    "  config.json\n"
    "  state/\n"
)
# What the stand-in's "litter" adds to the package's folder, as the run names it.
LITTER = ("notes.txt", "state/", "tests/__pycache__/", "unread/")
ADDED = (
    "FAILED: the tests added 4 entries to the package's folder; a test writes only in folders of its own:\n"
    + "".join(f"  {name}\n" for name in LITTER)
)
STAYS = (
    "FAILED: the tests left their temporary folder, which could not be removed (a process they started may still "
    "write there): "
)
# The stand-in's modules, as a run split by module names them.
MODULES = ["test_00_no_github", "test_other", "test_standin"]


def by_module(found, module="test_standin"):
    """What a run split by module names, each entry FOUND names (a tuple, or the lines of a check) after MODULE."""
    if isinstance(found, tuple):
        return tuple(f"{module}: {name}" for name in found)
    return found.replace("\n  ", f"\n  {module}: ")


def ends(output: str):
    """The tests a run's output says it ran and its result, as its last lines give them (None: no such lines)."""
    m = re.search(r"^-{70}\nRan (\d+) tests? in \d+\.\d{3}s\n\n(.+)\n\Z", output, re.M)
    return m and m.groups()


def reports(output: str) -> list:
    """Each failure's and error's report in a run's output in one process, sorted: its heading and its traceback."""
    found = re.findall(
        r"^={70}\n((?:FAIL|ERROR): .*?)(?=^={70}$|^-{70}\nRan )", output, re.M | re.S
    )
    return sorted(found)


class TestSelftest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        # The stand-in package, and the folder outside it where it writes what it sees (STANDIN_OUT).
        self.pkg, self.out = pathlib.Path(t.name) / "pkg", pathlib.Path(t.name) / "out"
        tests = self.pkg / "tests"
        tests.mkdir(parents=True)
        self.out.mkdir()
        for name, text in (
            ("test_00_no_github.py", STANDIN_GUARD),
            ("test_other.py", STANDIN_OTHER),
            ("test_standin.py", STANDIN),
        ):
            (tests / name).write_text(text, encoding="utf-8")
        p = mock.patch.dict(os.environ, STANDIN_OUT=str(self.out))
        p.start()
        self.addCleanup(p.stop)

    def suite(self, does="", timeout=300, names=(), jobs=1, **env):
        env = dict(os.environ, STANDIN=does, PYTHONDONTWRITEBYTECODE="1", **env)
        return selftest.run(
            self.pkg, env, capture=True, timeout=timeout, names=names, jobs=jobs
        )

    def more(self):
        """The stand-in's fourth module (STANDIN_MORE)."""
        (self.pkg / "tests" / "test_more.py").write_text(STANDIN_MORE, "utf-8")

    def launcher(self):
        """The stand-in's rkmon.py (LAUNCHER)."""
        (self.pkg / "rkmon.py").write_text(LAUNCHER.format(root=str(ROOT)), "utf-8")

    def read_only(self):
        """The stand-in's folder read-only, as the image's package folder is (docker/Dockerfile: chmod -R a-w)."""
        for p in [*self.pkg.rglob("*"), self.pkg]:
            p.chmod(p.stat().st_mode & ~0o222)

    def seen(self):
        """The TMPDIR the stand-in's test ran with, and what was in it when the test started."""
        text = (self.out / "seen.txt").read_text(encoding="utf-8")
        tmp, *there = text.split("\n")
        return pathlib.Path(tmp), there

    def seen_monitor_home(self):
        """The monitor home (RKMON_HOME) the stand-in's test ran with, and what was in it when the test started (None:
        no such folder)."""
        monitor, there = json.loads((self.out / "monitor.json").read_text("utf-8"))
        return pathlib.Path(monitor), there

    def seen_home(self):
        """The home folder (HOME) the stand-in's test ran with, what was in it when the test started, and the XDG_
        names it had."""
        home, there, xdg = json.loads((self.out / "home.json").read_text("utf-8"))
        return pathlib.Path(home), there, xdg

    def seen_env(self) -> dict:
        """The names of KINDS (RKMON_HOME aside) and PYTHONDONTWRITEBYTECODE the stand-in's test ran with."""
        return json.loads((self.out / "env.json").read_text("utf-8"))

    def ran(self) -> list:
        """What the stand-in's modules and tests wrote to ran.txt, in order."""
        return (self.out / "ran.txt").read_text("utf-8").split("\n")[:-1]

    @staticmethod
    def environment(**names) -> dict:
        """This process's environment with no name of KINDS and no PYTHONDONTWRITEBYTECODE, then NAMES."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(KINDS) and k != "PYTHONDONTWRITEBYTECODE"
        }
        return dict(env, **names)

    def beats(self) -> bool:
        """Whether the stand-in's writer still writes: its beat grows within a few tenths of a second."""
        beat = self.out / "beat"
        if not beat.exists():  # no writer started
            return False
        before = beat.stat().st_size
        time.sleep(0.3)
        return beat.stat().st_size != before

    def stop_writer(self):
        """The stand-in's writer, when it still runs, stops before the test's folder goes."""
        (self.out / "stop").touch()
        until = time.monotonic() + 10
        while time.monotonic() < until and self.beats():
            pass

    def running(self, *how, jobs=1) -> subprocess.Popen:
        """A process running the stand-in's suite through selftest.run (RUNNER; HOW: "nohup" or nothing), JOBS modules
        at a time, its test's writer started and the test asleep."""
        beat = self.out / "beat"
        beat.unlink(missing_ok=True)
        p = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                RUNNER,
                str(ROOT),
                str(self.pkg),
                str(jobs),
                *how,
            ],
            env=dict(os.environ, STANDIN="writer sleep"),
            stdout=subprocess.DEVNULL,
        )
        self.addCleanup(p.wait, 60)
        self.addCleanup(p.terminate)
        until = time.monotonic() + 60
        while not beat.exists() and time.monotonic() < until:
            time.sleep(0.05)
        self.assertTrue(beat.exists(), "the stand-in's writer never started")
        return p

    def units(self, does: str):
        """tests/container/image.py's units stage on this machine, STANDIN set to DOES and the image's switch on, as the
        container has it: its interpreter this one, and its package folder the stand-in, whose rkmon.py (`launcher`)
        runs this package's, its package folder pointed at the stand-in. The stage's output, and the folder that stands
        for the container's own temporary folder.
        """
        script = image.UNITS.replace("/opt/rkmon/venv/bin/python", sys.executable)
        script = script.replace("/opt/rocke_monitor", str(self.pkg))
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = subprocess.Popen(
            ["sh", "-c", script],
            env=dict(os.environ, STANDIN=does, TMPDIR=t.name, RKMON_CONTAINER="1"),
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out = p.communicate(timeout=300)[0]
        finally:
            # Past the timeout, what is left of the stage is stopped, as `timeout` stops a command (SIGTERM).
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)
            p.wait()
        return out, pathlib.Path(t.name)

    def test_the_suite_runs_in_a_new_home_folder_of_its_own_gone_after(self):
        # Its TMPDIR and its monitor home in there; never the home folder, the XDG folders or the monitor home its
        # caller's environment names (the image names the monitor's).
        callers = self.out / "callers-home"
        callers.mkdir()
        r = self.suite(
            HOME=str(callers),
            XDG_CONFIG_HOME=str(callers / ".config"),
            XDG_CACHE_HOME=str(callers / ".cache"),
            RKMON_HOME=str(callers),
        )
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home, r.added, r.stays),
            (0, (), (), (), (), ""),
            r.output,
        )
        self.assertTrue(r.output.endswith("\nOK\n"), r.output)
        (tmp, there), (monitor, at_monitor) = self.seen(), self.seen_monitor_home()
        home, at_home, xdg = self.seen_home()
        self.assertEqual((there, at_monitor), ([], []))  # new and empty
        # The home folder holds them and nothing else, and XDG's folders where a home folder has them.
        self.assertEqual(
            (tmp.parent, monitor.parent, at_home), (home, home, ["monitor", "tmp"])
        )
        homes = {
            "XDG_CONFIG_HOME": f"{home}/.config",
            "XDG_CACHE_HOME": f"{home}/.cache",
            "XDG_DATA_HOME": f"{home}/.local/share",
            "XDG_STATE_HOME": f"{home}/.local/state",
        }
        self.assertEqual({k: xdg.get(k) for k in homes}, homes)
        self.assertEqual(home.parent, pathlib.Path(tempfile.gettempdir()))
        self.assertFalse(home.exists())  # gone after the run
        # Every module, the guard first.
        self.assertEqual(self.ran(), ["guard", "other", "guard test", "standin"])

    def test_the_suite_runs_with_no_git_setting_no_container_switch_and_no_bytecode(
        self,
    ):
        # A caller's git hook or `git rebase --exec` names its repository: the suite's git commands never see it, nor
        # any other git setting of the caller's. The image's switch goes too: the tests run in the image as on a host.
        located = (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_PREFIX",
        )
        env = self.environment(
            **dict.fromkeys(located, "/elsewhere"),
            GIT_AUTHOR_NAME="theirs",
            RKMON_CONTAINER="1",
        )
        r = selftest.run(self.pkg, env, capture=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.output)
        self.assertEqual(self.seen_env(), {"PYTHONDONTWRITEBYTECODE": "1"})

    def test_no_login_key_or_gateway_reaches_a_test_the_test_tools_places_do(self):
        # The logins' folders (the image names them), a key and a gateway never reach a test; the places of the tools
        # some tests run do.
        tools = {
            "RKMON_VENV": "/opt/venv",
            "RKMON_TORCH_PY": "/opt/torch/bin/python",
            "RKMON_BLACK": "/opt/black/bin/black",
        }
        env = self.environment(
            GH_CONFIG_DIR="/s/gh",
            ATLASSIAN_CONFIG_DIR="/s/atlassian",
            CLAUDE_CONFIG_DIR="/s/claude",
            ANTHROPIC_API_KEY="k",
            ANTHROPIC_BASE_URL="https://example.com",
            **tools,
        )
        r = selftest.run(self.pkg, env, capture=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.output)
        self.assertEqual(self.seen_env(), {**tools, "PYTHONDONTWRITEBYTECODE": "1"})

    def test_no_github_token_model_server_or_host_name_reaches_a_test(self):
        # GitHub's tokens, which gh reads beside its own GH_ names, the Ollama model server's address and the Windows
        # host's name never reach a test.
        env = self.environment(
            GITHUB_TOKEN="t",
            GITHUB_ENTERPRISE_TOKEN="t",
            OLLAMA_HOST="example.com:11434",
            OLLAMA_BASE_URL="https://example.com",
            WINHOST="<host>",
        )
        r = selftest.run(self.pkg, env, capture=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.output)
        self.assertEqual(self.seen_env(), {"PYTHONDONTWRITEBYTECODE": "1"})

    def test_the_tests_named_run_after_the_guard_and_what_they_leave_fails_the_run(
        self,
    ):
        r = self.suite("leave", names=("tests.test_00_no_github", "tests.test_standin"))
        self.assertEqual(
            (r.returncode, r.left), (1, ("left-folder/", "left.txt")), r.output
        )
        self.assertTrue(r.output.endswith("\nOK\n" + LEFT), r.output)
        self.assertFalse(self.seen()[0].exists())
        # The guard first, its test once though it is named too, then the test named; the other module never runs.
        self.assertEqual(self.ran(), ["guard", "guard test", "standin"])

    def test_a_test_that_leaves_a_file_fails_the_run_which_names_each_entry(self):
        r = self.suite("leave")
        self.assertEqual(
            (r.returncode, r.left), (1, ("left-folder/", "left.txt")), r.output
        )
        self.assertTrue(r.output.endswith("\nOK\n" + LEFT), r.output)
        self.assertFalse(self.seen()[0].exists())  # gone, its read-only folder too

    def test_a_test_that_writes_in_its_home_folder_fails_the_run_which_names_each_entry(
        self,
    ):
        # Its own folders aside, which are named on their own.
        r = self.suite("home")
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home),
            (1, (), (".cache/", ".toolrc"), ()),
            r.output,
        )
        self.assertTrue(r.output.endswith("\nOK\n" + IN_HOME), r.output)
        self.assertFalse(self.seen_home()[0].exists())

    def test_a_test_that_writes_in_its_monitor_home_fails_the_run_which_names_each_entry(
        self,
    ):
        r = self.suite("monitor")
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home),
            (1, (), (), ("config.json", "state/")),
            r.output,
        )
        self.assertTrue(r.output.endswith("\nOK\n" + IN_MONITOR_HOME), r.output)
        self.assertFalse(self.seen_monitor_home()[0].exists())

    def test_what_a_test_adds_to_the_package_s_folder_fails_the_run_which_names_it(
        self,
    ):
        # Ignored files too: the package is a git repository that ignores them all; and an entry in its .git. A new
        # folder is named, not what is in it, one no one can read too.
        env = self.environment()
        git = ["git", "-C", str(self.pkg)]
        subprocess.run([*git, "init", "-q"], env=env, check=True)
        (self.pkg / ".gitignore").write_text(
            "notes.txt\nstate/\n__pycache__/\nunread/\n", encoding="utf-8"
        )
        r = self.suite("litter git")
        named = (".git/litter", *LITTER)
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home, r.added),
            (1, (), (), (), named),
            r.output,
        )
        self.assertTrue(
            r.output.endswith(
                "\nOK\nFAILED: the tests added 5 entries to the package's folder; a test writes only in folders of its "
                "own:\n" + "".join(f"  {name}\n" for name in named)
            ),
            r.output,
        )
        ignored = subprocess.run(
            [*git, "check-ignore", *LITTER], env=env, capture_output=True, text=True
        )
        self.assertEqual(ignored.stdout.split(), list(LITTER))

    def test_the_run_s_own_folder_is_no_entry_of_the_package_s_folder(self):
        # When this process's temporary folder is in the package's folder, the run's own folder is made there: what the
        # tests leave in it is named as left in their home folder, their TMPDIR or their monitor home, not as added to
        # the package's folder.
        inner = self.pkg / "tmp"
        inner.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(inner)):
            r = self.suite("leave home monitor")
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home, r.added),
            (
                1,
                ("left-folder/", "left.txt"),
                (".cache/", ".toolrc"),
                ("config.json", "state/"),
                (),
            ),
            r.output,
        )
        self.assertTrue(
            r.output.endswith("\nOK\n" + LEFT + IN_HOME + IN_MONITOR_HOME), r.output
        )
        self.assertEqual(self.seen()[0].parent.parent, inner)

    def test_a_failing_test_keeps_its_failure_and_what_it_left_is_named_too(self):
        r = self.suite("leave fail")
        self.assertEqual(r.returncode, 1)
        self.assertIn("\nFAILED (failures=1)\n", r.output)
        self.assertTrue(r.output.endswith(LEFT), r.output)
        self.assertFalse(self.seen()[0].exists())

    def test_a_process_the_suite_leaves_running_ends_before_the_folder_is_read(self):
        # The writer holds the suite's stdout and writes in the folder: it ends with the suite, what it wrote is named,
        # and the folder goes.
        self.addCleanup(self.stop_writer)
        r = self.suite("writer", timeout=60)
        self.assertFalse(self.beats())
        self.assertEqual((r.returncode, r.stays), (1, ""), r.output)
        self.assertTrue(r.left and set(r.left) <= {"w0", "w1", "w2"}, r.left)
        self.assertIn("\nOK\nFAILED: the tests left ", r.output)
        self.assertTrue(
            r.output.endswith("".join(f"  {name}\n" for name in r.left)), r.output
        )
        self.assertFalse(self.seen()[0].exists())

    def test_past_its_timeout_the_suite_ends_with_what_it_left_running(self):
        # The folder goes too.
        self.addCleanup(self.stop_writer)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.suite("writer sleep", timeout=5)
        self.assertFalse(self.beats())
        self.assertFalse(self.seen()[0].exists())

    def test_sigterm_or_sighup_stop_the_run_its_processes_ended_and_its_folder_gone(
        self,
    ):
        # As `timeout`, `kill` or a terminal that closes stops it: the run exits 128 + the signal's number.
        self.addCleanup(self.stop_writer)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                p = self.running()
                p.send_signal(signum)
                self.assertEqual(p.wait(60), 128 + signum)
                self.assertFalse(self.beats())
                self.assertFalse(self.seen()[0].exists())

    def test_a_sighup_its_caller_ignores_does_not_stop_the_run(self):
        # As under nohup.
        self.addCleanup(self.stop_writer)
        p = self.running("nohup")
        p.send_signal(signal.SIGHUP)
        with self.assertRaises(subprocess.TimeoutExpired):
            p.wait(1)
        self.assertTrue(self.beats())
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(60), 128 + signal.SIGTERM)
        self.assertFalse(self.beats())
        self.assertFalse(self.seen()[0].exists())

    def test_the_tests_take_no_input_of_their_caller_s(self):
        """The suite's tests run with no input (/dev/null), in one process and split by module, whatever input their
        caller has (here a pipe): a test that reads its input never waits for the caller's, which may never end.
        """
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                (self.out / "stdin.txt").unlink(missing_ok=True)
                p = subprocess.run(
                    [sys.executable, "-B", "-c", RUNNER, str(ROOT), str(self.pkg)]
                    + [str(jobs)],
                    env=dict(os.environ, STANDIN=""),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    timeout=300,
                )
                self.assertEqual(p.returncode, 0)
                self.assertEqual(
                    (self.out / "stdin.txt").read_text(encoding="utf-8"), "/dev/null"
                )

    def test_a_run_in_another_thread_leaves_the_signals_as_they_are(self):
        # Only the main thread can set a handler.
        handlers = [signal.getsignal(s) for s in selftest.STOPS]
        runs = []
        t = threading.Thread(target=lambda: runs.append(self.suite()))
        t.start()
        t.join(300)
        self.assertEqual([r.returncode for r in runs], [0])
        self.assertEqual([signal.getsignal(s) for s in selftest.STOPS], handlers)

    def test_a_folder_that_cannot_be_removed_fails_the_run_which_names_it(self):
        def rmtree(path, ignore_errors=False, onerror=None, *, onexc=None, **kw):
            # The removal fails as it does while a process the suite started in a session of its own writes there.
            try:
                raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), path)
            except OSError as exc:
                if onexc is not None:
                    onexc(os.rmdir, path, exc)
                elif onerror is not None:
                    onerror(os.rmdir, path, sys.exc_info())
                elif not ignore_errors:
                    raise

        # The run's own folder: the suite's home folder, or a split run's folder of its modules' home folders.
        for jobs, up in ((1, 1), (3, 2)):
            with self.subTest(jobs=jobs):
                with mock.patch("shutil.rmtree", rmtree):
                    r = self.suite(jobs=jobs)
                own = self.seen()[0].parents[up - 1]
                # That folder, and no other, is the one this test removes.
                self.assertTrue(own.name.startswith(selftest.FOLDER_PREFIX), own)
                self.addCleanup(shutil.rmtree, own, ignore_errors=True)
                self.assertTrue(own.is_dir())
                self.assertEqual(
                    (r.returncode, r.left, r.stays), (1, (), str(own)), r.output
                )
                self.assertTrue(r.output.endswith(f"\nOK\n{STAYS}{own}\n"), r.output)

    def test_rk_selftest_fails_on_what_a_test_leaves_or_adds(self):
        # rk selftest over the stand-in (LAUNCHER): the whole suite, then the test named; never with the monitor home
        # its caller's environment names, nor the image's switch.
        self.launcher()
        callers = self.out / "callers-home"
        callers.mkdir()
        whole = ["guard", "other", "guard test", "standin"]
        for does, names, rc, ran, named in (
            ("", [], 0, whole, ""),
            ("leave", [], 1, whole, LEFT),
            (
                "leave",
                ["tests.test_standin.T.test_it"],
                1,
                ["guard", "guard test", "standin"],
                LEFT,
            ),
            ("home monitor litter", [], 1, whole, IN_HOME + IN_MONITOR_HOME + ADDED),
        ):
            (self.out / "ran.txt").unlink(missing_ok=True)
            p = subprocess.run(
                [sys.executable, "-B", str(self.pkg / "rkmon.py"), "selftest", *names],
                env=self.environment(
                    STANDIN=does,
                    GIT_DIR="/elsewhere",
                    RKMON_HOME=str(callers),
                    RKMON_CONTAINER="1",
                ),
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(p.returncode, rc, p.stdout + p.stderr)
            self.assertTrue(p.stderr.endswith("\nOK\n" + named), p.stderr)
            (tmp, _), (monitor, at_monitor) = self.seen(), self.seen_monitor_home()
            home = self.seen_home()[0]
            self.assertEqual((tmp.parent, monitor.parent, at_monitor), (home, home, []))
            self.assertFalse(home.exists())
            self.assertEqual(self.seen_env(), {"PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(self.ran(), ran)

    def test_the_package_build_fails_on_what_a_test_leaves_or_adds(self):
        # The suite it runs has no git repository a caller names and writes no bytecode, as `rk selftest`'s; bytecode
        # a test's own child writes in the built folder all the same fails it too.
        printed = io.StringIO()
        with mock.patch.dict(
            os.environ,
            self.environment(STANDIN="leave home monitor litter", GIT_DIR="/elsewhere"),
            clear=True,
        ), contextlib.redirect_stdout(printed), self.assertRaises(SystemExit) as stop:
            make_package.run_tests(self.pkg)
        self.assertTrue(
            str(stop.exception).startswith(
                "the package's tests failed in the built folder:\n"
            )
        )
        self.assertTrue(
            str(stop.exception).endswith(LEFT + IN_HOME + IN_MONITOR_HOME + ADDED),
            str(stop.exception),
        )
        self.assertIn("tests: ", printed.getvalue())
        self.assertFalse(self.seen()[0].parent.exists())
        self.assertEqual(self.seen_env(), {"PYTHONDONTWRITEBYTECODE": "1"})

    def test_the_image_s_test_runs_rk_selftest_in_a_read_only_package_folder(self):
        # Read-only, as the image's package folder is; in the environment rk selftest gives the suite everywhere.
        self.launcher()
        self.read_only()
        out, container_tmp = self.units("leave home monitor")
        self.assertTrue(out.startswith("units=1\nRan 2 tests in "), out)
        self.assertTrue(out.endswith("\nOK\n" + LEFT + IN_HOME + IN_MONITOR_HOME), out)
        tmp, home = self.seen()[0], self.seen_home()[0]
        # The run's own folder, the suite's home folder, in the container's temporary folder.
        self.assertEqual((tmp.parent, home.parent), (home, container_tmp))
        self.assertFalse(home.exists())
        # The image's switch stays with the stage: the tests run as on a host.
        self.assertNotIn("RKMON_CONTAINER", self.seen_env())

    def test_the_image_s_test_names_what_a_test_adds_to_the_package_s_folder(self):
        self.launcher()
        out, _ = self.units("litter")
        self.assertTrue(out.startswith("units=1\nRan 2 tests in "), out)
        self.assertTrue(out.endswith("\nOK\n" + ADDED), out)

    def test_a_split_run_runs_each_module_in_a_process_and_a_home_of_its_own(self):
        # Every module's process loads the guard first, whose tests run once, in its own module's process. Each home
        # folder is numbered in the run's own folder, its TMPDIR in it, and all of them go after the run.
        self.more()
        r = self.suite(jobs=3)
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home, r.added, r.stays),
            (0, (), (), (), (), ""),
            r.output,
        )
        self.assertEqual(
            sorted(self.ran()), ["guard"] * 4 + ["guard test", "other", "standin"]
        )
        more = json.loads((self.out / "more.json").read_text("utf-8"))
        home, more_home = self.seen_home()[0], pathlib.Path(more["home"])
        self.assertTrue(more["guard first"])
        self.assertEqual(
            (more_home.parent, more["tmp"]), (home.parent, f"{more_home}/tmp")
        )
        self.assertNotEqual(more_home, home)
        self.assertTrue(home.name.isdigit() and more_home.name.isdigit())
        self.assertEqual(home.parent.parent, pathlib.Path(tempfile.gettempdir()))
        self.assertFalse(home.parent.exists())

    def test_a_split_run_ends_as_the_suite_in_one_process_does(self):
        # The same tests and counts, the same failures and errors with their tracebacks (two modules': "more-fail" fails
        # the stand-in's test too), the same exit code; one summary, the whole run's.
        self.more()
        for does, result in (
            ("", "OK (skipped=1)"),
            ("more-fail more-error", "FAILED (failures=2, errors=1, skipped=1)"),
        ):
            with self.subTest(does=does):
                one, split = self.suite(does), self.suite(does, jobs=3)
                self.assertEqual(ends(one.output), ("6", result), one.output)
                self.assertEqual(ends(split.output), ("6", result), split.output)
                self.assertEqual(len(re.findall("^Ran ", split.output, re.M)), 1)
                self.assertEqual(split.returncode, one.returncode)
                self.assertEqual(one.returncode, 1 if does else 0)
                found = reports(one.output)
                self.assertEqual(len(found), 3 if does else 0)
                for report in found:
                    self.assertIn(report, split.output)
                self.assertEqual(
                    sorted(re.findall("^(?:FAIL|ERROR): .*", split.output, re.M)),
                    sorted(re.findall("^(?:FAIL|ERROR): .*", one.output, re.M)),
                )
        self.assertIn("RuntimeError: the error the run reports\n", split.output)

    def test_a_split_run_ends_so_when_unittest_colors_each_module_s_result(self):
        # FORCE_COLOR, with no NO_COLOR or PYTHON_COLORS that turns the colors off: each module's last lines go all the
        # same, and the whole run's come at its end.
        self.more()
        colors = dict(FORCE_COLOR="1", NO_COLOR="", PYTHON_COLORS="")
        r = self.suite("more-fail", jobs=3, **colors)
        self.assertEqual(len(re.findall("^Ran ", r.output, re.M)), 1, r.output)
        self.assertEqual(
            ends(r.output), ("6", "FAILED (failures=2, skipped=1)"), r.output
        )
        if sys.version_info >= (3, 14):  # unittest writes in color
            self.assertIn("\x1b[", r.output)

    def test_a_split_run_s_modules_are_those_the_discovery_runs(self):
        # By unittest's pattern and its rule for a module's name: no helper, and no file whose name no module has.
        for name in ("helper.py", "test-dash.py", "test_notes.txt"):
            (self.pkg / "tests" / name).write_text("x = 1\n", "utf-8")
        self.assertEqual(selftest.modules(self.pkg), [f"{n}.py" for n in MODULES])

    def test_a_module_whose_process_ends_before_its_result_fails_a_split_run(self):
        # Counted as an error and named; the run's exit code is the module's.
        r = self.suite("exit", jobs=3)
        self.assertEqual(r.returncode, 3, r.output)
        self.assertIn(
            "FAILED: tests/test_standin.py's process ended with exit code 3 before its result",
            r.output.splitlines(),
        )
        self.assertEqual(ends(r.output), ("1", "FAILED (errors=1)"), r.output)

    def test_what_a_split_run_s_modules_leave_or_add_fails_it_named_after_each(self):
        r = self.suite("leave home monitor litter", jobs=3)
        self.assertEqual(
            (r.returncode, r.left, r.in_home, r.in_monitor_home, r.added),
            (
                1,
                by_module(("left-folder/", "left.txt")),
                by_module((".cache/", ".toolrc")),
                by_module(("config.json", "state/")),
                LITTER,
            ),
            r.output,
        )
        self.assertTrue(
            r.output.endswith(
                "\nOK\n" + by_module(LEFT + IN_HOME + IN_MONITOR_HOME) + ADDED
            ),
            r.output,
        )
        self.assertFalse(self.seen()[0].parent.parent.exists())  # the run's own folder

    def test_a_split_run_s_own_folder_is_no_entry_of_the_package_s_folder(self):
        inner = self.pkg / "tmp"
        inner.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(inner)):
            r = self.suite("leave", jobs=3)
        self.assertEqual(
            (r.returncode, r.left, r.added),
            (1, by_module(("left-folder/", "left.txt")), ()),
            r.output,
        )
        self.assertEqual(self.seen()[0].parent.parent.parent, inner)

    def test_a_split_run_starts_the_largest_module_first(self):
        # The modules in the discovery's order, then the largest file first: a run starts their processes so.
        self.more()
        files = [f"{name}.py" for name in sorted([*MODULES, "test_more"])]
        self.assertEqual(selftest.modules(self.pkg), files)
        largest = [
            "test_standin.py",
            "test_more.py",
            "test_00_no_github.py",
            "test_other.py",
        ]
        sizes = [(self.pkg / "tests" / name).stat().st_size for name in largest]
        self.assertEqual(sizes, sorted(set(sizes), reverse=True))
        self.assertEqual(selftest.order(self.pkg, files), largest)
        with mock.patch.object(selftest, "_Module", wraps=selftest._Module) as module:
            r = self.suite(jobs=2)
        self.assertEqual(r.returncode, 0, r.output)
        self.assertEqual([c.args[4] for c in module.call_args_list], largest)
        # Files of one size keep the order they are given in.
        for name in ("tie_a.py", "tie_b.py"):
            (self.pkg / "tests" / name).write_text("x", "utf-8")
        for names in (["tie_a.py", "tie_b.py"], ["tie_b.py", "tie_a.py"]):
            self.assertEqual(selftest.order(self.pkg, names), names)

    def test_a_split_run_refuses_a_package_in_the_suite_s_folder(self):
        # Its tests, which the discovery reads, would not run.
        (self.pkg / "tests" / "sub").mkdir()
        (self.pkg / "tests" / "sub" / "__init__.py").write_text("", "utf-8")
        with self.assertRaisesRegex(selftest.CannotSplit, "^tests/sub is a package"):
            self.suite(jobs=2)
        self.assertFalse((self.out / "ran.txt").exists())

    def test_rk_selftest_refuses_jobs_below_1_and_a_suite_it_cannot_split(self):
        # In plain words and with no traceback, as the package refuses a count, before any test runs.
        self.launcher()
        (self.pkg / "tests" / "sub").mkdir()
        (self.pkg / "tests" / "sub" / "__init__.py").write_text("", "utf-8")
        below = "REFUSED: --jobs must be 1 or more\n"
        package = (
            "REFUSED: tests/sub is a package: the discovery reads its tests, and a run split by module would not; "
            "run rk selftest without --jobs\n"
        )
        for jobs, said in (("0", below), ("-3", below), ("2", package)):
            with self.subTest(jobs=jobs):
                p = subprocess.run(
                    [sys.executable, "-B", str(self.pkg / "rkmon.py"), "selftest"]
                    + ["--jobs", jobs],
                    env=self.environment(),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                self.assertEqual((p.returncode, p.stdout, p.stderr), (4, said, ""))
        self.assertFalse((self.out / "ran.txt").exists())

    def test_a_process_a_module_leaves_running_ends_when_the_module_ends(self):
        # Before the module's folders are read: what it wrote there is named after the module, and the run's folder
        # goes.
        self.addCleanup(self.stop_writer)
        r = self.suite("writer", timeout=60, jobs=2)
        self.assertFalse(self.beats())
        self.assertEqual((r.returncode, r.stays), (1, ""), r.output)
        written = set(by_module(("w0", "w1", "w2")))
        self.assertTrue(r.left and set(r.left) <= written, r.left)
        self.assertFalse(self.seen()[0].parent.parent.exists())

    def test_past_its_timeout_a_split_run_ends_with_what_it_left_running(self):
        # The run's folder goes too; the timeout names the command that ran past it, the stand-in's module's.
        self.addCleanup(self.stop_writer)
        with self.assertRaises(subprocess.TimeoutExpired) as stop:
            self.suite("writer sleep", timeout=5, jobs=2)
        self.assertFalse(self.beats())
        self.assertFalse(self.seen()[0].parent.parent.exists())
        cmd = stop.exception.cmd
        self.assertEqual(cmd[:4], [sys.executable, "-B", "-c", selftest.UNIT])
        self.assertEqual(cmd[6], "test_standin.py")

    def test_sigterm_stops_a_split_run_its_processes_ended_and_its_folder_gone(self):
        self.addCleanup(self.stop_writer)
        p = self.running(jobs=2)
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(60), 128 + signal.SIGTERM)
        self.assertFalse(self.beats())
        self.assertFalse(self.seen()[0].parent.parent.exists())

    def test_rk_selftest_jobs_splits_the_suite_and_writes_nothing_in_the_caller_s_home(
        self,
    ):
        # The tests named run in one process. The caller's home folder and XDG's folders in it stay empty.
        self.launcher()
        callers = self.out / "callers-home"
        callers.mkdir()
        xdg = {name: str(callers / where) for name, where in selftest.XDG_HOMES.items()}
        for does, names, rc, named in (
            ("", [], 0, ""),
            ("leave", [], 1, by_module(LEFT)),
            ("leave", ["tests.test_standin"], 1, LEFT),
        ):
            p = subprocess.run(
                [sys.executable, "-B", str(self.pkg / "rkmon.py"), "selftest"]
                + ["--jobs", "2", *names],
                env=self.environment(STANDIN=does, HOME=str(callers), **xdg),
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(p.returncode, rc, p.stdout + p.stderr)
            self.assertTrue(p.stderr.endswith("\nOK\n" + named), p.stderr)
        self.assertEqual(list(callers.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

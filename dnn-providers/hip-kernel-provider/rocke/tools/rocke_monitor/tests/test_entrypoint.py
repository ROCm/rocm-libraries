# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's own pieces, outside a container: the entry point's health check, its usage for a mode it does not
have (setup's too), its wait while the work volume is not set up, log cuts, the parts it runs (none, and no tick,
while the configuration cannot be read), supervised orchestrator and tick loop, and their stop; the deployment's user
ids (docker/rk_ids.py: what it refuses, the commands that map them, the volumes' files that follow), and the logins'
walk's one-off container, whose own temporary folders alone get the start's modes (docker/rk_layout.py --system-tmp);
the recipe (the Python headers, the HEALTHCHECK), the build context's exclusions, the atlassian installer's stage; the
orchestrator's process groups, its stale sessions and its review driver; and the doctor's answers in the container (no
services' rows, the Claude login, the fixes as the operator types them).
"""

import ast
import asyncio
import collections
import contextlib
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
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import build_context  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    config,
    doctor,
    features,
    orchestrator,
    program,
    reach,
)

DOCKER = ROOT / "docker"
ENTRY = DOCKER / "entrypoint.sh"


def entry() -> str:
    return ENTRY.read_text(encoding="utf-8")


def function(name: str) -> str:
    """One shell function of the entry point, as written: a one-line one, or its first line to its closing brace at
    column 0."""
    text = entry()
    start = text.index(f"\n{name}() {{") + 1
    first = text[start : text.index("\n", start) + 1]
    if re.search(r";\s*\}\s*(#.*)?$", first.rstrip("\n")):
        return first
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def fake_rk(d: pathlib.Path, body: str) -> pathlib.Path:
    rk = d / "rk"
    rk.write_text("#!/bin/sh\n" + body)
    rk.chmod(0o755)
    return rk


class TestHealth(unittest.TestCase):
    """`entrypoint.sh health`, the image's HEALTHCHECK."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.logs = self.d / "logs"
        self.logs.mkdir()
        self.home = self.d / "monitor"
        self.home.mkdir()

    def run_health(self, stopped=False, **env) -> subprocess.CompletedProcess:
        """The check, with RKMON_LOGS and ENV, and without this environment's RKMON_HOME."""
        # The entry point's rk is /opt/rocke_monitor/rk: a copy of the script names a stand-in instead. Its
        # `provision --check` says the work volume is set up once the home holds the test's flag, and refuses, as rk
        # does when the internal state cannot be read, while the home holds the file `refuses`.
        rk = fake_rk(
            self.d,
            '[ "$1 $2" = "provision --check" ] && { [ -f "$RKMON_HOME/refuses" ] && '
            '{ cat "$RKMON_HOME/refuses"; exit 4; }; [ -f "$RKMON_HOME/set-up" ] && exit 0; '
            'echo "not set up yet"; exit 1; }\n'
            f'[ "$1" = stopped ] && exit {0 if stopped else 1}\nexit 0\n',
        )
        script = self.d / "entrypoint.sh"
        script.write_text(entry().replace('RK="$PKG/rk"', f'RK="{rk}"'))
        outside = {k: v for k, v in os.environ.items() if k != "RKMON_HOME"}
        return subprocess.run(
            ["sh", str(script), "health"],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(outside, RKMON_LOGS=str(self.logs), **env),
        )

    def health(self, stopped=False) -> tuple[int, str]:
        p = self.run_health(stopped, RKMON_HOME=str(self.home))
        return p.returncode, p.stdout

    def test_not_set_up_stopped_and_crash_loops(self):
        """Set up as rk provision --check says: the entry point looks for no file of its own in the home."""
        (self.home / "config.json").write_text("{}")
        self.assertEqual(self.health(), (0, "healthy: not set up yet\n"))
        (self.home / "set-up").write_text("")
        self.assertEqual(self.health(), (0, "healthy\n"))
        now = int(time.time())
        lines = [
            f"{now - 7200} the orchestrator (exit 1)\n"
        ] * 9  # long ago: not counted
        lines += [f"{now - 60 * i} the orchestrator (exit 1)\n" for i in range(5)]
        (self.logs / "restarts").write_text("".join(lines))
        rc, out = self.health()
        self.assertEqual(rc, 1)
        self.assertIn("unhealthy: 5 restarts in the last 30 minutes", out)
        self.assertIn("the orchestrator (exit 1)", out)
        self.assertEqual(
            self.health(stopped=True), (0, "healthy: stopped (rk start resumes)\n")
        )

    def test_unhealthy_while_whether_it_is_set_up_cannot_be_told(self):
        """Only rk provision --check's exit 1 means not set up yet. Any other exit, such as its refusal while the
        monitor's internal state cannot be read, is unhealthy, with its first line as written (a backslash in it is no
        escape); one with no line says its exit.
        """
        refusal = r"REFUSED: the monitor's internal state cannot be read: not JSON (\t, \n and \c as written)"
        (self.home / "refuses").write_text(f"\n{refusal}\n  fix: something\n")
        self.assertEqual(self.health(), (1, f"unhealthy: {refusal}\n"))
        (self.home / "set-up").write_text("")  # its record cannot be read either way
        self.assertEqual(self.health(), (1, f"unhealthy: {refusal}\n"))
        (self.home / "refuses").write_text("")
        self.assertEqual(
            self.health(),
            (
                1,
                "unhealthy: whether the work volume is set up cannot be told (exit 4)\n",
            ),
        )

    def test_the_home_is_the_one_the_image_names(self):
        """The monitor's home is RKMON_HOME, which the image's environment names: without it the check stops,
        naming it, and looks in no folder of its own."""
        p = self.run_health()
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(p.stdout, "")
        self.assertIn("RKMON_HOME", p.stderr)


class TestTheWritersCli(unittest.TestCase):
    """`entrypoint.sh atlassian ARGS`, the atlassian CLI by hand: its login is the writer's (`rkmon login atlassian`
    runs it as the writer, into the writer's volume), so as any other user it is refused and the CLI never runs.
    `entrypoint.sh rk ARGS` as the writer (`rkmon secret NAME` for a secret the writer keeps) runs the writer's broker,
    docker/rk_write.py, never rk itself.
    """

    def test_the_writer_s_rk_is_its_broker(self):
        if os.geteuid() == 0:
            self.skipTest("as root the entry point maps the users' ids first")
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            pkg, stubs, said = d / "pkg", d / "bin", d / "said"
            (pkg / "docker").mkdir(parents=True)
            stubs.mkdir()
            for path, who in (
                (pkg / "rk", "rk"),
                (pkg / "docker" / "rk_write.py", "the broker"),
            ):
                path.write_text(f'#!/bin/sh\necho "{who}: $*" >> {said}\n')
                path.chmod(0o755)
            script = d / "entrypoint.sh"
            script.write_text(
                entry().replace("PKG=/opt/rocke_monitor", f"PKG={pkg}", 1)
            )
            for user, want in (("rkwrite", "the broker"), ("rkmon", "rk")):
                (stubs / "id").write_text(
                    f'#!/bin/sh\n[ "$*" = -un ] && {{ echo {user}; exit 0; }}\nexec /usr/bin/id "$@"\n'
                )
                (stubs / "id").chmod(0o755)
                said.unlink(missing_ok=True)
                p = subprocess.run(
                    ["sh", str(script), "rk", "secret-set", "teams_webhook"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={
                        "PATH": f"{stubs}:/usr/bin:/bin",
                        "HOME": t,
                        "RKMON_LOGS": str(d / "logs"),
                    },
                )
                with self.subTest(user=user):
                    self.assertEqual(
                        (p.returncode, said.read_text()),
                        (0, f"{want}: secret-set teams_webhook\n"),
                        p.stderr,
                    )

    def test_only_the_writer_runs_it(self):
        if os.geteuid() == 0:
            self.skipTest("as root the entry point maps the users' ids first")
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            ran = d / "ran"
            (d / "bin").mkdir()
            cli = d / "bin" / "atlassian"
            cli.write_text(f"#!/bin/sh\ntouch {ran}\n")
            cli.chmod(0o755)
            p = subprocess.run(
                ["sh", str(ENTRY), "atlassian", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=30,
                env=dict(
                    os.environ,
                    RKMON_LOGS=str(d / "logs"),
                    PATH=f"{d / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
                ),
            )
            self.assertEqual((p.returncode, ran.exists()), (2, False))
        self.assertIn("atlassian: its login is the writer's only", p.stderr)
        text = entry()
        self.assertIn("ATLASSIAN_LOGIN=/secrets-write/atlassian", text)
        self.assertIn("ATLASSIAN_CONFIG_DIR=$ATLASSIAN_LOGIN", text)


class TestModes(unittest.TestCase):
    """`entrypoint.sh MODE`: a mode it does not have is its usage error, exit 2, and runs nothing. setup is one: the
    deploy path sets a deployment up, and the usage names no setup mode.
    """

    def test_setup_reaches_the_usage(self):
        if os.geteuid() == 0:
            self.skipTest("as root the entry point maps the users' ids first")
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            pkg, ran = d / "pkg", d / "ran"
            pkg.mkdir()
            fake_rk(pkg, f'echo "rk: $*" >> {ran}\n')
            script = d / "entrypoint.sh"
            script.write_text(
                entry().replace("PKG=/opt/rocke_monitor", f"PKG={pkg}", 1)
            )
            for args in (("setup",), ("setup", "--section", "gh"), ("nosuch",)):
                ran.unlink(missing_ok=True)
                with self.subTest(args=args):
                    p = subprocess.run(
                        ["sh", str(script), *args],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env={
                            "PATH": "/usr/bin:/bin",
                            "HOME": t,
                            "RKMON_LOGS": str(d / "logs"),
                        },
                    )
                    self.assertEqual(
                        (p.returncode, p.stdout, ran.exists()), (2, "", False), p.stderr
                    )
                    self.assertRegex(p.stderr, r"^usage: run \| rk ARGS \| ")
                    self.assertNotIn("setup", p.stderr)


class TestRunLoop(unittest.TestCase):
    """The run's pieces, each a shell function of the entry point run on its own."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def test_logs_are_cut_in_place(self):
        logs = self.d / "logs"
        logs.mkdir()
        (logs / "tick.log").write_text("x" * 5000)
        (logs / "web.log").write_text("small\n")
        (logs / "restarts").write_text("".join(f"{i} the tick\n" for i in range(1500)))
        script = f'LOGS={logs}\nLOG_MAX=1000\n{function("rotate_logs")}rotate_logs\n'
        subprocess.run(["sh", "-c", script], check=True, timeout=30)
        self.assertEqual(
            (logs / "tick.log").read_text(), ""
        )  # emptied in place: a writer keeps its file
        self.assertEqual((logs / "tick.log.1").read_text(), "x" * 5000)
        self.assertEqual((logs / "web.log").read_text(), "small\n")
        self.assertFalse((logs / "web.log.1").exists())
        self.assertEqual(len((logs / "restarts").read_text().splitlines()), 200)

    def test_logs_are_cut_at_the_configuration_s_size_only(self):
        """No size of its own: while `rk loop-config` gives none, no log is cut, and the next cut asks again."""
        logs, size = self.d / "logs", self.d / "size"
        logs.mkdir()
        (logs / "tick.log").write_text("x" * 5000)
        rk = fake_rk(
            self.d,
            f'[ "$1" = loop-config ] && [ -f {size} ] && echo "LOOP_LOG_MAX_BYTES=\'1000\'"\nexit 0\n',
        )
        script = (
            f"RK={rk}\nLOGS={logs}\nLOG_MAX=\n{function('loop_value')}{function('rotate_logs')}"
            f'rotate_logs\nwc -c <{logs / "tick.log"}\n: >{size}\nrotate_logs\n'
        )
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30, check=True
        ).stdout
        self.assertEqual(out.split(), ["5000"])  # none cut while there was no size
        self.assertEqual((logs / "tick.log").read_text(), "")
        self.assertEqual((logs / "tick.log.1").read_text(), "x" * 5000)

    def test_nothing_runs_until_the_work_volume_is_set_up(self):
        """While `rk provision --check` says the work volume is not set up, the run says its line once, which names
        the first start as you run it on the host, then looks again every 30 s; once it is set up, the run goes on.
        """
        flag, rounds = self.d / "set-up", self.d / "rounds"
        rk = fake_rk(
            self.d,
            '[ "$1 $2" = "provision --check" ] || exit 2\n'
            f"[ -f {flag} ] && exit 0\n"
            'echo "not set up yet: the first start ($RKMON_WRAPPER start) sets the work volume up"\nexit 1\n',
        )
        script = (
            f"RK={rk}\nRKMON_HOME={self.d}\n"
            'say() { echo "said $*"; }\n'
            # sleep is the wait's clock, run in the background: the second sets the work volume up
            f'sleep() {{ echo "slept $1"; echo >>{rounds}; [ "$(wc -l <{rounds})" -lt 2 ] || : >{flag}; }}\n'
            f'{function("provisioned")}{function("wait_until_set_up")}wait_until_set_up\necho "goes on"\n'
        )
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        ).stdout
        self.assertEqual(
            out.splitlines(),
            [
                "said not set up yet: the first start (rkmon start) sets the work volume up",
                "said waiting: the monitor starts once the work volume is set up",
                "slept 30",
                "slept 30",
                "said set up: starting",
                "goes on",
            ],
        )
        # Set up already: nothing said, nothing waited.
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        ).stdout
        self.assertEqual(out, "goes on\n")

    def test_the_wait_says_why_it_cannot_tell_then_what_changes(self):
        """Only `rk provision --check`'s exit 1 means not set up yet: while it cannot tell (its refusal, exit 4), the
        wait says its lines, and waits on; it says the lines again only once they change (not set up yet), and goes on
        once the work volume is set up."""
        rounds = self.d / "rounds"
        refusal = "REFUSED: the monitor's internal state cannot be read: not JSON"
        rk = fake_rk(
            self.d,
            '[ "$1 $2" = "provision --check" ] || exit 2\n'
            f"n=$(cat {rounds} 2>/dev/null | wc -l)\n"
            f'[ "$n" -lt 2 ] && {{ echo "{refusal}"; exit 4; }}\n'
            '[ "$n" -lt 3 ] && { echo "not set up yet: the first start ($RKMON_WRAPPER start) sets the work volume '
            'up"; exit 1; }\n'
            "exit 0\n",
        )
        script = (
            f"RK={rk}\nRKMON_HOME={self.d}\n"
            'say() { echo "said $*"; }\n'
            f'sleep() {{ echo "slept $1"; echo >>{rounds}; }}\n'
            f'{function("provisioned")}{function("wait_until_set_up")}wait_until_set_up\necho "goes on"\n'
        )
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        ).stdout
        self.assertEqual(
            out.splitlines(),
            [
                f"said {refusal}",
                "said waiting: the monitor starts once the work volume is set up",
                "slept 30",
                "slept 30",
                "said not set up yet: the first start (rkmon start) sets the work volume up",
                "slept 30",
                "said set up: starting",
                "goes on",
            ],
        )

    def test_the_orchestrator_starts_once_it_is_set_up(self):
        """Supervised whatever the setup was at boot: it waits until `rk orchestrate --check` passes, then runs."""
        flag = self.d / "ready"
        rk = fake_rk(
            self.d,
            f'[ "$1 $2" = "orchestrate --check" ] || exit 0\n[ -f {flag} ] && echo READY && exit 0\n'
            "echo 'ORCHESTRATOR NOT SET UP: headless-claude is off'; exit 5\n",
        )
        py = self.d / "py"
        py.write_text('#!/bin/sh\necho runs "$RKMON_CODE"\necho orchestrating "$@"\n')
        py.chmod(0o755)
        script = (
            f'RK={rk}\nORCH_PY={py}\nsay() {{ echo "$*"; }}\ncode() {{ echo /code; }}\n'
            f'{function("runs")}{function("unreadable")}{function("orch_ready")}{function("orchestrator")}'
            "orchestrator\n"
        )
        p = subprocess.Popen(
            ["sh", "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            env=dict(os.environ, RKMON_ORCH_WAIT_S="0.2"),
        )
        time.sleep(1)
        self.assertIsNone(p.poll())  # still waiting
        flag.write_text("")
        out, _ = p.communicate(timeout=30)
        self.assertIn("ORCHESTRATOR NOT SET UP: headless-claude is off", out)
        self.assertIn("the orchestrator is set up: starting it", out)
        # the session, and every rk it starts, runs the code it started from (RKMON_CODE)
        self.assertTrue(
            out.rstrip().endswith(
                "runs /code\norchestrating -B /code/rkmon.py orchestrate"
            ),
            out,
        )

    def test_a_part_runs_only_when_the_features_say_it_does(self):
        """`runs PART` only on `rk features --runs PART`'s exit 0: a part the deployment does not run (1), a
        configuration that cannot be read (4, which `unreadable` tells) and any other failure start nothing.
        """
        for code, said in ((0, "runs"), (1, ""), (4, "unreadable"), (2, "")):
            rk = fake_rk(self.d, f"exit {code}\n")
            script = (
                f'RK={rk}\n{function("runs")}{function("unreadable")}'
                "runs web && echo runs\nunreadable && echo unreadable\nexit 0\n"
            )
            out = subprocess.run(
                ["sh", "-c", script], capture_output=True, text=True, timeout=30
            ).stdout
            self.assertEqual(out.strip(), said, code)

    def test_the_orchestrator_waits_while_the_configuration_cannot_be_read(self):
        """No Claude session starts while `rk features --runs` says the configuration cannot be read (exit 4); once it
        can be read, the orchestrator starts as the features say."""
        readable = self.d / "readable"
        rk = fake_rk(
            self.d,
            f'if [ "$1 $2" = "features --runs" ] && ! [ -f {readable} ]; then echo "REFUSED: x"; exit 4; fi\n'
            "exit 0\n",
        )
        py = self.d / "py"
        py.write_text('#!/bin/sh\necho orchestrating "$@"\n')
        py.chmod(0o755)
        script = (
            f'RK={rk}\nORCH_PY={py}\nsay() {{ echo "$*"; }}\ncode() {{ echo /code; }}\n'
            f'{function("runs")}{function("unreadable")}{function("orch_ready")}{function("orchestrator")}'
            "orchestrator\n"
        )
        p = subprocess.Popen(
            ["sh", "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            env=dict(os.environ, RKMON_ORCH_WAIT_S="0.2"),
        )
        time.sleep(1)
        self.assertIsNone(p.poll())  # still waiting: no session starts
        readable.write_text("")
        out, _ = p.communicate(timeout=30)
        self.assertEqual(
            out.splitlines(),
            [
                "the configuration cannot be read, so no Claude session starts (rk doctor says why and how to fix it)",
                "the configuration can be read",
                "orchestrating -B /code/rkmon.py orchestrate",
            ],
        )

    def test_no_tick_runs_while_the_configuration_cannot_be_read(self):
        """`rk tick-min` fails while the configuration cannot be read: the tick loop that runs without the
        orchestrator runs no tick, says so once and looks again every minute; once it can be read, a tick runs every
        tick_min minutes, the configuration's (the loop has no minutes of its own)."""
        calls, readable = self.d / "calls", self.d / "readable"
        rk = fake_rk(
            self.d,
            f'echo "$*" >>{calls}\n'
            'case "$1" in\n'
            f'tick-min) [ -f {readable} ] || {{ echo "REFUSED: x"; exit 4; }}; echo 7 ;;\n'
            "loop-config) echo \"TICK_TIMEOUT_S='60'\" ;;\n"
            "features|tick-wait) exit 1 ;;\n"
            "esac\nexit 0\n",
        )
        script = (
            f"RK={rk}\nn=0\n"
            'say() { echo "said $*"; }\nrotate_logs() { :; }\nis_stopped() { return 1; }\n'
            "orch_ready() { return 1; }\ndate() { echo 1000; }\n"
            # sleep is the loop's clock: the configuration can be read after the second, and the third ends the run
            f'sleep() {{ echo "slept $1"; n=$((n + 1)); [ "$n" -ne 2 ] || : >{readable}; [ "$n" -lt 3 ] || exit 0; }}\n'
            f'{function("loop_value")}{function("ticks_without_orchestrator")}ticks_without_orchestrator\n'
        )
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        ).stdout
        self.assertEqual(
            out.splitlines(),
            [
                "said no tick runs: the configuration cannot be read (rk doctor says why and how to fix it)",
                "slept 60",
                "slept 60",
                "said the configuration can be read: the ticks run again",
                "slept 420",
            ],
        )
        self.assertEqual(
            [
                re.sub(r"--loop \d+", "--loop PID", c)
                for c in calls.read_text().splitlines()
            ],
            ["tick-wait --loop PID 0"]
            + ["tick-min"] * 3
            + [
                "loop-config",
                "features --runs rules",
                "tick",
                "tick-wait --loop PID 420",
            ],
        )

    def test_the_tick_s_time_limit_is_the_configuration_s_only(self):
        """No limit of its own: without one from `rk loop-config` no tick runs that round, and the loop says so, and
        where to look (rk doctor); with one, the tick has it, and a tick by rules ten minutes more.
        """
        no_limit = (
            "said no tick ran: its time limit could not be read from the configuration (rk doctor says why and how to "
            "fix it); the next round asks again"
        )
        for given, rules, want in (
            ("", 1, no_limit),
            ("TICK_TIMEOUT_S='60'", 1, "timeout -k 30 60 RK tick"),
            ("TICK_TIMEOUT_S='60'", 0, "timeout -k 30 660 RK rules-tick"),
        ):
            rk = fake_rk(
                self.d,
                'case "$1" in\ntick-min) echo 7 ;;\n'
                f'loop-config) echo "{given}" ;;\nfeatures) exit {rules} ;;\ntick-wait) exit 1 ;;\nesac\nexit 0\n',
            )
            script = (
                f"RK={rk}\n"
                'say() { echo "said $*"; }\nrotate_logs() { :; }\nis_stopped() { return 1; }\n'
                'orch_ready() { return 1; }\ntimeout() { echo "timeout $*"; }\nsleep() { exit 0; }\n'
                f'{function("loop_value")}{function("ticks_without_orchestrator")}ticks_without_orchestrator\n'
            )
            out = subprocess.run(
                ["sh", "-c", script], capture_output=True, text=True, timeout=30
            ).stdout
            with self.subTest(given=given, rules=rules):
                self.assertEqual(out.replace(str(rk), "RK"), want + "\n")

    def test_rk_tick_now_s_request_starts_a_tick_at_once(self):
        """The loop waits between its ticks with `rk tick-wait --loop PID SECONDS`, PID its own (keep runs it with &,
        so the entry point's pid would not do), which marks it for rk tick-now. A request it takes (exit 0) starts a
        tick at once, without the rest of the wait, though the orchestrator is set up and its tick is not late; after
        a wait that went by whole (exit 1 at its end), the next round comes at once; when tick-wait ends in any other
        way before the wait is over (it failed, or did not run), the loop sleeps the rest of the wait. The loop's first
        tick-wait, at its start, waits for nothing: it marks the loop, and takes a request already there.
        """
        calls, clock, ends = self.d / "calls", self.d / "clock", self.d / "ends"
        clock.write_text("1000\n")
        # tick-wait's ends in turn, each its exit and the seconds it took: at the start, no request; a request taken;
        # the whole wait; a failure at once; and one that did not run, ten seconds in
        ends.write_text("1 0\n0 0\n1 420\n1 0\n2 10\n")
        rk = fake_rk(
            self.d,
            f'echo "$*" >>{calls}\n'
            'case "$1" in\n'
            "tick-min) echo 7 ;;\n"
            "loop-config) echo \"TICK_TIMEOUT_S='60'\" ;;\n"
            "features|tick-due) exit 1 ;;\n"
            f"tick-wait) read -r rc took <{ends}; sed -i 1d {ends}\n"
            f'    echo "$(($(cat {clock}) + took))" >{clock}; exit "$rc" ;;\n'
            "esac\nexit 0\n",
        )
        script = (
            f"RK={rk}\nn=0\n"
            'say() { echo "said $*"; }\nrotate_logs() { :; }\nis_stopped() { return 1; }\n'
            'orch_ready() { return 0; }\ntimeout() { echo "timeout $*"; }\n'
            # the loop's clock: the time the fake tick-wait moves on; the second sleep ends the run
            f"date() {{ cat {clock}; }}\n"
            'sleep() { echo "slept $1"; n=$((n + 1)); [ "$n" -lt 2 ] || exit 0; }\n'
            f'{function("loop_value")}{function("ticks_without_orchestrator")}'
            'ticks_without_orchestrator &\nloop=$!\nwait\necho "loop $loop, entry point $$"\n'
        )
        out = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
        loop, entry_point = re.fullmatch(
            r"loop (\d+), entry point (\d+)", out.pop()
        ).groups()
        self.assertNotEqual(loop, entry_point)
        self.assertEqual(
            [x.replace(str(rk), "RK") for x in out],
            ["timeout -k 30 60 RK tick", "slept 420", "slept 410"],
        )
        wait = f"tick-wait --loop {loop} 420"
        late = "tick-due --minutes 19"
        self.assertEqual(
            calls.read_text().splitlines(),
            [f"tick-wait --loop {loop} 0", "tick-min", late, wait]
            + ["tick-min", "loop-config", "features --runs rules", wait]
            + ["tick-min", late, wait] * 2,
        )

    def test_the_start_says_nothing_runs_while_the_configuration_cannot_be_read(self):
        text = entry()
        block = text[
            text.index("# What runs, as the features say") : text.index(
                '\nwhile [ "$stopping" -eq 0 ]'
            )
        ]
        for answer, said in (
            (
                'echo "REFUSED: x"; exit 4',
                "nothing runs while the configuration cannot be read (rk doctor says why and how to fix it)",
            ),
            (
                "echo 7",
                "running: the dashboard service, the orchestrator (once it is set up), the tick every 7 min while the "
                "orchestrator is not set up or is late",
            ),
        ):
            rk = fake_rk(self.d, f'if [ "$1" = tick-min ]; then {answer}; fi\nexit 0\n')
            script = f'RK={rk}\nsay() {{ echo "$*"; }}\n{function("runs")}{block}\n'
            out = subprocess.run(
                ["sh", "-c", script], capture_output=True, text=True, timeout=30
            ).stdout
            self.assertEqual(out.strip(), said)

    def test_every_piece_runs_with_umask_027(self):
        """The entry point's umask, which every piece and the shells it starts inherit, is 027
        (worker/perms.py UMASK); what the users share takes its folder's default ACL, so no start makes /work
        group-writable by hand, and no start hands PR code a folder of the monitor's."""
        sys.path.insert(0, str(ROOT / "worker"))
        import perms

        self.assertRegex(entry(), rf"(?m)^umask {perms.UMASK:03o} ")
        self.assertNotIn("chmod g+w", entry())
        self.assertNotIn("pr-user --prepare", entry())

    def test_the_start_keeps_the_clones_own_files_the_runners_groups_to_read(self):
        repo = self.d / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=30)
        rk = fake_rk(self.d, f'[ "$1" = env ] && echo "export RKMON_REPO_DIR={repo}"\n')
        script = f"RK={rk}\n{function('shared_repository')}shared_repository\n"
        subprocess.run(["sh", "-c", script], check=True, timeout=30)
        got = subprocess.run(
            ["git", "-C", str(repo), "config", "core.sharedRepository"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        self.assertEqual(got, "0640")
        # no configuration yet, or no clone: nothing to do, and no failure
        rk = fake_rk(self.d, "exit 1\n")
        script = f"RK={rk}\n{function('shared_repository')}shared_repository\n"
        subprocess.run(["sh", "-c", script], check=True, timeout=30)

    def test_the_run_keeps_every_piece_and_bounds_the_tick_loop(self):
        text = entry()
        run = text[text.index("# run ---") :]
        # Kept unconditionally (no boot-time check), each restarted when it dies.
        for name in (
            "the loop that restarts the admin dashboard",
            "the orchestrator",
            "the tick loop that runs without the orchestrator",
        ):
            self.assertRegex(run, rf'(?m)^keep "{name}" ')
        self.assertNotIn("orchestrated", run)
        # The tick loop that runs without the orchestrator: whether the orchestrator is ready, asked at every round;
        # the tick's time limit; a cut.
        ticks = function("ticks_without_orchestrator")
        self.assertIn('elif [ -n "$asked" ] || ! orch_ready ||', ticks)
        self.assertIn('timeout -k 30 "$limit" "$RK" tick', ticks)
        self.assertIn("TICK_TIMEOUT_S", ticks)
        self.assertIn("rotate_logs", ticks)
        # The worker tools from the code that runs, at each start, never fatal.
        self.assertIn('"$RK" worker-tools --install >>"$LOGS/start.log" 2>&1 ||', run)
        # A stop waits for a write in flight before it signals the group.
        stop = function("stop_all")
        self.assertLess(
            stop.index('flock -w 120 "$STATE/write.lock" true'),
            stop.index("kill -TERM 0"),
        )
        # Each end of a kept piece is recorded for the health check.
        self.assertIn('>>"$LOGS/restarts"', function("keep"))


class TestRecipe(unittest.TestCase):
    def test_the_root_certificates_come_from_a_build_context_of_their_own(self):
        """The ca-certificates stage, empty unless the build gives the context of that name, is mounted for the base
        stage's one step; nothing copies a folder of certificates from the package's code.
        """
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertLess(
            text.index("FROM scratch AS ca-certificates\n"),
            text.index("FROM ${BASE_IMAGE} AS base\n"),
        )
        base = text.split("FROM ${BASE_IMAGE} AS base\n", 1)[1].split("\nFROM ", 1)[0]
        self.assertIn(
            "RUN --mount=type=bind,from=ca-certificates,target=/opt/rkmon/build/ca-certificates \\\n",
            base,
        )
        self.assertNotIn(
            "ca-certificates/",
            "\n".join(x for x in text.splitlines() if x.startswith("COPY")),
        )

    def test_the_image_has_the_python_headers_and_a_healthcheck(self):
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "sh /opt/rkmon/build/pkgs.sh base /opt/rkmon/build/ca-certificates", text
        )
        pkgs = (DOCKER / "pkgs.sh").read_text(encoding="utf-8")
        python = pkgs[pkgs.index("python() {") : pkgs.index("link_python() {")]
        self.assertIn(" python3-dev ", python)  # apt's name; dnf's python3.12-devel
        self.assertIn("python3.12-devel", python)
        self.assertIn("Python.h", python)  # the build checks they are there
        self.assertIn(
            "HEALTHCHECK --interval=5m --timeout=30s --start-period=10m --retries=2 \\\n"
            '    CMD ["/opt/rocke_monitor/docker/entrypoint.sh", "health"]',
            text,
        )
        example = (DOCKER / "compose.local.example.yaml").read_text(encoding="utf-8")
        self.assertIn(
            "- /ramdisk:size=24g,mode=1777,exec", example
        )  # Docker's tmpfs is noexec by default


def rk_ids():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rk_ids", DOCKER / "rk_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PASSWD = (
    "root:x:0:0:root:/root:/bin/bash\n"
    "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
    "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
    "rkmon:x:1000:1000::/home/rkmon:/bin/bash\n"
    "rkwrite:x:1999:1000::/home/rkwrite:/usr/sbin/nologin\n"
    "rkpr1:x:1998:1982::/ramdisk/slots/rkpr1/home:/usr/sbin/nologin\n"
    "rkpr2:x:1997:1981::/ramdisk/slots/rkpr2/home:/usr/sbin/nologin\n"
    "other:x:4242:4242::/nonexistent:/bin/sh\n"
)
GROUP = (
    "root:x:0:\ndaemon:x:1:\nusers:x:100:\nrkmon:x:1000:\nrkpr:x:1998:rkmon,rkpr1,rkpr2\nrkpr1:x:1982:\n"
    "rkpr2:x:1981:\nother:x:4242:\n"
    "nogroup:x:65534:\n"
)


class TestUserIds(unittest.TestCase):
    """docker/rk_ids.py: the deployment's ids onto the image's users, as the entry point runs it (as root)."""

    def setUp(self):
        self.m = rk_ids()
        self.users, self.groups = self.m.read_db(PASSWD, GROUP)
        self.have = self.m.current(self.users, self.groups)

    def plan(self, **env):
        want = self.m.wanted({k: str(v) for k, v in env.items()}, self.have)
        return want, self.m.plan(self.users, self.groups, want)

    def rec(self, want: dict) -> dict:
        """The record of a start that gave the volumes the runners' layout."""
        return dict(want, **{self.m.RUNNERS_KEY: self.m.LAYOUT.SLOTS})

    def test_no_ids_given_changes_nothing(self):
        want, cmds = self.plan()
        self.assertEqual(want, {"uid": 1000, "gid": 1000, "write_uid": 1999})
        self.assertEqual(cmds, [])
        self.assertEqual(self.m.file_maps(self.have, want, self.rec(want)), ({}, {}))

    def test_the_operator_s_ids(self):
        want, cmds = self.plan(RKMON_UID=4243, RKMON_GID=4343)
        self.assertEqual(
            cmds,
            [["groupmod", "-g", "4343", "rkmon"], ["usermod", "-u", "4243", "rkmon"]],
        )
        # The first start: the volumes' files of the image's ids follow (and the single PR-code user's, 1998 in the
        # images before the runners, become the monitor's); a container made again: nothing to walk.
        self.assertEqual(
            self.m.file_maps(self.have, want, {}),
            ({1000: 4243, 1998: 4243}, {1000: 4343}),
        )
        self.assertEqual(self.m.file_maps(self.have, want, self.rec(want)), ({}, {}))
        # Ids changed since the last start: the volumes' files of the last ones follow.
        last = self.rec(dict(want, uid=5000, gid=5001))
        self.assertEqual(
            self.m.file_maps(self.have, want, last), ({5000: 4243}, {5001: 4343})
        )

    def test_the_single_pr_code_users_files_become_the_monitors_once(self):
        """Before the runners, one PR-code user owned what PR code wrote in the volumes: the first start with the
        runners gives its files to the monitor, and records that it did; no start after it moves a runner's file. That
        user's id is the package's, the first runner's (its group numbered as it, the runners' group's id), never the
        RKMON_PR_UID a record of a start before the runners may hold."""
        want, _cmds = self.plan(RKMON_UID=4243, RKMON_GID=4343)
        old = {"uid": 4243, "gid": 4343, "write_uid": 1999, "pr_uid": 4245}
        self.assertEqual(self.m.file_maps(self.have, want, old), ({1998: 4243}, {}))
        self.assertEqual(self.m.file_maps(self.have, want, self.rec(old)), ({}, {}))

    def test_no_file_of_an_id_another_user_or_group_holds_moves(self):
        """The record is the monitor's to write: an id in it that another user or group of the container holds (taken:
        as a user's, root's, a system account's, a PR runner's, the monitor's or the GitHub writer's; as a group's,
        root's, the runners' group's or a runner's own) moves no file, whatever key names it, and a start says so in
        one line; the record's other ids move as they would."""
        want, _cmds = self.plan(RKMON_UID=4243, RKMON_GID=4343)
        last = self.rec(dict(want, uid=5000, gid=5001))
        users, groups = {5000: 4243}, {5001: 4343}
        self.assertEqual(self.m.file_maps(self.have, want, last), (users, groups))
        for key, value in (
            ("uid", 0),
            ("uid", 999),
            ("uid", 65534),
            ("uid", 1990),
            ("uid", 1999),
            ("write_uid", 0),
            ("write_uid", 1998),
            ("write_uid", 4243),
            ("gid", 0),
            ("gid", 1998),
            ("gid", 1970),
        ):
            with self.subTest(key=key, value=value):
                forged = dict(last, **{key: value})
                self.assertEqual(self.m.taken(forged, want), {key: value})
                self.assertEqual(
                    self.m.file_maps(self.have, want, forged),
                    ({} if key == "uid" else users, {} if key == "gid" else groups),
                )
        env = {"RKMON_UID": "4243", "RKMON_GID": "4343"}
        with tempfile.TemporaryDirectory() as t:
            rc, _ran, _owners, got, err = self.start(
                pathlib.Path(t), env, record=dict(last, uid=1999)
            )
        self.assertEqual(rc, 0, err)
        said = [x for x in err.splitlines() if "ids of another user or group" in x]
        self.assertEqual(len(said), 1, err)
        self.assertIn("(uid 1999)", said[0])
        self.assertEqual(got, self.rec(want))

    def test_a_base_image_account_on_a_wanted_id_goes_a_system_one_is_refused(self):
        _want, cmds = self.plan(RKMON_UID=4242)
        self.assertIn(["userdel", "-r", "other"], cmds)
        self.assertLess(
            cmds.index(["userdel", "-r", "other"]),
            cmds.index(["usermod", "-u", "4242", "rkmon"]),
        )
        # A system account's id (below 1000, or nobody's), whatever holds it: the base image's daemon, nobody. The
        # monitor's is set by its deploy setting.
        for env in ({"RKMON_UID": 1}, {"RKMON_UID": 65534}):
            with self.subTest(env=env), self.assertRaises(self.m.Refused) as cm:
                self.plan(**env)
            self.assertIn("a system account's id", str(cm.exception))
            self.assertIn(
                "set host.users.monitor_uid to an id of 1000 or more",
                str(cm.exception),
            )
        # A system account of the image on a regular id: the monitor's user takes another by its deploy setting.
        users, groups = self.m.read_db(
            PASSWD + "nfsnobody:x:4300:4300::/nonexistent:/usr/sbin/nologin\n", GROUP
        )
        with self.assertRaises(self.m.Refused) as cm:
            self.m.plan(users, groups, self.m.wanted({"RKMON_UID": "4300"}, self.have))
        self.assertIn("is the image's system account nfsnobody", str(cm.exception))
        self.assertIn(
            "set host.users.monitor_uid to another id in your deploy settings",
            str(cm.exception),
        )
        self.assertNotRegex(str(cm.exception), r"container\.")
        # A group the image has on the wanted id is taken as the monitor's group, not renumbered.
        _want, cmds = self.plan(RKMON_GID=100)
        self.assertEqual(
            cmds,
            [["usermod", "-g", "100", "rkmon"], ["usermod", "-g", "100", "rkwrite"]],
        )

    def test_ids_that_would_mix_the_users_are_refused(self):
        """Each refusal says what to change: the monitor's ids, by their deploy settings' keys (the GitHub writer's
        and the PR runners' are the package's: test_the_package_s_ids_are_no_deployment_s).
        """
        uid = "set host.users.monitor_uid to another id in your deploy settings"
        gid = "set host.users.monitor_gid to another group's id in your deploy settings"
        for env, words, fix in (
            ({"RKMON_UID": 1999}, "is the GitHub writer's id", uid),
            ({"RKMON_UID": 1990}, "is a PR runner's id (1983 to 1998", uid),
            ({"RKMON_UID": 1998}, "is a PR runner's id", uid),
            ({"RKMON_UID": 1983}, "is a PR runner's id", uid),
            ({"RKMON_GID": 1998}, "is the PR runners' group's id", gid),
            ({"RKMON_GID": 1982}, "is a PR runner's own group's id (1967 to 1982", gid),
            ({"RKMON_GID": 1967}, "is a PR runner's own group's id", gid),
            ({"RKMON_UID": "me"}, "is not a user id", ""),
            ({"RKMON_UID": 0}, "is not a user id", ""),
        ):
            with self.subTest(env=env):
                with self.assertRaises(self.m.Refused) as cm:
                    self.plan(**env)
                self.assertIn(words, str(cm.exception))
                self.assertIn(fix, str(cm.exception))
                self.assertNotRegex(
                    str(cm.exception), r"container\.(uid|gid|write_uid|pr_uid)"
                )

    def test_the_clashes_are_each_ids_with_its_first_reason(self):
        """One clash for each of the monitor's ids that would mix the users, the first reason found for it, in the ids'
        order; an id not known (None) is checked against nothing. The writer's id is the package's: it is checked
        against, never refused. The deploy settings' checks ask with the package's (settings_ids).
        """
        clash = self.m.Clash
        self.assertEqual(
            self.m.clashes({"uid": 1999, "gid": 1998, "write_uid": 1999}),
            [
                clash("uid", "the GitHub writer's id", "another id"),
                clash("gid", "the PR runners' group's id", "another group's id"),
            ],
        )
        # The monitor's id on a runner's is the monitor's clash; the writer's id on one is no clash (the package's).
        self.assertEqual(
            self.m.clashes({"uid": 1998, "gid": 4343, "write_uid": 1997}),
            [
                clash(
                    "uid",
                    "a PR runner's id (1983 to 1998, the package's)",
                    "another id",
                )
            ],
        )
        self.assertEqual(
            self.m.clashes({"uid": 999, "gid": None, "write_uid": 999}),
            [
                clash(
                    "uid",
                    "a system account's id (below 1000, or nobody's 65534)",
                    "an id of 1000 or more",
                )
            ],
        )
        self.assertEqual(self.m.clashes(self.m.settings_ids(None, None)), [])
        self.assertEqual(self.m.clashes(self.m.settings_ids(4242, 4343)), [])
        self.assertEqual(
            self.m.settings_ids(4242, 4343),
            {"uid": 4242, "gid": 4343, "write_uid": self.m.LAYOUT.WRITER.uid},
        )
        # A runner's own group is no group the monitor takes: each one's id, and none around them.
        gids = sorted(self.m.RUNNER_GIDS)
        self.assertEqual(gids, [u.gid for u in reversed(self.m.LAYOUT.PR_RUNNERS)])
        for g in gids:
            with self.subTest(gid=g):
                (c,) = self.m.clashes(self.m.settings_ids(4242, g))
                self.assertEqual(c.key, "gid")
                self.assertIn("a PR runner's own group's id", c.why)
        for g in (gids[0] - 1, gids[-1] + 1):
            self.assertEqual(self.m.clashes(self.m.settings_ids(4242, g)), [])

    def test_the_package_s_ids_are_no_deployment_s(self):
        """The GitHub writer's id and the PR runners' are the package's (container_layout): variables that would name
        them change nothing, whatever their values."""
        for env in (
            {"RKMON_WRITE_UID": 4244, "RKMON_PR_UID": 4245},
            {"RKMON_WRITE_UID": 1000, "RKMON_PR_UID": 1},
        ):
            with self.subTest(env=env):
                want, cmds = self.plan(**env)
                self.assertEqual(want, self.m.settings_ids(1000, 1000))
                self.assertEqual(cmds, [])
        want, cmds = self.plan(RKMON_UID=4243, RKMON_WRITE_UID=4244, RKMON_PR_UID=4245)
        self.assertEqual(want, self.m.settings_ids(4243, 1000))
        self.assertEqual(cmds, [["usermod", "-u", "4243", "rkmon"]])

    def test_the_volumes_files_follow(self):
        me, mine = os.getuid(), os.getgid()
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "a" / "b").mkdir(parents=True)
            (d / "a" / "b" / "f").write_text("x")
            os.symlink("f", d / "a" / "b" / "link")
            # Mapped onto themselves (a test runs without root): every entry is visited and counted, links too.
            self.assertEqual(self.m.remap_files([str(d)], {me: me}, {}), 5)
            self.assertEqual(
                self.m.remap_files([str(d)], {me + 1: me}, {mine + 1: mine}), 0
            )
            self.assertEqual(self.m.remap_files([str(d / "none")], {me: me}, {}), 0)

    def test_the_volumes_folders_get_their_owners_at_every_start(self):
        """Docker gives an empty volume's folder the image's owner in each new container: the folders' owners are set
        whatever the record says."""
        want = self.m.settings_ids(4242, 4343)
        self.assertEqual(
            self.m.root_owners(want),
            {
                "/work": (4242, 4343),
                "/secrets": (4242, 4343),
                "/secrets-write": (1999, 4343),
            },
        )
        me, mine = os.getuid(), os.getgid()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                self.m.fix_roots({d: (me, mine)}), 0
            )  # already right: nothing to do
            self.assertEqual(self.m.fix_roots({d + "/none": (me, mine)}), 0)

    def passwd_in(self, d: pathlib.Path) -> str:
        """PASSWD with the monitor's ids this test's own (it runs without root) and its home D/m, and the writer's home
        D/w, where there is none: every home rk_ids walks is in D, so a test reads nothing of the machine it runs on (an
        image has a /home/rkwrite of its own)."""
        return PASSWD.replace(
            "rkmon:x:1000:1000::/home/rkmon:",
            f"rkmon:x:{os.getuid()}:{os.getgid()}::{d}/m:",
        ).replace("::/home/rkwrite:", f"::{d}/w:")

    def start(
        self,
        d: pathlib.Path,
        env: dict,
        record=None,
        passwd=None,
        argv=(),
        chowned=None,
    ) -> tuple:
        """rk_ids.main as the entry point runs it, as root, with ARGV, on stand-ins: the image's passwd (`passwd`,
        else PASSWD with its homes in `d`) and group (GROUP), the volumes' folders and the record in `d`, the commands
        it runs and the owners it sets recorded (a test runs without root; the fchown calls, by an open file, in the
        list `chowned`, when given). (exit code, the commands, the lchown calls, the record, its stderr).
        """
        chowned = [] if chowned is None else chowned
        (d / "passwd").write_text(passwd or PASSWD.replace(":/home/", f":{d}/home/"))
        (d / "group").write_text(GROUP)
        vols = tuple(str(d / v) for v in ("work", "secrets", "secrets-write"))
        for v in vols:
            os.makedirs(v, exist_ok=True)
        rec = d / "work" / ".rkmon_ids.json"
        if record is not None:
            rec.write_text(json.dumps(record))
        ran, owners = [], []
        names = dict(
            PASSWD=str(d / "passwd"),
            GROUP=str(d / "group"),
            VOLUMES=vols,
            RECORD=str(rec),
        )
        err = io.StringIO()
        with contextlib.ExitStack() as s:
            for k, v in names.items():
                s.enter_context(mock.patch.object(self.m, k, v))
            s.enter_context(
                mock.patch.object(
                    self.m.subprocess,
                    "run",
                    lambda c, **k: ran.append(c)
                    or subprocess.CompletedProcess(c, 0, ""),
                )
            )
            s.enter_context(
                mock.patch.object(
                    self.m.os, "lchown", lambda p, u, g: owners.append((p, u, g))
                )
            )
            s.enter_context(
                mock.patch.object(self.m.os, "fchown", lambda *a: chowned.append(a))
            )
            s.enter_context(mock.patch.dict(os.environ, env))
            s.enter_context(contextlib.redirect_stderr(err))
            s.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rc = self.m.main(list(argv))
        got = json.loads(rec.read_text()) if rec.is_file() else None
        return rc, ran, owners, got, err.getvalue()

    def test_the_users_homes_are_the_ones_passwd_gives_the_monitor_and_the_writer(self):
        self.assertEqual(self.m.homes(PASSWD), ("/home/rkmon", "/home/rkwrite"))
        moved = PASSWD.replace("::/home/rkmon:", "::/srv/m:").replace(
            "::/home/rkwrite:", "::/:"
        )
        self.assertEqual(self.m.homes(moved), ("/srv/m",))

    def test_a_renumbering_gives_the_users_homes_the_new_ids(self):
        """usermod moves no file: what is in the monitor's home (the one /etc/passwd gives it) gets its new ids, as
        the volumes' files do; a runner's home is not touched."""
        env = {"RKMON_UID": "4243", "RKMON_GID": "4343"}
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            for user in ("m", "r"):
                (d / user).mkdir()
                (d / user / "f").write_text("x")
            passwd = self.passwd_in(d).replace("/ramdisk/slots/rkpr1/home", f"{d}/r")
            rc, _ran, owners, _got, err = self.start(d, env, passwd=passwd)
            self.assertEqual(rc, 0, err)
            mine_now = {p for p, u, g in owners if (u, g) == (4243, 4343)}
            self.assertLessEqual({str(d / "m"), str(d / "m" / "f")}, mine_now)
            self.assertFalse([p for p, _u, _g in owners if p.startswith(f"{d}/r")])

    def test_each_start_maps_the_ids_and_gives_the_volumes_folders_their_owners(self):
        """The first start renumbers the users and records the ids; each start after it, with the ids recorded (a new
        container: docker gives an empty volume's folder the image's owner again), still gives each volume's own
        folder its owner."""
        env = {"RKMON_UID": "4243", "RKMON_GID": "4343"}
        want = {"uid": 4243, "gid": 4343, "write_uid": 1999}
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            roots = {
                (str(d / "work"), 4243, 4343),
                (str(d / "secrets"), 4243, 4343),
                (str(d / "secrets-write"), 1999, 4343),
            }
            rc, ran, owners, got, _err = self.start(d, env)
            self.assertEqual(rc, 0)
            self.assertIn(["usermod", "-u", "4243", "rkmon"], ran)
            self.assertIn(["groupmod", "-g", "4343", "rkmon"], ran)
            self.assertTrue(roots <= set(owners), owners)
            self.assertEqual(got, self.rec(want))
            # The record is a file the monitor shares: its group's, no one else's.
            self.assertEqual(
                (d / "work" / ".rkmon_ids.json").stat().st_mode & 0o777, 0o660
            )
            self.assertFalse([c for c in ran if c[-1].startswith("rkpr")], ran)
            rc, ran, owners, got, _err = self.start(d, env, record=self.rec(want))
            self.assertEqual(rc, 0)
            self.assertEqual(
                sorted(owners), sorted(roots)
            )  # the folders only: no file moves

    def test_a_start_writes_its_record_through_no_link(self):
        """The record's folder is the monitor's (write_record): a start writes the record whole, the monitor's, by a
        file it makes new and gives its owner by its descriptor; the links put at the record's name and at the name a
        start wrote it by before are not followed, and the files they name stay as they were.
        """
        env = {"RKMON_UID": "4243", "RKMON_GID": "4343"}
        want = {"uid": 4243, "gid": 4343, "write_uid": 1999}
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            record = d / "work" / ".rkmon_ids.json"
            record.parent.mkdir()
            links = (record, record.with_name(record.name + ".new"))
            roots = (d / "root_0", d / "root_1")
            for target, link in zip(roots, links):
                target.write_text("root's\n")
                target.chmod(0o600)
                link.symlink_to(target)
            chowned = []
            rc, _ran, _owners, got, err = self.start(d, env, chowned=chowned)
            self.assertEqual(rc, 0, err)
            self.assertEqual(
                [(p.read_text(), p.stat().st_mode & 0o7777) for p in roots],
                [("root's\n", 0o600)] * len(roots),
            )
            self.assertFalse(record.is_symlink())
            self.assertEqual(got, self.rec(want))
            self.assertEqual(record.stat().st_mode & 0o777, 0o660)
            self.assertEqual(
                [(type(f), u, g) for f, u, g in chowned], [(int, 4243, 4343)]
            )
            self.assertEqual(
                sorted(p.name for p in record.parent.iterdir()),
                sorted(link.name for link in links),
            )

    def read_record(self, path: pathlib.Path) -> tuple:
        """rk_ids.recorded(PATH) and what it said, on a thread given a few seconds: a read that waits fails the test
        (the pipe's write end, opened then, lets it go)."""
        got, err = [], io.StringIO()
        reader = threading.Thread(
            target=lambda: got.append(self.m.recorded(str(path))), daemon=True
        )
        with contextlib.redirect_stderr(err):
            reader.start()
            reader.join(5)
            if reader.is_alive():
                with open(path, "wb"):
                    pass
                reader.join(5)
                self.fail(f"reading {path.name} waited")
        return got[0], err.getvalue()

    def test_a_record_that_is_not_a_regular_file_is_no_record(self):
        """The record's folder is the monitor's: a link there (to a record a start would take), a pipe (no writer holds
        it open: a read of it waits for one) or a folder is no record, read without waiting and said in one line; a
        regular file is read."""
        want = {"uid": 4243, "gid": 4343, "write_uid": 1999}
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            real = d / "real.json"
            real.write_text(json.dumps(self.rec(want)))
            kinds = {
                "link": lambda p: p.symlink_to(real),
                "pipe": os.mkfifo,
                "folder": os.mkdir,
            }
            for kind, make in kinds.items():
                with self.subTest(kind=kind):
                    path = d / kind
                    make(path)
                    self.assertEqual(
                        self.read_record(path),
                        (
                            {},
                            f"rk_ids: {path} is not a regular file: read as no record\n",
                        ),
                    )
            self.assertEqual(self.read_record(real), (self.rec(want), ""))

    def test_a_one_off_of_the_logins_walk_gives_the_users_their_ids_alone(self):
        """--users, in a one-off container of the logins' walk beside the running one, which may run an older
        package's image: the users get the deployment's ids and their homes follow; a volume's own folder gets its
        owner only while it has the image's own (one docker just made); and nothing else of the volumes changes: no
        file walked, a folder of another owner left as it is, the record neither read nor written, also an older
        package's, from before the runners' layout, which a start of this image would move from.
        """
        me, mine = os.getuid(), os.getgid()
        env = {"RKMON_UID": "4243", "RKMON_GID": "4343"}
        new = (4243, 4343)
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            (d / "m").mkdir()
            (d / "m" / "f").write_text("x")
            (d / "work").mkdir()
            (d / "work" / "state").write_text(
                "x"
            )  # the image's monitor's, as a start would give it the new id
            passwd = self.passwd_in(d)
            older = {"uid": me, "gid": mine, "write_uid": 1999}
            rc, ran, owners, got, err = self.start(
                d, env, record=older, passwd=passwd, argv=["--users"]
            )
            self.assertEqual(rc, 0, err)
            self.assertIn(["usermod", "-u", "4243", "rkmon"], ran)
            self.assertIn(["groupmod", "-g", "4343", "rkmon"], ran)
            home = {(str(d / "m"), *new), (str(d / "m" / "f"), *new)}
            self.assertLessEqual(home, set(owners))
            # the work and secrets volumes' folders are the image's monitor's: docker just made them; the write
            # volume's is another owner's, and stays
            self.assertEqual(
                sorted(set(owners) - home),
                [(str(d / "secrets"), *new), (str(d / "work"), *new)],
            )
            self.assertEqual(got, older)
            # a start of this image, on the same volumes, walks their files and writes its record
            rc, ran, owners, got, err = self.start(d, env, record=older, passwd=passwd)
            self.assertEqual(rc, 0, err)
            self.assertIn((str(d / "work" / "state"), *new), owners)
            self.assertEqual(
                got, self.rec({"uid": 4243, "gid": 4343, "write_uid": 1999})
            )

    def test_a_one_off_of_the_logins_walk_gives_its_own_temporary_folders_the_start_s_modes_alone(
        self,
    ):
        """rk_layout.py --system-tmp, the walk's root step after the ids (the entry point's order: the test below):
        the one-off container's own /tmp and /var/tmp (SYSTEM_TMP; stand-ins here, as the image leaves them: no one
        else's to write) get the start's owner, group and mode, so the walk's rk, as the monitor or as the writer (in
        the monitor's group), makes its temporary folders there. Nothing of the volumes changes (a working folder as an
        older start left it, no RAM disk folder or slots made), and nothing is said: what the walk's command prints is
        the walk's to read.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rk_layout", DOCKER / "rk_layout.py"
        )
        layout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layout)
        me, mine = os.getuid(), os.getgid()
        other = ([g for g in os.getgroups() if g != mine] or [mine])[0]
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            work, ramdisk = d / "work", d / "ramdisk"
            (work / "logs").mkdir(parents=True)
            (work / "logs" / "tick.log").write_text("x")
            os.chmod(work / "logs" / "tick.log", 0o664)
            os.chmod(work / "logs", 0o2775)
            tmp_dirs = (d / "tmp", d / "var_tmp")
            for f in tmp_dirs:
                f.mkdir()
                os.chown(f, -1, other)
                os.chmod(f, 0o1775)

            def volume() -> dict:
                return {
                    p: (os.lstat(p).st_mode, os.lstat(p).st_gid)
                    for p in (work, *work.rglob("*"))
                }

            before = volume()
            who = types.SimpleNamespace(
                root=me,
                monitor=me,
                group=mine,
                writer=me + 100000,
                pr=other,
                runners={"rkpr1": me},
            )
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(layout, "ids", lambda: who), mock.patch.multiple(
                layout.LAYOUT,
                WORK_DIR=str(work),
                RAMDISK=str(ramdisk),
                SYSTEM_TMP=tuple(str(f) for f in tmp_dirs),
            ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(layout.main(["--system-tmp"]), 0)
            for f in tmp_dirs:
                st = os.lstat(f)
                self.assertEqual(
                    (st.st_mode & 0o7777, st.st_uid, st.st_gid), (0o1770, me, mine), f
                )
            self.assertEqual(volume(), before)
            self.assertFalse(ramdisk.exists())
            self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_refused_ids_stop_the_start_with_nothing_changed(self):
        with tempfile.TemporaryDirectory() as t:
            rc, ran, owners, got, err = self.start(
                pathlib.Path(t), {"RKMON_UID": "1999"}
            )
        self.assertEqual(rc, 2)
        self.assertIn("is the GitHub writer's id", err)
        self.assertIn("nothing changed", err)
        self.assertEqual((ran, owners, got), ([], [], None))

    def test_the_entry_point_maps_the_ids_before_it_drops_to_the_user(self):
        """As root, the entry point runs rk_ids.py first, then rk_layout.py: a refusal or a failure of either ends the
        start with its exit code before setpriv drops to the user; the health check drops to rkmon without them. For
        `run` (the default command), the root step (rk_root.py start) comes third, its lines in the log; when it fails
        (any exit but 0), nothing starts, and the container waits. No other command runs it; code-install and
        code-back, as root (RKMON_AS=root), run it with their action, without the other steps. An rk command handed
        your confirmed settings (RKMON_CONFIRMED_LAYER: the logins' walk, beside the running container) runs
        rk_ids.py --users, then rk_layout.py --system-tmp (that container's own temporary folders alone), then has
        them written by root (rk_root.py hand-layer, the value on its stdin), and runs without the variable; when one
        of them fails, it does not run. No other command has them written, and none keeps the variable.
        """
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            (d / "pkg" / "docker").mkdir(parents=True)
            ids = d / "pkg" / "docker" / "rk_ids.py"
            layout = d / "pkg" / "docker" / "rk_layout.py"
            root = d / "pkg" / "docker" / "rk_root.py"
            order = d / "order"
            stubs = d / "bin"
            stubs.mkdir()
            marker = d / "setpriv.args"
            kept = (
                d / "setpriv.handed"
            )  # the handed settings, as the dropped command's environment holds them
            handed = d / "handed"  # what root's hand-layer got on its stdin
            for name, body in (
                (
                    "id",
                    '[ "$*" = -u ] && { echo 0; exit 0; }\n'
                    '[ "$1" = -g ] && { echo 1000; exit 0; }\nexec /usr/bin/id "$@"\n',
                ),
                (
                    "setpriv",
                    f'echo "$*" > {marker}\necho "${{RKMON_CONFIRMED_LAYER-unset}}" > {kept}\n',
                ),
            ):
                (stubs / name).write_text("#!/bin/sh\n" + body)
                (stubs / name).chmod(0o755)
            script = d / "entrypoint.sh"
            script.write_text(
                entry().replace("PKG=/opt/rocke_monitor", f"PKG={d / 'pkg'}", 1)
            )

            said = []

            def run(rc_ids: int, *args, rc_layout: int = 0, rc_root: int = 0, **env):
                for f in (marker, order, kept, handed):
                    f.unlink(missing_ok=True)
                ids.write_text(
                    f"import sys\nopen({str(order)!r}, 'a').write(' '.join(['ids', *sys.argv[1:]]) + ' ')\n"
                    "print('rk_ids: refused: nothing changed', file=sys.stderr)\n"
                    f"sys.exit({rc_ids})\n"
                )
                layout.write_text(
                    f"import sys\nopen({str(order)!r}, 'a').write(' '.join(['layout', *sys.argv[1:]]) + ' ')\n"
                    f"sys.exit({rc_layout})\n"
                )
                root.write_text(
                    f"import sys\nopen({str(order)!r}, 'a').write('root:' + ' '.join(sys.argv[1:]) + ' ')\n"
                    f"if sys.argv[1:] == ['hand-layer']: open({str(handed)!r}, 'w').write(sys.stdin.read())\n"
                    "print('moved the GitHub write login from A to B')\n"
                    f"sys.exit({rc_root})\n"
                )
                p = subprocess.run(
                    ["sh", str(script), *args],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": t, **env},
                )
                said[:] = [p.stdout]
                return p.returncode, marker.read_text() if marker.exists() else None

            self.assertEqual(run(2, "rk", "status"), (2, None))
            self.assertEqual(order.read_text(), "ids ")
            self.assertEqual(
                run(0, "rk", "status"),
                (
                    0,
                    f"--reuid=rkmon --regid=1000 --init-groups -- {script} rk status\n",
                ),
            )
            self.assertEqual(order.read_text(), "ids layout ")
            self.assertNotIn("moved", said[0])
            self.assertEqual(run(0, "rk", "status", rc_layout=1), (1, None))
            self.assertEqual(
                run(2, "health", rc_layout=1),
                (0, f"--reuid=rkmon --regid=1000 --init-groups -- {script} health\n"),
            )
            self.assertFalse(order.exists())
            for args in (("run",), ()):
                with self.subTest(args=args):
                    self.assertEqual(
                        run(0, *args),
                        (
                            0,
                            f"--reuid=rkmon --regid=1000 --init-groups -- {script}"
                            + "".join(f" {a}" for a in args)
                            + "\n",
                        ),
                    )
                    self.assertEqual(order.read_text(), "ids layout root:start ")
                    self.assertRegex(
                        said[0],
                        r"(?m)^\S+Z rkmon-container: moved the GitHub write login from A to B$",
                    )
            self.assertEqual(run(0, "run", rc_layout=1), (1, None))
            self.assertEqual(order.read_text(), "ids layout ")
            # A root step that fails, whatever its exit (5: the code volume fails its check; 3: no write user; 1: an
            # error, or a traceback): the container waits (here its sleep stops it), and nothing starts.
            (stubs / "sleep").write_text("#!/bin/sh\nkill -TERM $PPID\n")
            (stubs / "sleep").chmod(0o755)
            for rc in (5, 3, 1):
                with self.subTest(rc_root=rc):
                    self.assertEqual(run(0, "run", rc_root=rc), (0, None))
                    self.assertEqual(order.read_text(), "ids layout root:start ")
                    self.assertIn(
                        f"rkmon-container: the monitor does not start: the root step failed (exit {rc}; the lines "
                        "above say why); waiting",
                        said[0],
                    )
            (stubs / "sleep").unlink()
            # Root changes the code alone: no other step runs first, and no other user runs it.
            for action in ("code-install", "code-back"):
                with self.subTest(action=action):
                    self.assertEqual(
                        run(0, action, rc_root=4, RKMON_AS="root"), (4, None)
                    )
                    self.assertEqual(order.read_text(), f"root:{action} ")
                    self.assertEqual(
                        run(0, action),
                        (
                            0,
                            f"--reuid=rkmon --regid=1000 --init-groups -- {script} {action}\n",
                        ),
                    )
                    self.assertFalse(order.exists())
                    p = subprocess.run(
                        ["sh", str(script), action],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env={"PATH": "/usr/bin:/bin", "HOME": t},
                    )
                    self.assertEqual(
                        (p.returncode, p.stderr.strip()),
                        (
                            2,
                            f"{action}: as root only (rkmon apply and rkmon rollback run it)",
                        ),
                    )
                    self.assertFalse(order.exists())
            # Your confirmed settings handed to an rk command, as the monitor or as the writer: written by root third,
            # whole, then the command runs without the variable; not written, the command does not run.
            layer = '{\n "kind": "confirmed_deploy_settings"\n}\n'
            for user in ("rkmon", "rkwrite"):
                with self.subTest(user=user):
                    self.assertEqual(
                        run(
                            0,
                            "rk",
                            "secret-set",
                            "teams_webhook",
                            RKMON_AS=user,
                            RKMON_CONFIRMED_LAYER=layer,
                        ),
                        (
                            0,
                            f"--reuid={user} --regid=1000 --init-groups -- {script} rk secret-set "
                            "teams_webhook\n",
                        ),
                    )
                    self.assertEqual(
                        order.read_text(),
                        "ids --users layout --system-tmp root:hand-layer ",
                    )
                    self.assertEqual(handed.read_text(), layer)
                    self.assertEqual(kept.read_text(), "unset\n")
            self.assertEqual(
                run(0, "rk", "logins", rc_root=1, RKMON_CONFIRMED_LAYER=layer),
                (1, None),
            )
            self.assertEqual(
                order.read_text(), "ids --users layout --system-tmp root:hand-layer "
            )
            self.assertEqual(
                run(2, "rk", "logins", RKMON_CONFIRMED_LAYER=layer), (2, None)
            )
            self.assertEqual(order.read_text(), "ids --users ")
            self.assertFalse(handed.exists())
            self.assertEqual(
                run(0, "rk", "logins", rc_layout=1, RKMON_CONFIRMED_LAYER=layer),
                (1, None),
            )
            self.assertEqual(order.read_text(), "ids --users layout --system-tmp ")
            self.assertFalse(handed.exists())
            # Any other command: nothing written, and the variable gone.
            for args, steps in (
                (("run",), "ids layout root:start "),
                (("shell",), "ids layout "),
                (("health",), None),
            ):
                with self.subTest(args=args):
                    rc, dropped = run(0, *args, RKMON_CONFIRMED_LAYER=layer)
                    self.assertEqual((rc, dropped is None), (0, False))
                    self.assertEqual(
                        order.read_text() if order.exists() else None, steps
                    )
                    self.assertFalse(handed.exists())
                    self.assertEqual(kept.read_text(), "unset\n")
            self.assertEqual(
                run(0, "code-install", RKMON_AS="root", RKMON_CONFIRMED_LAYER=layer),
                (0, None),
            )
            self.assertEqual(order.read_text(), "root:code-install ")
            self.assertFalse(handed.exists())
        rk = (ROOT / "rk").read_text(encoding="utf-8")
        self.assertIn(
            "exec setpriv --reuid=rkmon", rk
        )  # rk never runs as root in the container


class TestDockerignore(unittest.TestCase):
    """The image's build context, as .dockerignore makes it (tools/generated.py writes it): the package's own entries
    (image.BUILD_CONTEXT), and under them nothing that may hold a secret or a private value.
    """

    def patterns(self) -> list[str]:
        return build_context.patterns(ROOT)

    def tracked(self) -> list[str]:
        files = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True
        ).stdout.split()
        if not files:
            self.skipTest("not a git checkout")
        return files

    def test_no_secret_or_settings_file_reaches_the_image(self):
        pats = self.patterns()
        for path in (
            ".env",
            "docker/.env",
            "docker/.env.local",
            "admin_token",
            "sub/dir/claude_key",
            "sub/dir/claude_env",  # its older name
            "teams_webhook",
            "worker_api_key",
            "jira_mcp_headers",
            "web_host_ssh_key",
            "web_host_ssh_key.pub",
            "webhost_ed25519",  # its older name
            "webhost_ed25519.pub",
            "privacy_deny.txt",
            "key.pem",
            "docker/x/key.pem",
            "config.json",
            "config.json.bak-20261111",
            "docker/rkmon-settings-20261111.yaml",
            "docker/my-settings.yaml",
            "settings.json",
            "state/state.json",
            "docker/compose.override.yaml",
            "docker/compose.override.yaml.pre-shadow",
            "tools/__pycache__/x.pyc",
            "docker/ca-certificates/corp.crt",
            "docker/ca-certificates/corp.pem",
            "docker/.env.prev",
            "docker/compose.override.yaml.prev",
            "docker/.rkmon/records.json",
            "docker/.rkmon/records.lock",
            "docker/.rkmon/under_way.lock",
            "docker/gh-write/hosts.yml",
            "rkmon_lib/deep/secrets/anything",
            "tests/.config/gh/hosts.yml",
            "tools/glm_api_key",
            "notes.txt",  # at the top, not one of the package's entries
            "sandbox/x.py",
        ):
            self.assertTrue(build_context.ignored(pats, path), path)
        for path in (
            "rkmon.py",
            "rk",
            "docker/Dockerfile",
            "docker/compose.yaml",
            "rkmon_lib/settings.py",
            "tests/test_settings.py",
            "tests/container/deploy.yaml",
            "config.defaults.json",
        ):
            self.assertFalse(build_context.ignored(pats, path), path)

    def test_every_name_gitignore_keeps_out_is_kept_out_of_the_image(self):
        """Each line of .gitignore that leaves a name out, as a file or a folder (its trailing /) of that name: at the
        top when it is anchored there (its leading /), else at the top and deep inside the package's own entries too.
        A line that lets a name in again (its leading !) leaves nothing out.
        """
        pats = self.patterns()
        text = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for line in [x.strip() for x in text if x.strip() and x[0] not in "#!"]:
            name = line.strip("/").replace("*", "x")
            places = (
                [name]
                if line.startswith("/")
                else [name, f"docker/{name}", f"rkmon_lib/deep/{name}"]
            )
            for where in places:
                path = f"{where}/f" if line.endswith("/") else where
                with self.subTest(gitignore=line, path=path):
                    self.assertTrue(build_context.ignored(pats, path))

    def test_every_file_reaches_the_image_but_the_package_only_ones(self):
        """The image takes every file of the package but those under an entry of image.PACKAGE_ONLY (the setup
        skill's among them: tests/test_setup_skill.py), and none of those."""
        from rkmon_lib import image

        pats = self.patterns()
        files = self.tracked()
        package_only = [f for f in files if f.split("/", 1)[0] in image.PACKAGE_ONLY]
        self.assertEqual(
            [
                f
                for f in files
                if build_context.ignored(pats, f) and f not in package_only
            ],
            [],
        )
        self.assertEqual(
            [f for f in package_only if not build_context.ignored(pats, f)], []
        )

    def test_the_image_s_code_is_the_build_context_s_files(self):
        """image.code_files, the code rkmon apply installs as a release: each regular file .dockerignore lets in, the
        package's every file but the package-only ones; no link, nothing it leaves out, nothing of a folder it leaves
        out (a checkout's .git)."""
        from rkmon_lib import image

        pats = self.patterns()
        found = image.code_files(ROOT)
        self.assertEqual([f for f in found if build_context.ignored(pats, f)], [])
        here = [f for f in self.tracked() if (ROOT / f).is_file()]
        self.assertEqual(
            sorted(set(here) - set(found)),
            sorted(f for f in here if f.split("/", 1)[0] in image.PACKAGE_ONLY),
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = pathlib.Path(tmp)
            shutil.copy(ROOT / ".dockerignore", pkg / ".dockerignore")
            for rel in (
                "rkmon.py",
                "docker/rkmon",
                "docker/.env",
                "docker/__pycache__/x.pyc",
                "notes.txt",
                ".git/config",
                "config.json",
            ):
                (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
                (pkg / rel).write_text("x\n", encoding="utf-8")
            (pkg / "rkmon_lib").mkdir()
            (pkg / "rkmon_lib" / "link.py").symlink_to(pkg / "rkmon.py")
            (pkg / "rkmon_lib" / "folder").symlink_to(pkg / "docker")
            self.assertEqual(
                image.code_files(pkg), [".dockerignore", "docker/rkmon", "rkmon.py"]
            )

    def test_the_code_s_sha256(self):
        """image.code_sha256, which the settings' fingerprint covers: code_volume's digest of the code
        files (image.code_files: their paths, bytes and execute bits), so a file .dockerignore leaves out changes
        nothing, while a code file's bytes or its execute bit change it; a folder with no .dockerignore has none.
        """
        from rkmon_lib import code_volume, image

        with tempfile.TemporaryDirectory() as tmp:
            pkg = pathlib.Path(tmp)
            shutil.copy(ROOT / ".dockerignore", pkg / ".dockerignore")
            for rel in ("rkmon.py", "docker/rkmon", "rkmon_lib/a.py"):
                (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
                (pkg / rel).write_text("x\n", encoding="utf-8")
                (pkg / rel).chmod(0o644)
            code = image.code_sha256(pkg)
            files = image.code_files(pkg)
            self.assertEqual(code, "sha256:" + code_volume.digest(pkg, files))
            for rel in ("docker/.env", "notes.txt", "docker/__pycache__/x.pyc"):
                (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
                (pkg / rel).write_text("y\n", encoding="utf-8")
            self.assertEqual(image.code_sha256(pkg), code)
            # the whole tree's digest is not the code's
            self.assertNotEqual("sha256:" + code_volume.digest(pkg), code)
            (pkg / "docker" / "rkmon").chmod(0o755)
            bit = image.code_sha256(pkg)
            self.assertNotEqual(bit, code)
            (pkg / "rkmon_lib" / "a.py").write_text("z\n", encoding="utf-8")
            self.assertNotIn(image.code_sha256(pkg), (code, bit))
            (pkg / ".dockerignore").unlink()
            with self.assertRaises(OSError):
                image.code_sha256(pkg)

    def test_the_build_context_is_the_package_s_own_entries(self):
        """image.BUILD_CONTEXT, the entries .dockerignore lets in, is every entry at the package's top but those of
        image.PACKAGE_ONLY, which it leaves out: an entry the package gains is let in by adding it to BUILD_CONTEXT, or
        kept out by adding it to PACKAGE_ONLY.
        """
        from rkmon_lib import image

        top = {f.split("/", 1)[0] for f in self.tracked()}
        self.assertEqual(len(image.BUILD_CONTEXT), len(set(image.BUILD_CONTEXT)))
        self.assertEqual(set(image.BUILD_CONTEXT), top - set(image.PACKAGE_ONLY))
        self.assertEqual(set(image.BUILD_CONTEXT) & set(image.PACKAGE_ONLY), set())
        for entry in image.PACKAGE_ONLY:
            self.assertTrue(build_context.ignored(self.patterns(), entry), entry)


class TestAtlassianStage(unittest.TestCase):
    def test_the_installer_with_a_build_secret(self):
        if not (shutil.which("curl") and shutil.which("sha256sum")):
            self.skipTest("curl and sha256sum are not here")
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tests" / "container" / "stages.py"),
                    "atlassian",
                    "--home",
                    d,
                    "--port",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(ROOT / "tests" / "container"),
            )
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-2000:])
        for check in (
            "a matching binary, from the build secret's URL, is installed, and its version and SHA-256 recorded",
            "a failed download installs nothing, and says why in fixed words",
            "neither the URL nor its host is ever printed",
        ):
            self.assertIn(f"PASS  {check}", p.stdout)
        self.assertIn(
            "PASS  no URL: the build stops (every image carries the CLI), naming the parameter and its settings key",
            p.stdout,
        )


class TestOrchestrator(unittest.TestCase):
    def test_a_tick_past_its_time_ends_its_whole_group(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            marker = f"sleep 88{os.getpid()}"
            fake_rk(d, f"{marker} &\nsleep 60\n")
            with mock.patch.object(orchestrator.paths, "PACKAGE", d):
                with self.assertRaises(asyncio.TimeoutError):
                    asyncio.run(orchestrator.rk_command(["tick"], 1))
            time.sleep(0.5)
            ps = subprocess.run(
                ["ps", "-eo", "args"], capture_output=True, text=True
            ).stdout
            self.assertNotIn(marker, ps)  # the child the rk started ended with it

    def test_what_a_dead_orchestrator_left_is_found(self):
        me = os.getpid()
        table = {
            me: {"pid": me, "ppid": 5, "start": 100, "uid": os.getuid()},
            7001: {
                "pid": 7001,
                "ppid": 1,
                "start": 9,
                "uid": os.getuid(),
            },  # an orphan of a dead one
            7002: {
                "pid": 7002,
                "ppid": 1,
                "start": 9,
                "uid": os.getuid(),
            },  # of a live one (not this)
            7003: {
                "pid": 7003,
                "ppid": 7002,
                "start": 9,
                "uid": os.getuid(),
            },  # not an orphan
            7004: {
                "pid": 7004,
                "ppid": 1,
                "start": 9,
                "uid": os.getuid() + 1,
            },  # another user's
            7005: {
                "pid": 7005,
                "ppid": 1,
                "start": 9,
                "uid": os.getuid(),
            },  # this orchestrator's own
            7006: {
                "pid": 7006,
                "ppid": 1,
                "start": 9,
                "uid": os.getuid(),
            },  # another pid namespace
            6000: {
                "pid": 6000,
                "ppid": 1,
                "start": 50,
                "uid": os.getuid(),
            },  # a live orchestrator
        }
        own = orchestrator.own_mark(me)
        marks = {
            7001: "4321:77",
            7002: "6000:50",
            7003: "4321:77",
            7004: "4321:77",
            7005: own,
            7006: "4321:77",
        }
        found = orchestrator.stale_sessions(
            table,
            env=lambda pid: (
                {orchestrator.ORCH_MARK: marks[pid]} if pid in marks else {}
            ),
            me=me,
            ns=lambda pid: "pid:[2]" if pid == 7006 else "pid:[1]",
        )
        self.assertEqual([e["pid"] for e in found], [7001])
        self.assertEqual(orchestrator.session_env.__name__, "session_env")

    def test_a_killed_driver_starts_again_at_once(self):
        cfg = {"deployment": {"phase": "review"}, "repo": "o/r"}

        async def rk(args, timeout):
            return -9, "REVIEW #7: worker running\n"

        s = orchestrator.Session(cfg, clock=lambda: 0.0, say=lambda line: None)
        s.driver_wanted = False
        with mock.patch.object(orchestrator, "load_state", lambda cfg: {}):
            asyncio.run(orchestrator.drive(cfg, s, rk))
        self.assertTrue(s.driver_wanted)
        s.driver_wanted = False

        async def idle(args, timeout):
            return 0, "DRIVER: nothing to start now\n"

        asyncio.run(orchestrator.drive(cfg, s, idle))
        self.assertFalse(
            s.driver_wanted
        )  # an idle driver waits for the next tick, as before

    def test_the_sessions_key_is_the_claude_login(self):
        """The session's key is the claude login's file, the key alone (orchestrator.read_key), sent as the deploy
        settings say (a bearer token here: orchestrator.session_variables); while it is set the orchestrator lacks
        no key, and without it the orchestrator names the command that sets it. An older claude_env beside it is
        never read."""
        with tempfile.TemporaryDirectory() as d:
            key = pathlib.Path(d) / "claude_key"
            key.write_text("t0k\n")
            (pathlib.Path(d) / "claude_env").write_text(
                json.dumps({"ANTHROPIC_AUTH_TOKEN": "older"})
            )
            cfg = {
                "claude": {
                    "gateway": "https://gw.example.invalid",
                    "api_key_sent_as": "bearer",
                }
            }

            def missing() -> str:
                with mock.patch.object(
                    orchestrator.features, "on", lambda cfg, name: False
                ), mock.patch.object(
                    orchestrator, "settings", lambda cfg: {"workspace": ""}
                ):
                    return " ".join(orchestrator.check(cfg))

            with mock.patch.object(orchestrator, "key_path", lambda cfg: key):
                self.assertEqual(orchestrator.read_key(cfg), "t0k")
                self.assertEqual(
                    orchestrator.session_variables(cfg, orchestrator.read_key(cfg)),
                    {
                        "ANTHROPIC_BASE_URL": "https://gw.example.invalid",
                        "ANTHROPIC_AUTH_TOKEN": "t0k",
                    },
                )
                self.assertNotIn("no key", missing())
                key.unlink()
                self.assertEqual(orchestrator.read_key(cfg), "")
                self.assertIn(
                    "the Claude session has no key (the login claude: ", missing()
                )


class TestDoctorInTheContainer(unittest.TestCase):
    def test_no_prerequisite_row_is_for_services(self):
        """The monitor has no services of its own: rk doctor asks for neither systemd nor
        a WSL distribution's settings for it.
        """
        names = [r.name for r in self.bare_prerequisites()]
        self.assertIn("gh (GitHub CLI)", names)
        self.assertFalse(
            [n for n in names if "systemd" in n or "service" in n or n == "WSL"], names
        )

    @staticmethod
    def bare_prerequisites(cfg=None, wrapper="rkmon") -> list:
        """rk doctor's prerequisite rows for CFG in the container, its wrapper named WRAPPER, with nothing of this
        machine's: no tool on PATH and none that runs, a large disk, no Claude Code and no ROCm record.
        """
        usage = collections.namedtuple("usage", "total used free")
        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": wrapper}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            doctor, "run", lambda *a, **k: (127, "")
        ), mock.patch.object(
            doctor.shutil, "which", lambda *a, **k: None
        ), mock.patch.object(
            doctor.shutil, "disk_usage", lambda p: usage(2**40, 2**40, 2**30)
        ), mock.patch.object(
            doctor, "claude_code", lambda *a, **k: ("", "unknown")
        ), mock.patch.object(
            doctor, "rocm_image", lambda: {}
        ):
            return doctor.prerequisites(cfg)

    def test_each_prerequisites_fix_names_only_rkmon_commands(self):
        """In the container everything prerequisites.txt lists is the image's own: a missing one's fix runs an image
        this package built (doctor.image_fix) or checks the image (doctor.image_lacks_fix), and no fix names a host's
        installer, gh's private copy or a command that runs anywhere but through rkmon.
        """
        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"}
        rows = self.bare_prerequisites()
        fixed = [r for r in rows if r.fix is not None]
        self.assertGreater(len(fixed), 10, [r.name for r in rows])
        for r in fixed:
            for c in r.fix.commands:
                self.assertTrue(c.startswith("rkmon "), (r.name, c))
        gh = next(r for r in rows if r.name == "gh (GitHub CLI)")
        self.assertEqual(str(gh.fix), str(doctor.image_fix(env)))
        self.assertNotIn("checksum", str(gh.fix))
        with mock.patch.dict(os.environ, env):  # ... and the GitHub service row's
            (row,) = reach.github({}, run=lambda cmd, env=None: (127, ""))
        self.assertEqual(str(row.fix), str(doctor.image_fix(env)))

    def test_each_github_row_logs_the_read_login_in_by_its_own_command(self):
        """A GitHub row whose fix logs the read login in names that login's own command (doctor.gh_login_fix): gh's
        prerequisite row whose login lacks a scope the features need, and the GitHub service row with no login.
        """
        login = "./rkmon login github_read"
        g = {
            "found": True,
            "version_ok": True,
            "logged_in": True,
            "login": "op-login",
            "scopes": ["gist"],
            "binary": "gh",
            "version": "2.46.0",
            "config_dir": "/w/.config/gh",
        }
        with mock.patch.object(
            doctor, "gh_status", lambda *a, **k: g
        ), mock.patch.object(
            doctor, "gh_needed_scopes", lambda cfg: ["repo"]
        ), mock.patch.object(
            doctor.gh_mod, "login_dirs", lambda cfg: {"read": ""}
        ):
            rows = self.bare_prerequisites(wrapper="./rkmon")
        gh = next(r for r in rows if r.name == "gh (GitHub CLI)")
        self.assertEqual(
            str(gh.fix),
            f"{login} sets up the read login (a read-only token), which reads without those scopes",
        )
        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        with mock.patch.dict(os.environ, env):
            (row,) = reach.github({}, run=lambda cmd, env=None: (1, "not logged in"))
        self.assertEqual(
            str(row.fix),
            f"{login} (the read login, a read-only token: the one gh uses here)",
        )

    def test_rocm_s_llvm_tools_are_the_image_s(self):
        """In the container ROCm's LLVM tools are the image's (paths.rocm_llvm_bin, derived from its ROCm record): a
        missing or broken one is the image's, whose fix checks the image.
        """
        name = doctor.PREREQS["llvm"].name
        with tempfile.TemporaryDirectory() as d:
            for llvm, found in (
                ("", "missing: the image records none (paths.rocm_llvm_bin is blank)"),
                (d, f"no clang in {d} (paths.rocm_llvm_bin)"),
            ):
                with self.subTest(llvm=llvm):
                    rows = self.bare_prerequisites(
                        {"paths": {"rocm_llvm_bin": llvm}}, wrapper="./rkmon"
                    )
                    row = next(r for r in rows if r.name == name)
                    self.assertTrue(row.found.startswith(found), row.found)
                    self.assertEqual(
                        str(row.fix),
                        "./rkmon image check (the image lacks ROCm's LLVM tools)",
                    )

    def test_the_ollama_rows_fix_sets_the_key_of_its_address(self):
        """In the container the Ollama row points the reviewer model at an Ollama the container reaches at the key
        the deploy settings give an Ollama's address, never at the binding as a whole; the reviewer model's rows name
        the key of its provider's address."""
        env = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon"}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            doctor, "run", lambda *a, **k: (127, "")
        ), mock.patch.object(
            doctor, "claude_code", lambda *a, **k: ("1.0 (here)", "logged in")
        ):
            (row,) = [
                r for r in doctor.prerequisites(None) if r.name.startswith("Ollama")
            ]
        self.assertIn(
            "set model_bindings.worker.server_url in your deploy settings file to the address of that Ollama, then "
            "rkmon confirm",
            str(row.fix),
        )
        for backend, key in (
            ("ollama", "model_bindings.worker.server_url"),
            ("openai", "model_bindings.worker.base_url"),
        ):
            with self.subTest(backend=backend):
                w = dict(config.worker({}), backend=backend, url="", model="m")
                self.assertTrue(
                    doctor.model_check({}, w).found.startswith(
                        f"{key} or model_id is blank"
                    )
                )

    def test_each_container_fix_edits_a_key_the_deploy_settings_file_has(self):
        """In the container a fix that sets a configuration key edits the deploy settings file (doctor.settings_parts,
        features.Need.setting's and features.problems' edits) at a key that file has: deploy_tree.node finds it. The
        keys come from the package's code, each call that names one: a literal one as it is, a computed one by the keys
        it takes (COMPUTED); the deploy settings key a call gives in its place, the same way; a function that hands its
        own parameter on is checked at its callers. No fix edits a key the deploy settings file does not have.
        """
        from rkmon_lib import deploy_tree

        # the function -> (the module it is in, None for any; the place of its configuration key, and its keyword; the
        # place and keyword of the deploy settings key it may be given in that key's place)
        sites = {
            "settings_parts": (None, 0, "internal", 3, "key"),
            "settings_fix": (None, 0, "internal"),
            "_settings_fix": ("provision.py", 0, "internal"),
            "setting": (None, 0, "internal", 2, "key"),  # Need.setting
            "_Cause": ("provision.py", 2, "key"),
            "edit": ("features.py", 0, "internal"),  # features.problems'
            "source": ("features.py", 0, "internal"),
        }
        worker_urls = sorted(
            {
                doctor.worker_url_key({"backend": b})
                for b in deploy_tree.WORKER_BACKEND.values()
            }
        )
        computed = {  # (module, a computed key as written) -> its configuration keys, or the file keys a KEY= names
            ("doctor.py", "worker_url_key(w)"): worker_urls,
            ("features.py", "internal"): features._team_dashboard_settings({}),
            ("features.py", "k"): features._team_dashboard_settings({}),
            (
                "features.py",
                "self.file_key",
            ): [],  # Need.setting's: checked at its calls
            ("gpu.py", "f'gpu.{k}'"): [
                f"gpu.{k}"
                for k in (
                    "login_host",
                    "ssh_user",
                    "partition",
                    "remote_python",
                    "accounts",
                    "gres",
                )
            ],
            ("triage.py", "doctor.worker_url_key(w)"): worker_urls,
            ("triage.py", "doctor.url_key(name, s.get('kind') or '')"): [
                doctor.url_key("gw", kind)
                for kind in ("ollama", "openai-compatible", "claude")
            ],
            ("triage.py", "f'model_bindings.{name}.model'"): [
                "model_bindings.gw.model"
            ],
            ("provision.py", "cause.key"): [],  # a _Cause's: checked at its calls
        }
        seen, wrong, unknown, used = 0, [], [], set()

        def argument(call, place, keyword):
            if len(call.args) > place:
                return call.args[place]
            return next((k.value for k in call.keywords if k.arg == keyword), None)

        def keys(module, node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return [node.value]
            written = (module, ast.unparse(node))
            if written not in computed:
                unknown.append(written)
                return []
            used.add(written)
            return computed[written]

        for path in sorted((ROOT / "rkmon_lib").glob("*.py")):
            module, stack = path.name, []

            def visit(node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stack.append(node)
                if isinstance(node, ast.Call):
                    check(node)
                for child in ast.iter_child_nodes(node):
                    visit(child)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stack.pop()

            def check(call):
                nonlocal seen
                f = call.func
                name = getattr(f, "id", None) or getattr(f, "attr", "")
                site = sites.get(name)
                if not site or site[0] not in (None, module):
                    return
                if name == "setting" and getattr(f.value, "id", "") != "Need":
                    return
                arg = argument(call, *site[1:3])
                own = stack[-1] if stack else None
                if (
                    isinstance(arg, ast.Name)
                    and own is not None
                    and own.name in sites
                    and arg.id in [a.arg for a in own.args.args]
                ):
                    return  # handed on: checked at the callers
                seen += 1
                given = argument(call, *site[3:]) if len(site) > 3 else None
                if given is not None:
                    found = [(k, k) for k in keys(module, given)]
                else:
                    found = [(k, deploy_tree.name(k)) for k in keys(module, arg)]
                for internal, key in found:
                    if not deploy_tree.node(key)[0]:
                        wrong.append(f"{module}:{call.lineno}: {internal} -> {key}")

            visit(ast.parse(path.read_text(encoding="utf-8")))
        self.assertGreater(seen, 20)
        self.assertEqual(
            unknown,
            [],
            "a computed key this test does not know: add its keys to COMPUTED",
        )
        self.assertEqual(
            sorted(set(computed) - used), [], "COMPUTED names a call that is gone"
        )
        self.assertEqual(wrong, [])

    def test_the_claude_login_is_the_key_the_session_reads(self):
        """Only the claude login counts, the key the Claude session reads: a sign-in where Claude Code
        keeps one, or a key in the environment, it never reads counts for nothing."""
        with tempfile.TemporaryDirectory() as d:
            conf = pathlib.Path(d) / "config"
            conf.mkdir()
            (conf / ".credentials.json").write_text("{}")
            env = {
                "CLAUDE_CONFIG_DIR": str(conf),
                "ANTHROPIC_API_KEY": "t",
                "ANTHROPIC_AUTH_TOKEN": "t",
            }
            with mock.patch.object(
                doctor, "run", lambda *a, **k: (0, "2.1.0 (Claude Code)")
            ), mock.patch.dict(os.environ, env):
                for key, said in (("", "not logged in"), ("t", "logged in")):
                    with self.subTest(key=key), mock.patch.object(
                        orchestrator, "read_key", lambda cfg: key
                    ):
                        self.assertEqual(
                            doctor.claude_code({"x": 1}), ("2.1.0 (here)", said)
                        )

    def test_an_rk_command_is_named_as_the_operator_types_it(self):
        """An rk command in a fix is the operator's line on the host (doctor.rk_command), wherever the monitor runs:
        through the wrapper as it was called (`<wrapper> rk ...`), never rk alone. The service checks' check again is
        the checks' own (doctor.check_again) with --fresh.
        """
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        self.assertEqual(str(doctor.rk_command("tick", environ=box)), "./rkmon rk tick")
        self.assertEqual(str(doctor.rk_command("tick", environ={})), "rkmon rk tick")
        self.assertEqual(str(reach.again(box)), "./rkmon doctor --fresh")
        self.assertEqual(str(reach.again({})), "rkmon doctor --fresh")
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, box):
            (pathlib.Path(d) / "actions.jsonl").write_text('{"a": 1}\n{"a": 2')
            rows = doctor.state_checks(pathlib.Path(d))
            rows += reach.gpu_cluster(
                {"gpu": {"enabled": True, "login_host": "<host>"}},
                check=lambda cfg: ["FAIL the login host did not answer"],
            )
        shown = checks.render(rows)
        self.assertIn("fix: ./rkmon rk tick writes the next record", shown)
        self.assertIn("fix: ./rkmon rk gpu-check says more;", shown)

    def test_every_rk_command_in_a_fix_comes_from_rk_command(self):
        """No fix (a Command) and no retry (private_files.again's command) spells out a bare rk command: each names
        rk through doctor.rk_command, the one place that knows how the operator types it here.
        """
        bare = []
        for f in [ROOT / "rkmon.py", *sorted((ROOT / "rkmon_lib").glob("*.py"))]:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            home = {
                id(n)
                for fn in ast.walk(tree)
                if isinstance(fn, ast.FunctionDef) and fn.name == "rk_command"
                for n in ast.walk(fn)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args or id(node) in home:
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", "")
                argv = node.args[0]
                if (
                    name in ("Command", "again")
                    and isinstance(argv, (ast.Tuple, ast.List))
                    and argv.elts
                    and isinstance(argv.elts[0], ast.Constant)
                    and argv.elts[0].value == "rk"
                ):
                    bare.append(f"{f.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(bare, [], "rk named bare: use doctor.rk_command")

    def test_the_fixes_as_the_operator_types_them(self):
        """A fix is the command the operator types on the host, wherever the monitor runs: the container's own (the
        wrapper, as it was called), here a secret's own (secrets_registry.set_fix). A capability's fix sets what it
        lacks, each by its own fix: a login's or a secret's command, a setting's edit of the deploy settings file (its
        section's, for the settings the section holds), the image's check for a part the image carries; with nothing
        lacking, the checks run again.
        """
        from rkmon_lib import secrets_registry as sr

        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}
        web = "secret admin_access_token --generate"
        token = sr.entry({}, "admin_access_token")
        self.assertEqual(str(sr.set_fix(token, box)), f"./rkmon {web}")
        self.assertEqual(str(sr.set_fix(token, {})), f"rkmon {web}")
        with mock.patch.object(program, "name", return_value="/opt/pkg/docker/rkmon"):
            self.assertEqual(
                str(sr.set_fix(token, {"RKMON_CONTAINER": "1"})),
                f"/opt/pkg/docker/rkmon {web}",
            )  # no name passed in: the program's own
        cfg = config.load_defaults()
        self.assertEqual(
            str(features.fix(cfg, "develop-checks", environ={})),
            "rkmon image check (the image lacks the checks venv and ROCm's LLVM tools)",
        )
        # In the container the Claude session has no setup section: its key is the claude login.
        ws = dict(
            cfg,
            claude_workspace="/ws",
            features=dict(
                cfg["features"],
                review_automation=dict(
                    cfg["features"]["review_automation"], enabled=True
                ),
            ),
        )
        for held, fix in (
            (
                False,
                "./rkmon login claude, and give a token on one line, with no spaces",
            ),
            (True, "./rkmon doctor"),  # nothing lacking: the checks run again
        ):
            with self.subTest(held=held), mock.patch.dict(os.environ, box), mock.patch(
                "shutil.which", return_value="/usr/bin/claude"
            ), mock.patch.object(
                features,
                "_secret_found",
                lambda cfg, name: {"set": held, "where": f"/s/{name}"},
            ):
                self.assertEqual(
                    str(features.fix(ws, "headless-claude", environ=box)), fix
                )
        # Outside the container no Claude session runs, so no setup section sets one up.
        self.assertNotIn(
            "services", str(features.fix(cfg, "headless-claude", environ={}))
        )
        cfg["features"]["merge_queue"].update(
            enabled=True, mode="live", auto_merge="enabled"
        )
        with tempfile.TemporaryDirectory() as d, mock.patch(
            "shutil.which", return_value=None
        ):
            cfg["paths"]["working_dir"] = d
            for env, w in ((box, "./rkmon"), ({}, "rkmon")):
                self.assertEqual(
                    str(features.fix(cfg, "auto-merge", environ=env)),
                    "set integrations.github_cli.username in your deploy settings file to the GitHub login the "
                    f"monitor posts as, then {w} confirm and {w} apply; then {w} image check (the image lacks the "
                    "gh CLI)",
                )


if __name__ == "__main__":
    unittest.main()

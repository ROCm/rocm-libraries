# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/tick_loop.py with a stand-in rk: the wake, the heartbeat, a failing or hanging tick, one loop at a time,
standing lines across loops, and a stop signal. The real `rk tick-filter` reads each tick's output.
"""

import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import tickfilter  # noqa: E402

LOOP = ROOT / "tools" / "tick_loop.py"
FAKE_RK = """#!/bin/bash
# A stand-in rk: loop-config from loop.conf, `tick` prints tick<N> (else tick_default), with tick<N>.rc as its exit code
# and tick<N>.sleep seconds first (tick<N>.pre printed before them); tick-filter is the real one.
D=$(dirname "$(readlink -f "$0")")
case "$1" in
  loop-config) cat "$D/loop.conf" ;;
  tick)
    n=$(( $(cat "$D/count" 2>/dev/null || echo 0) + 1 )); echo $n > "$D/count"
    f="$D/tick$n"; [ -f "$f" ] || f="$D/tick_default"
    [ -f "$f.pre" ] && cat "$f.pre"
    [ -f "$f.sleep" ] && sleep "$(cat "$f.sleep")"
    cat "$f"
    touch "$D/done$n"
    exit "$(cat "$f.rc" 2>/dev/null || echo 0)" ;;
  tick-filter) shift; exec python3 -B "$REAL_ROOT/rkmon.py" tick-filter "$@" ;;
  *) echo "fake rk: $*"; exit 9 ;;
esac
"""
IDLE = "TICK idle: no change since 18:20Z (5 PRs, develop abcdef0123)\n"
FULL = (
    "TICK 2026-11-06T18:30:00Z monitor ON reviews=live merge-queue=live auto-merge=hints_only\n"
    "develop abcdef0123 CI pending; last green 0123456789 (ci)\nEVENTS: none\n{extra}BOARD unchanged since "
    "2026-11-06T18:00:00Z\ntime: fingerprint 3s\n"
)
PROPOSE = (
    "PROPOSE #11: validated on develop abcdef0123; tell the user to merge in this order"
)


class TestWait(unittest.TestCase):
    def test_the_wait_is_never_more_than_one_interval(self):
        """A clock stepped back must not stretch the wait past the heartbeat."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("tick_loop", LOOP)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.next_wait(1000, 600, 1100), 500)
        self.assertEqual(mod.next_wait(1000, 600, 1700), 0)  # the tick ran long
        self.assertEqual(
            mod.next_wait(1000, 600, 100), 600
        )  # the clock went back 900 s


@unittest.skipIf(
    sys.platform != "linux", "the loop runs on Linux (flock, timeout, GNU date)"
)
class TestTickLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        rk = self.d / "rk"
        rk.write_text(FAKE_RK)
        rk.chmod(0o755)
        self.conf(interval=0, heartbeat=3600, timeout=60)
        (self.d / "tick_default").write_text(IDLE)

    def tearDown(self):
        self.tmp.cleanup()

    def conf(self, interval, heartbeat, timeout):
        files = {
            k: self.d / v
            for k, v in (
                ("LOOP_LOCK", "tick_loop.lock"),
                ("LOOP_LOG", "tick_loop.log"),
                ("LOOP_PID", "tick_loop.pid"),
                ("LOOP_LAST", "tick_loop.last"),
                ("LOOP_SEEN", "tick_loop.seen"),
                ("TICK_NOW", "tick_now"),
            )
        }
        (self.d / "loop.conf").write_text(
            f"TICK_INTERVAL_S={interval}\nLOOP_HEARTBEAT_S={heartbeat}\nTICK_TIMEOUT_S={timeout}\n"
            "LOOP_LOG_MAX_BYTES=100000\n"
            + "".join(f"{k}={v}\n" for k, v in files.items())
        )

    def ticks(self, *outputs):
        for i, out in enumerate(outputs, 1):
            (self.d / f"tick{i}").write_text(out)

    def loop(self, timeout=60, background=False, env=None):
        env = dict(
            os.environ,
            TICK_LOOP_RK=str(self.d / "rk"),
            REAL_ROOT=str(ROOT),
            **env or {},
        )
        cmd = [sys.executable, "-B", str(LOOP)]
        if background:
            return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
        return subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout
        )

    def wait_for_pid(self):
        for _ in range(200):
            if (self.d / "tick_loop.pid").exists():
                return
            time.sleep(0.05)
        self.fail("the loop never wrote its pid file")

    def lock_free(self) -> bool:
        with open(self.d / "tick_loop.lock", "rb") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            return True

    def test_idle_ticks_wake_nobody_and_a_line_to_act_on_does(self):
        self.ticks(
            IDLE,
            IDLE,
            FULL.format(extra=f"{PROPOSE}\nREVIEW #5: worker running, checks none\n"),
        )
        r = self.loop()
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.splitlines()
        self.assertTrue(out[0].startswith("LOOP_WAKE "), out)
        self.assertIn(
            "(exit 0) needs the orchestrator for the 1 line below",
            out[0],
        )
        self.assertEqual(out[1:], [PROPOSE])
        log = (self.d / "tick_loop.log").read_text()
        self.assertEqual(log.count("=== "), 3)
        self.assertTrue((self.d / "tick_loop.last").read_text().strip().isdigit())
        self.assertFalse((self.d / "tick_loop.pid").exists())  # removed on exit

    def test_the_heartbeat_and_the_standing_lines_across_loops(self):
        self.ticks(FULL.format(extra=PROPOSE + "\n"))
        self.assertEqual(self.loop().stdout.splitlines()[1:], [PROPOSE])
        # The next loop: the same proposal stands, so it wakes nobody; the heartbeat repeats it.
        (self.d / "count").write_text("0")
        self.conf(interval=0, heartbeat=0, timeout=60)
        out = self.loop().stdout.splitlines()
        self.assertTrue(out[0].startswith("LOOP_HEARTBEAT "), out)
        self.assertIn("1 ticks since ", out[0])
        self.assertIn("(0 idle, 1 full, 0 skipped while another tick ran)", out[0])
        self.assertEqual(out[1:], ["Still standing (reported before):", PROPOSE])

    def test_a_tick_skipped_while_another_runs_counts_as_skipped(self):
        """The loop reads no rkmon_lib module: its copy of the words rk tick prints when it finds another tick
        running (tickfilter.TICK_RUNNING) is checked here, by the heartbeat's count."""
        skipped = f"TICK 2026-11-06T18:30:00Z {tickfilter.TICK_RUNNING}; skipped\n"
        self.ticks(skipped)
        self.conf(interval=0, heartbeat=0, timeout=60)
        out = self.loop().stdout.splitlines()
        self.assertEqual(len(out), 1, out)
        self.assertTrue(out[0].startswith("LOOP_HEARTBEAT "), out)
        self.assertIn("(0 idle, 0 full, 1 skipped while another tick ran)", out[0])
        self.assertTrue(out[0].endswith(f"; the last: {skipped.strip()}"), out)

    def test_a_failing_tick_wakes_with_its_exit_code(self):
        self.ticks("!! CRITICAL TOOL failure: gh api graphql: HTTP 502\n")
        (self.d / "tick1.rc").write_text("2")
        out = self.loop().stdout.splitlines()
        self.assertIn("(exit 2)", out[0])
        self.assertEqual(
            out[1:],
            [
                "TICK_EXIT 2: a tool the tick needs (GitHub, git) failed",
                "!! CRITICAL TOOL failure: gh api graphql: HTTP 502",
            ],
        )

    def test_a_hung_tick_is_stopped_and_the_lock_is_free_again(self):
        self.conf(interval=0, heartbeat=3600, timeout=1)
        self.ticks("never printed\n")
        (self.d / "tick1.sleep").write_text("30")
        t0 = time.monotonic()
        r = self.loop()
        self.assertLess(time.monotonic() - t0, 20)
        self.assertTrue(r.stdout.startswith("LOOP_WAKE "), r.stdout)
        self.assertIn("ran past tick_timeout_min", r.stdout)
        self.conf(
            interval=0, heartbeat=0, timeout=60
        )  # a new loop starts: the lock went with the old one
        self.assertTrue(self.loop().stdout.startswith("LOOP_HEARTBEAT "))

    def test_a_tick_stopped_while_gh_calls_wait_for_the_cap_names_the_cap(self):
        """Its gh call's wait is open in the GitHub request cap's record (rkmon_lib/gh_cap.py), so the stop line says
        what the tick waited for."""
        state = self.d / "home" / "state"
        state.mkdir(parents=True)
        now = time.time()
        (state / "gh_requests.json").write_text(
            json.dumps(
                {
                    "cap_per_hour": 3,
                    "requests_by_minute": {},
                    "waits": [],
                    "open_waits": {"k": {"since": now - 600, "looked": now}},
                }
            )
        )
        self.conf(interval=0, heartbeat=3600, timeout=1)
        self.ticks("never printed\n")
        (self.d / "tick1.sleep").write_text("30")
        out = self.loop(env={"RKMON_HOME": str(self.d / "home")}).stdout.splitlines()
        self.assertIn("ran past tick_timeout_min", out[0])
        first = time.strftime("%H:%MZ", time.gmtime(now - 600))
        self.assertIn(
            ") while 1 gh call waited for the GitHub request cap (integrations.github_cli.max_requests_per_hour), the "
            f"first since {first}; its lines for the orchestrator so far",
            out[0],
        )

    def test_a_hung_ticks_lines_to_act_on_are_in_the_wake(self):
        """What a stopped tick printed before it hung: its lines to act on, then its last lines, so the orchestrator
        never reads the loop log."""
        self.conf(interval=0, heartbeat=3600, timeout=1)
        self.ticks("never printed\n")
        (self.d / "tick1.pre").write_text(
            f"TICK 2026-11-06T18:30:00Z monitor ON phase=queue\n{PROPOSE}\nEVENTS: none\n"
        )
        (self.d / "tick1.sleep").write_text("30")
        out = self.loop().stdout.splitlines()
        self.assertIn("ran past tick_timeout_min", out[0])
        self.assertEqual(out[1], PROPOSE)
        self.assertEqual(
            out[-3:],
            [
                "  | TICK 2026-11-06T18:30:00Z monitor ON phase=queue",
                f"  | {PROPOSE}",
                "  | EVENTS: none",
            ],
        )

    def test_one_loop_at_a_time(self):
        self.conf(interval=0, heartbeat=0, timeout=60)
        (self.d / "tick_default.sleep").write_text("3")
        first = self.loop(background=True)
        try:
            for _ in range(100):
                if (self.d / "tick_loop.pid").exists():
                    break
                time.sleep(0.05)
            r = self.loop()
            self.assertEqual(r.returncode, 2)
            self.assertTrue(
                r.stdout.startswith(
                    "LOOP_RUNNING: a tick loop is already running (pid "
                ),
                r.stdout,
            )
        finally:
            out, _ = first.communicate(timeout=30)
        self.assertTrue(out.startswith("LOOP_HEARTBEAT "), out)

    def test_a_restarted_loop_keeps_the_interval(self):
        self.conf(interval=2, heartbeat=0, timeout=60)
        (self.d / "tick_loop.last").write_text(str(int(time.time())))
        t0 = time.monotonic()
        self.loop()
        self.assertGreaterEqual(time.monotonic() - t0, 1.0)

    def test_rk_tick_now_cuts_the_wait_short(self):
        """An idle tick, then a wait of a whole interval; rk tick-now's request starts the next tick within seconds,
        which takes the request, and the log names it."""
        self.conf(interval=60, heartbeat=3600, timeout=60)
        self.ticks(IDLE, FULL.format(extra=PROPOSE + "\n"))
        first = self.loop(background=True)
        try:
            for _ in range(200):  # the first tick ran; the loop waits for the next one
                if (self.d / "done1").exists():
                    break
                time.sleep(0.05)
            t0 = time.monotonic()
            (self.d / "tick_now").write_text("2026-11-23T10:00:00Z\n")
            out, _ = first.communicate(timeout=20)
        finally:
            if first.poll() is None:
                first.kill()
                first.communicate()
        self.assertLess(time.monotonic() - t0, 15)
        self.assertEqual(out.splitlines()[1:], [PROPOSE])
        self.assertFalse((self.d / "tick_now").exists())
        log = (self.d / "tick_loop.log").read_text()
        self.assertEqual(log.count("=== "), 2)
        self.assertIn(" rk tick (rk tick-now asked for it): exit 0\n", log)

    def test_a_request_at_the_start_runs_the_first_tick_at_once(self):
        """A restarted loop keeps the interval, but not while rk tick-now's request waits."""
        self.conf(interval=60, heartbeat=0, timeout=60)
        (self.d / "tick_loop.last").write_text(str(int(time.time())))
        (self.d / "tick_now").write_text("2026-11-23T10:00:00Z\n")
        t0 = time.monotonic()
        r = self.loop(timeout=20)
        self.assertLess(time.monotonic() - t0, 15)
        self.assertTrue(r.stdout.startswith("LOOP_HEARTBEAT "), r.stdout)
        self.assertFalse((self.d / "tick_now").exists())

    def test_a_request_that_cannot_be_taken_leaves_the_interval_as_it_is(self):
        """A folder at the request's path is never taken, so it ends no wait: the ticks keep the interval (one every
        2 s for 3 s: three or four of them), not one after another until the heartbeat.
        """
        self.conf(interval=2, heartbeat=3, timeout=60)
        (self.d / "tick_now").mkdir()
        r = self.loop(timeout=30)
        self.assertTrue(r.stdout.startswith("LOOP_HEARTBEAT "), r.stdout)
        self.assertLessEqual(int((self.d / "count").read_text()), 4)
        self.assertNotIn(
            "rk tick-now asked for it", (self.d / "tick_loop.log").read_text()
        )
        self.assertTrue((self.d / "tick_now").is_dir())

    def test_a_failing_loop_config_wakes_and_runs_no_tick(self):
        (self.d / "loop.conf").unlink()  # the stand-in's `cat` fails: exit 1
        r = self.loop()
        self.assertEqual(r.returncode, 1)
        self.assertTrue(
            r.stdout.startswith(
                "LOOP_WAKE: rk loop-config failed, so no tick ran: cat: "
            ),
            r.stdout,
        )
        self.assertFalse((self.d / "count").exists())

    def test_a_stop_signal_between_ticks_ends_the_loop_and_removes_its_pid(self):
        self.conf(interval=60, heartbeat=3600, timeout=60)
        first = self.loop(background=True)
        self.wait_for_pid()
        for _ in range(100):  # the first tick ran; the loop waits for the next one
            if (self.d / "done1").exists():
                break
            time.sleep(0.05)
        time.sleep(0.3)
        first.send_signal(signal.SIGTERM)
        out, _ = first.communicate(timeout=30)
        self.assertEqual(
            first.returncode, -signal.SIGTERM
        )  # killed by the signal, as the shell loop was
        self.assertEqual(out, "")
        self.assertFalse((self.d / "tick_loop.pid").exists())
        self.assertTrue(self.lock_free())

    def test_a_stop_signal_during_a_tick_leaves_the_tick_to_finish(self):
        self.conf(interval=60, heartbeat=3600, timeout=60)
        (self.d / "tick_default.sleep").write_text("2")
        first = self.loop(background=True)
        self.wait_for_pid()
        time.sleep(0.5)
        first.send_signal(signal.SIGTERM)
        out, _ = first.communicate(timeout=30)
        self.assertEqual((first.returncode, out), (-signal.SIGTERM, ""))
        self.assertFalse((self.d / "tick_loop.pid").exists())
        self.assertFalse((self.d / "done1").exists())
        for _ in range(
            100
        ):  # the tick (under timeout, in its own process group) runs on to its end
            if (self.d / "done1").exists():
                break
            time.sleep(0.05)
        self.assertTrue((self.d / "done1").exists())
        self.assertNotIn(
            "===",
            (
                (self.d / "tick_loop.log").read_text()
                if (self.d / "tick_loop.log").exists()
                else ""
            ),
        )


if __name__ == "__main__":
    unittest.main()

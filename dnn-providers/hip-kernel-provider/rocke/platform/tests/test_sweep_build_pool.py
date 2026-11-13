# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the sweep build pool's tolerance of a dead worker.

``pool.map`` cannot survive a worker that dies instead of returning: the
result never arrives, so the whole stage waits on it forever with no output
and no CPU. That is indistinguishable from a slow build until you sample CPU
time, and a real backend abort produced exactly that -- a sweep frozen at a
fixed build count, twice, before the cause was found.

These tests kill a worker outright (SIGKILL, the harshest case -- no exception
to propagate, no cleanup) and assert the stage still finishes with every record
accounted for.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import unittest


def _work(n: int, suicide_at: int | None = None):
    return [
        (
            f"h{i}",
            {"name": ("suicide" if i == suicide_at else f"k{i}"), "tile": {}},
            "/tmp",
            "isa",
            "gfx1250",
        )
        for i in range(n)
    ]


def _fake_build(item):
    """Stand-in for a real build; the marked spec kills its own worker."""
    import rocke.sweep as sw

    _h, spec, _c, _i, _a = item
    if spec["name"] == "suicide":
        os.kill(os.getpid(), signal.SIGKILL)
    if spec["name"] == "slow":
        time.sleep(0.8)
    time.sleep(0.02)
    return sw.asdict(sw.BuildRecord(name=spec["name"], spec_dict=spec, ok=True))


# The pool's workers must inherit the patched _build_one, and _fake_build kills
# its worker with SIGKILL: both need the fork start method, which Windows lacks.
@unittest.skipUnless(
    "fork" in mp.get_all_start_methods(), "needs the fork start method"
)
class TestSweepBuildPool(unittest.TestCase):
    def _collect(self, work, stall_timeout):
        import rocke.sweep as sw

        original = sw._build_one
        sw._build_one = _fake_build
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(4) as pool:
                return sw._collect_builds(pool, work, stall_timeout=stall_timeout)
        finally:
            sw._build_one = original

    def test_dead_worker_does_not_hang_the_stage(self):
        """The stage finishes with every record, in the order of its specs.

        One run checks both, since the watchdog's stall_timeout is most of its
        time. 2 s is about a hundred times a fake build; a healthy build that
        made no progress for that long on a loaded machine would be written
        off, which the 7-of-8 count below would catch.
        """
        work = _work(8, suicide_at=3)
        t0 = time.monotonic()
        out = self._collect(work, stall_timeout=2.0)
        elapsed = time.monotonic() - t0

        self.assertEqual(len(out), len(work), "records were lost")
        # Bounded by the watchdog, not unbounded. Generous upper bound so a
        # loaded machine does not make this flaky.
        self.assertLess(elapsed, 60.0, "the pool hung instead of giving up")
        ok = [d for d in out if d["ok"]]
        bad = [d for d in out if not d["ok"]]
        self.assertEqual(len(ok), 7)
        self.assertEqual(len(bad), 1)
        self.assertIn("no result", bad[0]["error"])
        # Results stay aligned with the specs that produced them.
        self.assertEqual(out[3]["name"], "suicide")
        self.assertFalse(out[3]["ok"])
        for i in (0, 1, 2, 4, 5, 6, 7):
            self.assertEqual(out[i]["name"], f"k{i}")
            self.assertTrue(out[i]["ok"])

    def test_clean_run_is_unaffected(self):
        """No worker dies: every record is a success and nothing waits."""
        work = _work(8)
        t0 = time.monotonic()
        out = self._collect(work, stall_timeout=30.0)
        self.assertLess(
            time.monotonic() - t0, 20.0, "a clean run paid the watchdog cost"
        )
        self.assertEqual(len(out), 8)
        self.assertTrue(all(d["ok"] for d in out))

    def test_no_watchdog_waits_for_a_slow_build(self):
        """stall_timeout=None never writes a healthy build off."""
        work = _work(4)
        slow = (work[2][0], {"name": "slow", "tile": {}}, *work[2][2:])
        work[2] = slow
        # With a watchdog shorter than the build, the slow build is lost ...
        out = self._collect(work, stall_timeout=0.2)
        self.assertFalse(out[2]["ok"])
        # ... and without one it completes.
        out = self._collect(work, stall_timeout=None)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(d["ok"] for d in out), out)

    def test_watchdog_is_off_by_default(self):
        """Existing build_all_instances callers keep their behaviour."""
        import inspect

        import rocke.sweep as sw

        default = (
            inspect.signature(sw.build_all_instances)
            .parameters["stall_timeout"]
            .default
        )
        self.assertIsNone(default)


if __name__ == "__main__":
    unittest.main()

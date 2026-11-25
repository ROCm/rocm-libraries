# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PR runners' slots (rkmon_lib/slots.py): the records' rules, against a records folder and a /proc of the test's
own, with stand-ins for the launcher.

- A job takes the first free slot: one with no record, or one whose process is gone. A taken, kept or freeing slot is
  never given to another job.
- With no free slot, a job takes the slot kept longest: its shared build record goes, a record that names another build
  stays. With none kept either, it waits until one is free.
- Named work gets its own slot back, fresh only the first time, while the process that started it (its owner: the
  first ancestor of rk that is no shell) runs; once its owner is gone the slot is free.
- A slot whose folder PR code changed is broken until the container's next start: no job takes it, and with every slot
  broken a job takes none instead of waiting.
- A job's slot is begun when taken fresh; one the launcher refuses is marked broken, its why the launcher's own last
  line, and the next taken; one whose begin a signal ended while processes of its runner still run is marked broken,
  its why naming the fix (restart the container), and the next taken; one whose processes outlive SIGKILL, or whose
  begin a signal ended with none of them left, is let go and the next taken; any other failure takes none.
- A slot's path splits into the slot's folder and the path under it, and names the slot's runner.
- A build in a slot's folder may be reused as the review's shared build only while the slot is kept for that build; a
  build in no slot's folder always may.
- Freeing ends the slot (the launcher's end), then removes its record, and a kept slot's shared build record; only the
  holder's record is freed, and a record another process wrote while the end ran stays. A slot whose end a signal
  ended while processes of its runner still run is marked broken.
- Keeping makes the record kept after the launcher's keep, and frees the other slots kept for that PR; a slot another
  process holds is not kept, and its launcher is not called.
- The records change under one lock: a take waits while another process holds it, and a write that dies midway leaves
  the old record as it was.
- A review's end frees its kept slot and its named work, and nothing else; named work taken with no review in flight
  stays until it ends.
- A PR's slots are the ones taken by its jobs or its named work and the one kept for its review, while they hold it.
- The launcher is called as the slot's user, in /, its stdin the environment or nothing; with its begin and its end
  the slot's inputs folder is emptied, a link in it removed and not its target.
"""

import errno
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import slots, whole_file  # noqa: E402

RUNNERS = ["rkpr1", "rkpr2", "rkpr3"]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        top = pathlib.Path(self.tmp.name)
        self.locks = top / "locks"
        self.locks.mkdir()
        self.proc = top / "proc"
        (self.proc / "sys" / "kernel" / "random").mkdir(parents=True)
        (self.proc / "sys" / "kernel" / "random" / "boot_id").write_text("boot-1\n")
        self.out = top / "out"
        self.out.mkdir()
        self.t = 1000.0
        self.rec = slots.Records(
            str(self.locks), RUNNERS, str(self.proc), clock=lambda: self.t
        )
        self.ended = []

    def process(
        self, pid: int, start: str = "777", comm: str = "bash", ppid: int = 0
    ) -> int:
        """A process of the fake /proc, started at `start`, running `comm`, a child of `ppid`."""
        d = self.proc / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(
            f"{pid} ({comm}) S {ppid} " + " ".join(["0"] * 17) + f" {start} 0 0\n"
        )
        return pid

    def gone(self, pid: int) -> None:
        (self.proc / str(pid) / "stat").unlink()

    def of_runner(self, pid: int, runner: str, state: str = "S") -> int:
        """A process of PR runner `runner` in the fake /proc, in `state` (Z: a zombie), as worker/pr_results.py's
        `running` reads it."""
        uid = next(u.uid for u in L.PR_RUNNERS if u.name == runner)
        d = self.proc / str(pid)
        d.mkdir(exist_ok=True)
        (d / "status").write_text(
            f"Name:\tsleep\nState:\t{state}\nPPid:\t1\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
        )
        return pid

    def end(self, runner):
        self.ended.append(runner)
        return 0, [f"ended {runner}"]

    def kept(self, runner, pr, head, since, build=None) -> str:
        """`runner` kept for PR `pr` at `head`, with its shared build record in the out folder."""
        build = build or f"{L.SLOTS_DIR}/{runner}/build/wt{pr}m"
        record = self.out / f"shared_build_{runner}"
        record.write_text(f"{build}\n/work/wt{pr}m\n{head}\n")
        self.rec.write(
            runner,
            {
                "state": "kept",
                "pr": str(pr),
                "head": head,
                "build": build,
                "record": str(record),
                "since": since,
            },
        )
        return str(record)


class TestTake(Fixture):
    def test_the_first_free_slot_with_the_process_that_holds_it(self):
        pid = self.process(41)
        self.assertEqual(
            slots.take(self.rec, job="/j/1", pr=7, pid=pid), ("rkpr1", True)
        )
        rec = self.rec.read("rkpr1")
        self.assertEqual(
            rec,
            {
                "state": "taken",
                "job": "/j/1",
                "pr": "7",
                "since": 1000.0,
                "pid": 41,
                "start": "777",
                "boot": "boot-1",
            },
        )
        self.assertEqual(slots.take(self.rec, pid=self.process(42)), ("rkpr2", True))

    def test_a_slot_whose_process_is_gone_is_free_again(self):
        slots.take(self.rec, pid=self.process(41))
        slots.take(self.rec, pid=self.process(42))
        self.gone(41)
        self.assertEqual(slots.take(self.rec, pid=self.process(43))[0], "rkpr1")

    def test_a_reused_pid_or_a_reboot_does_not_hold_a_slot(self):
        slots.take(self.rec, pid=self.process(41, start="777"))
        self.process(41, start="900")  # pid 41 is another process now
        self.assertEqual(slots.take(self.rec, pid=self.process(42))[0], "rkpr1")
        (self.proc / "sys" / "kernel" / "random" / "boot_id").write_text("boot-2\n")
        self.assertEqual(slots.take(self.rec, pid=self.process(43))[0], "rkpr1")

    def test_no_job_runs_in_a_taken_kept_or_freeing_slot(self):
        self.kept("rkpr1", 7, "aaa", since=5)
        self.rec.write(
            "rkpr2", {"state": "freeing", **self.rec.holder(self.process(50))}
        )
        got = slots.take(self.rec, pid=self.process(51))
        self.assertEqual(got, ("rkpr3", True))
        self.assertEqual(self.rec.read("rkpr1")["state"], "kept")
        self.assertEqual(self.rec.read("rkpr2")["state"], "freeing")
        # None free: the kept slot goes to the next job, never the freeing or the taken one.
        self.assertEqual(slots.take(self.rec, pid=self.process(52)), ("rkpr1", True))
        self.assertEqual(self.rec.read("rkpr2")["state"], "freeing")
        self.assertEqual(self.rec.read("rkpr3")["pid"], 51)

    def test_with_none_free_the_slot_kept_longest_is_taken_and_its_record_goes(self):
        r1 = self.kept("rkpr1", 7, "aaa", since=300)
        # a build folder with a space: the record names it whole
        r2 = self.kept("rkpr2", 8, "bbb", since=100, build="/ramdisk/slot dir/wt8m")
        r3 = self.kept("rkpr3", 9, "ccc", since=200)
        self.assertEqual(slots.take(self.rec, pid=self.process(41)), ("rkpr2", True))
        self.assertFalse(os.path.exists(r2))
        self.assertTrue(os.path.exists(r1) and os.path.exists(r3))
        self.assertEqual(self.rec.read("rkpr2")["state"], "taken")
        self.assertEqual(slots.take(self.rec, pid=self.process(42)), ("rkpr3", True))
        self.assertEqual(slots.take(self.rec, pid=self.process(43)), ("rkpr1", True))

    def test_a_record_that_names_another_build_stays(self):
        record = self.kept("rkpr1", 7, "aaa", since=1)
        pathlib.Path(record).write_text(
            "/ramdisk/slots/rkpr2/build/wt7m\n/work/wt7m\nbbb\n"
        )
        for r in ("rkpr2", "rkpr3"):
            slots.take(self.rec, name=f"pr1-{r}")
        self.assertEqual(slots.take(self.rec, pid=self.process(41))[0], "rkpr1")
        self.assertTrue(os.path.exists(record))

    def test_with_none_free_and_none_kept_it_waits(self):
        for r in RUNNERS:
            slots.take(self.rec, name=f"pr1-{r}")
        said, slept = [], []

        def sleep(s):
            slept.append(s)
            if len(slept) == 3:
                self.rec.remove("rkpr2")

        got = slots.take(
            self.rec, pid=self.process(41), sleep=sleep, say=said.append, wait_s=7
        )
        self.assertEqual(got, ("rkpr2", True))
        self.assertEqual(slept, [7, 7, 7])
        self.assertEqual(len(said), 1)

    def test_named_work_gets_its_slot_back_fresh_the_first_time_only(self):
        self.assertEqual(slots.take(self.rec, name="pr7-build", pr=7), ("rkpr1", True))
        self.assertEqual(slots.take(self.rec, name="pr7-other", pr=7), ("rkpr2", True))
        self.assertEqual(slots.take(self.rec, name="pr7-build", pr=7), ("rkpr1", False))
        self.assertEqual(self.rec.read("rkpr1")["name"], "pr7-build")

    def test_a_pid_or_a_name(self):
        with self.assertRaises(ValueError):
            slots.take(self.rec)

    def test_the_name_rule(self):
        for good in ("pr7-build", "pr12345-a.b_c-d", "pr1-0"):
            self.assertTrue(slots.NAME_RE.match(good), good)
        for bad in (
            "pr0-x",
            "7-build",
            "pr7",
            "pr7-",
            "pr7-Build",
            "pr7-../x",
            "pr7-a b",
        ):
            self.assertFalse(slots.NAME_RE.match(bad), bad)
        self.assertEqual(slots.NAME_RE.match("pr12-build").group(1), "12")


class TestOwner(Fixture):
    """Named work's owner: the process that started it."""

    def test_the_owner_is_the_first_ancestor_that_is_no_shell(self):
        self.process(10, comm="claude", ppid=1, start="100")
        self.process(20, comm="bash", ppid=10)
        self.process(25, comm="timeout", ppid=20)
        self.process(30, comm="python3", ppid=25)  # rk, after its exec
        self.assertEqual(
            self.rec.owner_of(30), {"pid": 10, "start": "100", "boot": "boot-1"}
        )

    def test_none_when_only_shells_lead_up_to_the_first_process(self):
        self.process(20, comm="bash", ppid=1)
        self.process(30, comm="python3", ppid=20)
        self.assertIsNone(self.rec.owner_of(30))
        self.process(31, comm="python3", ppid=99)  # a parent this process cannot see
        self.assertIsNone(self.rec.owner_of(31))
        self.assertIsNone(self.rec.owner_of(98))  # no such process
        # a command name with spaces and a parenthesis is read whole
        self.process(40, comm="a (b) c", ppid=1)
        self.process(41, comm="bash", ppid=40)
        self.process(42, comm="python3", ppid=41)
        self.assertEqual(self.rec.owner_of(42)["pid"], 40)

    def test_named_work_is_free_once_its_owner_is_gone(self):
        owner = self.rec.holder(self.process(10, comm="claude", ppid=1))
        self.assertEqual(
            slots.take(self.rec, name="pr7-build", pr=7, owner=owner), ("rkpr1", True)
        )
        self.assertEqual(self.rec.read("rkpr1")["owner"], owner)
        self.assertEqual(
            slots.take(self.rec, name="pr7-build", pr=7, owner=owner), ("rkpr1", False)
        )
        self.gone(10)
        self.assertFalse(self.rec.held(self.rec.read("rkpr1")))
        # its name gets a fresh slot, and another job may take the one it held
        self.assertEqual(slots.take(self.rec, pid=self.process(41)), ("rkpr1", True))
        self.assertEqual(slots.take(self.rec, name="pr7-build", pr=7), ("rkpr2", True))

    def test_named_work_with_no_owner_holds_its_slot_until_it_ends(self):
        slots.take(self.rec, name="pr7-build", pr=7)
        self.assertNotIn("owner", self.rec.read("rkpr1"))
        self.assertTrue(self.rec.held(self.rec.read("rkpr1")))

    def test_whose_taking_a_record_is(self):
        rec = {"state": "taken", "pid": 41}
        self.assertTrue(slots.taken_by(rec, pid=41))
        self.assertFalse(slots.taken_by(rec, pid=42))
        named = {"state": "taken", "name": "pr7-x"}
        self.assertTrue(slots.taken_by(named, name="pr7-x"))
        self.assertFalse(slots.taken_by(named, name="pr7-y"))
        self.assertFalse(slots.taken_by(dict(rec, state="kept"), pid=41))


class TestBroken(Fixture):
    def setUp(self):
        super().setUp()
        self.process(1, start="5")  # the container's PID 1

    def test_broken_until_the_containers_next_start(self):
        slots.mark_broken(self.rec, "rkpr1", "PR_SLOT rkpr1: refused: changed")
        rec = self.rec.read("rkpr1")
        self.assertEqual(
            (rec["state"], rec["init"], rec["boot"], rec["why"]),
            ("broken", "5", "boot-1", "PR_SLOT rkpr1: refused: changed"),
        )
        self.assertEqual(slots.take(self.rec, pid=self.process(41))[0], "rkpr2")
        self.process(1, start="6")  # the container started again
        self.assertEqual(slots.take(self.rec, pid=self.process(42))[0], "rkpr1")

    def test_a_reboot_ends_it_too(self):
        slots.mark_broken(self.rec, "rkpr1", "x")
        (self.proc / "sys" / "kernel" / "random" / "boot_id").write_text("boot-2\n")
        self.assertEqual(slots.take(self.rec, pid=self.process(41))[0], "rkpr1")

    def test_with_every_slot_broken_none_is_taken_and_nothing_waits(self):
        for r in RUNNERS:
            slots.mark_broken(self.rec, r, "x")

        def sleep(_s):
            raise AssertionError("waited for a slot that cannot come")

        self.assertEqual(
            slots.take(self.rec, pid=self.process(41), sleep=sleep), ("", False)
        )

    def test_a_kept_slot_is_still_taken_when_the_others_are_broken(self):
        slots.mark_broken(self.rec, "rkpr1", "x")
        slots.mark_broken(self.rec, "rkpr2", "x")
        self.kept("rkpr3", 7, "aaa", since=1)
        self.assertEqual(slots.take(self.rec, pid=self.process(41)), ("rkpr3", True))

    def test_skipped_runners_are_not_taken(self):
        self.assertEqual(
            slots.take(self.rec, pid=self.process(41), skip={"rkpr1"}),
            ("rkpr2", True),
        )


class TestTakeBegun(Fixture):
    def setUp(self):
        super().setUp()
        self.process(1, start="5")
        self.begun = []

    def no_wait(self, _s):
        raise AssertionError("waited for a slot that cannot come")

    def begin(self, codes):
        """The launcher's begin as slots.begin hands it on: the launcher's own line, then lines_of's when it failed."""

        def call(runner):
            if len(self.begun) == len(RUNNERS):
                raise AssertionError("a slot begun again and again")
            self.begun.append(runner)
            code = codes.get(runner, 0)
            said = subprocess.CompletedProcess(
                [], code, b"", f"begin {runner}: {code}\n".encode()
            )
            return code, slots.lines_of(said, runner, "begin")

        return call

    def test_a_fresh_slot_is_begun(self):
        got = slots.take_begun(
            self.rec, self.begin({}), pid=self.process(41), job="/j", pr=7
        )
        self.assertEqual(got, ("rkpr1", ["begin rkpr1: 0"]))
        self.assertEqual(self.rec.read("rkpr1")["job"], "/j")

    def test_named_work_that_holds_its_slot_is_not_begun_again(self):
        slots.take(self.rec, name="pr7-build", pr=7)
        got = slots.take_begun(self.rec, self.begin({}), name="pr7-build", pr=7)
        self.assertEqual(got, ("rkpr1", []))
        self.assertEqual(self.begun, [])

    def test_a_refused_slot_is_marked_broken_and_the_next_is_taken(self):
        got = slots.take_begun(
            self.rec, self.begin({"rkpr1": slots.REFUSED}), pid=self.process(41)
        )
        self.assertEqual(got[0], "rkpr2")
        self.assertEqual(self.begun, ["rkpr1", "rkpr2"])
        self.assertEqual(self.rec.read("rkpr1")["state"], "broken")
        self.assertEqual(
            self.rec.read("rkpr1")["why"], "begin rkpr1: 3"
        )  # its own line
        self.assertEqual(slots.take(self.rec, pid=self.process(42))[0], "rkpr3")

    def test_the_why_of_a_refusal_is_the_launchers_own_last_line(self):
        def refuses(stdout: bytes, stderr: bytes):
            def call(runner):
                p = subprocess.CompletedProcess([], slots.REFUSED, stdout, stderr)
                return slots.REFUSED, slots.lines_of(p, runner, "begin")

            return call

        def why(stdout: bytes, stderr: bytes) -> str:
            self.rec.remove("rkpr1")
            slots.take_begun(
                self.rec,
                refuses(stdout, stderr),
                pid=self.process(41),
                skip={"rkpr2", "rkpr3"},
                sleep=self.no_wait,
            )
            return self.rec.read("rkpr1")["why"]

        stopped = b"PR_SLOT rkpr1: stopped 2 process(es)\n"
        refused = b"PR_SLOT rkpr1: refused: its mode changed\n"
        self.assertEqual(why(stopped, refused), refused.decode().strip())
        # none of its own: the failure's line says it
        self.assertEqual(why(b"", b""), "PR_SLOT rkpr1: begin failed (exit 3)")

    def test_a_slot_whose_processes_outlive_sigkill_is_let_go(self):
        got = slots.take_begun(
            self.rec, self.begin({"rkpr1": slots.LEFT}), pid=self.process(41)
        )
        self.assertEqual(got[0], "rkpr2")
        self.assertIsNone(self.rec.read("rkpr1"))  # free: the next take begins it again

    def test_a_slot_whose_begin_a_signal_ended_is_let_go(self):
        """A process of the slot's user can end the launcher (they are one user): sudo then ends as it did. With none
        of the runner's processes left (a zombie has ended), the next take begins it again.
        """
        self.of_runner(60, "rkpr1", state="Z")
        self.of_runner(61, "rkpr2")  # another runner's
        got = slots.take_begun(
            self.rec, self.begin({"rkpr1": -9}), pid=self.process(41)
        )
        self.assertEqual(
            got,
            (
                "rkpr2",
                [
                    "begin rkpr1: -9",
                    "PR_SLOT rkpr1: begin failed (exit -9)",
                    "begin rkpr2: 0",
                ],
            ),
        )
        self.assertIsNone(self.rec.read("rkpr1"))  # free: the next take begins it again

    def test_a_slot_whose_begin_a_signal_ended_while_its_runner_runs_is_broken(self):
        """A process of the runner that ends every process of the runner's ends each launcher call of its slot: the
        slot is broken until the container's next start stops it, and the next is taken.
        """
        self.of_runner(60, "rkpr1")
        got = slots.take_begun(
            self.rec, self.begin({"rkpr1": -9}), pid=self.process(41)
        )
        why = slots.stuck("rkpr1", "begin", -9, 1)
        self.assertIn(slots.RESTART, why)
        rec = self.rec.read("rkpr1")
        self.assertEqual((rec["state"], rec["init"], rec["why"]), ("broken", "5", why))
        self.assertEqual(
            got,
            (
                "rkpr2",
                [
                    "begin rkpr1: -9",
                    "PR_SLOT rkpr1: begin failed (exit -9)",
                    f"PR_SLOT rkpr1: broken: {why}",
                    "begin rkpr2: 0",
                ],
            ),
        )
        self.assertEqual(slots.take(self.rec, pid=self.process(42))[0], "rkpr3")

    def test_any_other_failure_takes_none(self):
        self.of_runner(60, "rkpr1")
        got = slots.take_begun(
            self.rec, self.begin({"rkpr1": 127}), pid=self.process(41)
        )
        self.assertEqual(
            got, ("", ["begin rkpr1: 127", "PR_SLOT rkpr1: begin failed (exit 127)"])
        )
        self.assertEqual(self.begun, ["rkpr1"])
        self.assertIsNone(self.rec.read("rkpr1"))

    def test_when_every_slot_is_refused_none_is_taken(self):
        refuse = {r: slots.REFUSED for r in RUNNERS}
        runner, lines = slots.take_begun(
            self.rec, self.begin(refuse), pid=self.process(41), sleep=self.no_wait
        )
        self.assertEqual(runner, "")
        self.assertEqual(self.begun, RUNNERS)
        self.assertIn("no slot can be taken", lines[-1])

    def test_release_frees_only_the_holders_taking(self):
        slots.take(self.rec, pid=self.process(41))
        slots.release(self.rec, "rkpr1", pid=99)
        self.assertEqual(self.rec.read("rkpr1")["pid"], 41)
        slots.release(self.rec, "rkpr1", pid=41)
        self.assertIsNone(self.rec.read("rkpr1"))
        slots.take(self.rec, name="pr7-x", pr=7)
        slots.release(self.rec, "rkpr1", name="pr7-y")
        self.assertEqual(self.rec.read("rkpr1")["name"], "pr7-x")
        slots.release(self.rec, "rkpr1", name="pr7-x")
        self.assertIsNone(self.rec.read("rkpr1"))


class TestFree(Fixture):
    def test_end_then_the_record_goes(self):
        slots.take(self.rec, pid=self.process(41))
        seen = []

        def end(runner):
            seen.append(self.rec.read(runner)["state"])
            return self.end(runner)

        self.assertEqual(slots.free(self.rec, "rkpr1", end), ["ended rkpr1"])
        self.assertEqual(seen, ["freeing"])
        self.assertIsNone(self.rec.read("rkpr1"))

    def test_a_kept_slots_record_goes_before_its_build(self):
        record = self.kept("rkpr1", 7, "aaa", since=1)
        seen = []

        def end(runner):
            seen.append(os.path.exists(record))
            return 0, []

        slots.free(self.rec, "rkpr1", end)
        self.assertEqual(seen, [False])
        self.assertIsNone(self.rec.read("rkpr1"))

    def test_only_the_holders_record_is_freed(self):
        slots.take(self.rec, pid=self.process(41))
        self.assertEqual(
            slots.free(self.rec, "rkpr1", self.end, holds=lambda r: r.get("pid") == 99),
            [],
        )
        self.assertEqual(self.ended, [])
        self.assertEqual(self.rec.read("rkpr1")["pid"], 41)

    def test_a_free_slot_needs_nothing(self):
        self.assertEqual(slots.free(self.rec, "rkpr2", self.end), [])
        self.assertEqual(self.ended, [])

    def test_a_record_another_process_wrote_while_the_end_ran_stays(self):
        for other in (
            {"state": "freeing", **self.rec.holder(self.process(42)), "since": 2.0},
            {"state": "taken", **self.rec.holder(self.process(43)), "since": 3.0},
        ):
            slots.take(self.rec, pid=self.process(41))

            def end(runner, other=other):
                self.rec.write(runner, other)
                return 0, []

            slots.free(self.rec, "rkpr1", end)
            self.assertEqual(self.rec.read("rkpr1"), other)
            self.rec.remove("rkpr1")

    def test_a_slot_whose_end_a_signal_ended_while_its_runner_runs_is_broken(self):
        """A process of the runner can end the launcher's end too: the slot is then broken until the container's
        next start, which stops what is left; with nothing of the runner left it is freed.
        """
        self.process(1, start="5")

        def killed(runner):
            return -9, [slots.failed(runner, "end", -9)]

        slots.take(self.rec, pid=self.process(41))
        slots.take(self.rec, pid=self.process(42))
        self.of_runner(60, "rkpr1")
        lines = slots.free(self.rec, "rkpr1", killed)
        why = slots.stuck("rkpr1", "end", -9, 1)
        rec = self.rec.read("rkpr1")
        self.assertEqual((rec["state"], rec["init"], rec["why"]), ("broken", "5", why))
        self.assertEqual(
            lines,
            ["PR_SLOT rkpr1: end failed (exit -9)", f"PR_SLOT rkpr1: broken: {why}"],
        )
        self.assertEqual(
            slots.free(self.rec, "rkpr2", killed), [slots.failed("rkpr2", "end", -9)]
        )
        self.assertIsNone(self.rec.read("rkpr2"))  # none of its processes left
        # a failure that is no signal leaves the slot to the next begin, as before
        slots.take(self.rec, pid=self.process(43))
        self.of_runner(61, "rkpr2")
        self.assertEqual(slots.free(self.rec, "rkpr2", lambda r: (slots.LEFT, [])), [])
        self.assertIsNone(self.rec.read("rkpr2"))


class TestKeep(Fixture):
    def test_kept_after_the_launchers_keep(self):
        pid = self.process(41)
        slots.take(self.rec, pid=pid, pr=7)
        ok, lines = slots.keep(
            self.rec,
            "rkpr1",
            lambda r: (True, [f"kept {r}"]),
            self.end,
            pid=pid,
            pr=7,
            head="aaa",
            build="/ramdisk/slots/rkpr1/build/wt7m",
            record="/x/shared_build",
        )
        self.assertTrue(ok)
        self.assertEqual(lines, ["kept rkpr1"])
        self.assertEqual(
            self.rec.read("rkpr1"),
            {
                "state": "kept",
                "pr": "7",
                "head": "aaa",
                "build": "/ramdisk/slots/rkpr1/build/wt7m",
                "record": "/x/shared_build",
                "since": 1000.0,
            },
        )

    def test_the_other_slots_kept_for_the_pr_are_freed(self):
        old = self.kept("rkpr2", 7, "old", since=1)
        other = self.kept("rkpr3", 8, "ccc", since=1)
        pid = self.process(41)
        slots.take(self.rec, pid=pid, pr=7)
        ok, _ = slots.keep(
            self.rec,
            "rkpr1",
            lambda r: (True, []),
            self.end,
            pid=pid,
            pr=7,
            head="new",
            build="b",
            record="r",
        )
        self.assertTrue(ok)
        self.assertEqual(self.ended, ["rkpr2"])
        self.assertIsNone(self.rec.read("rkpr2"))
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(other))

    def test_not_kept_when_the_launcher_fails_or_another_holds_it(self):
        pid = self.process(41)
        slots.take(self.rec, pid=pid)
        ok, lines = slots.keep(
            self.rec,
            "rkpr1",
            lambda r: (False, ["no"]),
            self.end,
            pid=pid,
            pr=7,
            head="a",
            build="b",
            record="r",
        )
        self.assertEqual((ok, lines), (False, ["no"]))
        called = []
        ok, lines = slots.keep(
            self.rec,
            "rkpr1",
            lambda r: called.append(r) or (True, []),
            self.end,
            pid=99,
            pr=7,
            head="a",
            build="b",
            record="r",
        )
        self.assertEqual(
            (ok, lines), (False, ["PR_SLOT rkpr1: not taken by process 99: not kept"])
        )
        self.assertEqual(called, [])  # the record is checked before the launcher's keep
        self.assertEqual(self.rec.read("rkpr1")["state"], "taken")

    def test_not_kept_when_the_record_changed_while_the_launcher_kept_it(self):
        pid = self.process(41)
        slots.take(self.rec, pid=pid, pr=7)
        other = {"state": "freeing", **self.rec.holder(self.process(42)), "since": 2.0}

        def keep_call(runner):
            self.rec.write(runner, other)
            return True, ["kept"]

        ok, lines = slots.keep(
            self.rec,
            "rkpr1",
            keep_call,
            self.end,
            pid=pid,
            pr=7,
            head="a",
            build="b",
            record="r",
        )
        self.assertEqual(
            (ok, lines),
            (False, ["kept", "PR_SLOT rkpr1: not taken by process 41: not kept"]),
        )
        self.assertEqual(self.rec.read("rkpr1"), other)


class TestReviewEnd(Fixture):
    def test_the_review_end_frees_its_kept_slot_and_named_work_only(self):
        live = {("7", "aaa"), ("8", "bbb")}

        def is_live(pr, head):
            return (pr, head) in live

        self.rec.runners.append("rkpr4")
        r7 = self.kept("rkpr1", 7, "aaa", since=1)
        r9 = self.kept("rkpr2", 9, "ccc", since=1)
        self.assertEqual(
            slots.take(self.rec, name="pr8-build", pr=8, head="bbb")[0], "rkpr3"
        )
        self.assertEqual(
            slots.take(self.rec, name="pr9-build", pr=9, head="ccc")[0], "rkpr4"
        )
        self.assertEqual(self.rec.read("rkpr3")["head"], "bbb")
        lines = slots.free_ended(self.rec, is_live, self.end)
        self.assertEqual(sorted(self.ended), ["rkpr2", "rkpr4"])
        self.assertEqual(sorted(lines), ["ended rkpr2", "ended rkpr4"])
        self.assertTrue(os.path.exists(r7))
        self.assertFalse(os.path.exists(r9))
        self.assertEqual(self.rec.read("rkpr1")["state"], "kept")
        self.assertEqual(self.rec.read("rkpr3")["name"], "pr8-build")
        self.assertIsNone(self.rec.read("rkpr2"))
        self.assertIsNone(self.rec.read("rkpr4"))

    def test_a_superseded_head_frees_the_slot_kept_for_the_old_one(self):
        self.kept("rkpr1", 7, "old", since=1)
        slots.take(self.rec, name="pr7-build", pr=7, head="old")
        slots.free_ended(self.rec, lambda pr, head: head == "new", self.end)
        self.assertEqual(self.ended, ["rkpr1", "rkpr2"])

    def test_named_work_taken_with_no_review_in_flight_stays_until_it_ends(self):
        slots.take(self.rec, name="pr7-build", pr=7)
        self.assertEqual(self.rec.read("rkpr1")["head"], "")
        self.assertEqual(
            slots.free_ended(self.rec, lambda pr, head: False, self.end), []
        )
        self.assertEqual(self.rec.read("rkpr1")["name"], "pr7-build")

    def test_named_work_whose_owner_is_gone_is_freed(self):
        """Work left without --end: its owner gone, its slot is freed with the reviews that ended, in flight or not."""
        owner = self.rec.holder(self.process(51, comm="claude"))
        slots.take(self.rec, name="pr7-build", pr=7, head="aaa", owner=owner)
        slots.take(self.rec, name="pr8-build", pr=8, owner=owner)
        slots.take(self.rec, name="pr9-build", pr=9)  # no owner: only shells led to it
        live = lambda pr, head: True  # noqa: E731
        self.assertEqual(slots.free_ended(self.rec, live, self.end), [])
        self.gone(51)
        self.assertEqual(
            slots.free_ended(self.rec, live, self.end), ["ended rkpr1", "ended rkpr2"]
        )
        self.assertEqual(self.ended, ["rkpr1", "rkpr2"])
        self.assertEqual(self.rec.read("rkpr3")["name"], "pr9-build")

    def test_slots_taken_by_a_process_are_left(self):
        slots.take(self.rec, pid=self.process(41), pr=7)
        self.assertEqual(
            slots.free_ended(self.rec, lambda pr, head: False, self.end), []
        )
        self.assertEqual(self.rec.read("rkpr1")["state"], "taken")


class TestByPr(Fixture):
    def test_the_slots_taken_or_kept_for_each_pr(self):
        """What the RAM disk's footprints count as a PR's: its jobs' slots, its named work's and the one kept for its
        review. A slot whose process is gone, a job's of no PR, one being freed and a broken one are no PR's.
        """
        slots.take(self.rec, job="/j/1", pr=7, pid=self.process(41))
        slots.take(self.rec, name="pr8-tests", pr=8, head="bbb")
        self.kept("rkpr3", 7, "aaa", since=1)
        self.assertEqual(slots.by_pr(self.rec), {7: ["rkpr1", "rkpr3"], 8: ["rkpr2"]})
        self.gone(41)
        self.assertEqual(slots.by_pr(self.rec), {7: ["rkpr3"], 8: ["rkpr2"]})
        self.assertEqual(
            slots.take(self.rec, job="/j/2", pid=self.process(42))[0], "rkpr1"
        )
        self.assertEqual(slots.by_pr(self.rec), {7: ["rkpr3"], 8: ["rkpr2"]})
        self.rec.write(
            "rkpr3",
            {"state": "freeing", **self.rec.holder(self.process(43)), "since": 2},
        )
        slots.mark_broken(self.rec, "rkpr2", "refused")
        self.assertEqual(slots.by_pr(self.rec), {})

    def test_the_layouts_records_when_none_are_named(self):
        self.kept("rkpr1", 7, "aaa", since=1)
        with mock.patch.object(L, "SLOT_LOCKS", str(self.locks)):
            self.assertEqual(slots.by_pr(), {7: ["rkpr1"]})


class TestKeptBuild(Fixture):
    def test_a_build_in_a_slot_only_while_the_slot_is_kept_for_it(self):
        build = f"{L.SLOTS_DIR}/rkpr1/build/wt7m"
        self.assertFalse(slots.kept_build(build, self.rec))  # a free slot
        self.kept("rkpr1", 7, "aaa", since=1)
        self.assertTrue(slots.kept_build(build, self.rec))
        self.assertTrue(slots.kept_build(build + "/", self.rec))
        # another tree's build in the same slot, and the build of another slot
        self.assertFalse(slots.kept_build(f"{L.SLOTS_DIR}/rkpr1/build/wt8m", self.rec))
        self.assertFalse(slots.kept_build(f"{L.SLOTS_DIR}/rkpr2/build/wt7m", self.rec))
        # a record that names the build but is not kept
        self.rec.write("rkpr1", dict(self.rec.read("rkpr1"), state="freeing"))
        self.assertFalse(slots.kept_build(build, self.rec))
        self.kept("rkpr1", 7, "aaa", since=1)
        # taken back for another job
        slots.take(self.rec, pid=self.process(41))
        slots.take(self.rec, pid=self.process(42))
        self.assertEqual(slots.take(self.rec, pid=self.process(43))[0], "rkpr1")
        self.assertFalse(slots.kept_build(build, self.rec))

    def test_a_build_in_no_slot_always(self):
        for path in ("/work/build/wt7m", f"{L.SLOTS_DIR}/locks/x", "relative/wt7m"):
            self.assertTrue(slots.kept_build(path, self.rec), path)

    def test_the_records_folder_is_the_layouts_when_none_is_named(self):
        self.kept("rkpr1", 7, "aaa", since=1)
        build = f"{L.SLOTS_DIR}/rkpr1/build/wt7m"
        with mock.patch.object(L, "SLOT_LOCKS", str(self.locks)):
            self.assertTrue(slots.kept_build(build))
            self.assertEqual(slots.Records().folder, str(self.locks))


# Another process that holds the records' lock (its path: argv[1]) until its stdin closes.
HOLD = """import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
sys.stdin.read()
"""


class TestTheRecordsLock(Fixture):
    def test_the_lock_is_a_file_the_monitor_shares(self):
        old = os.umask(0)
        try:
            with self.rec.lock():
                pass
        finally:
            os.umask(old)
        self.assertEqual((self.locks / slots.LOCK).stat().st_mode & 0o777, 0o660)

    def test_a_take_waits_while_another_process_holds_the_lock(self):
        holder = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", HOLD, str(self.locks / slots.LOCK)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        got = []
        taker = threading.Thread(
            target=lambda: got.append(slots.take(self.rec, pid=self.process(41))),
            daemon=True,
        )

        def let_go():
            holder.kill()
            holder.wait(10)
            holder.stdin.close()
            holder.stdout.close()
            if taker.ident is not None:
                taker.join(10)

        self.addCleanup(let_go)
        self.assertEqual(holder.stdout.readline(), b"held\n")
        taker.start()
        taker.join(0.5)
        self.assertTrue(taker.is_alive())  # waiting for the lock
        self.assertIsNone(self.rec.read("rkpr1"))
        holder.stdin.close()  # the other process lets go
        taker.join(10)
        self.assertEqual(got, [("rkpr1", True)])

    def test_a_write_that_dies_midway_leaves_the_old_record(self):
        # The record is written whole (whole_file): a write that fails partway, as on a full disk, leaves the old
        # record, and nothing beside it.
        old = {"state": "kept", "pr": "7", "since": 1.0}
        self.rec.write("rkpr1", old)
        fill = whole_file.fill

        def dies(fd, data, mode, durable):
            fill(fd, data[:9], mode, durable)
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        with mock.patch.object(whole_file, "fill", dies), self.assertRaises(OSError):
            self.rec.write("rkpr1", {"state": "taken", "pid": 41, "since": 2.0})
        self.assertEqual(self.rec.read("rkpr1"), old)
        self.assertEqual(os.listdir(self.locks), ["rkpr1.json"])

    def test_a_record_has_the_shared_mode_whatever_the_umask(self):
        umask = os.umask(0o077)
        try:
            self.rec.write("rkpr2", {"state": "kept", "pr": "8", "since": 1.0})
        finally:
            os.umask(umask)
        self.assertEqual(
            stat.S_IMODE((self.locks / "rkpr2.json").stat().st_mode),
            slots.paths.worker_module("perms").SHARED_FILE_MODE,
        )


class TestRealProc(unittest.TestCase):
    def test_this_process_holds_what_it_takes(self):
        with tempfile.TemporaryDirectory() as d:
            rec = slots.Records(d, RUNNERS)
            self.assertEqual(slots.take(rec, pid=os.getpid()), ("rkpr1", True))
            self.assertTrue(rec.held(rec.read("rkpr1")))
            self.assertEqual(slots.take(rec, pid=os.getpid())[0], "rkpr2")


class TestLauncherSteps(unittest.TestCase):
    """The launcher's begin, end and keep as the slot's user, and the slot's inputs emptied with begin and end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.top = pathlib.Path(self.tmp.name) / "inputs"
        mine = self.top / "rkpr1"
        (mine / "sub").mkdir(parents=True)
        (mine / "winsim_scan.txt").write_text("x")
        (mine / "sub" / "a").write_text("y")
        self.outside = pathlib.Path(self.tmp.name) / "outside"
        self.outside.mkdir()
        (self.outside / "keep.txt").write_text("z")
        (mine / "link").symlink_to(self.outside)
        self.calls = []

    def run_as(self, code):
        def run(argv, **kw):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, code, b"said\n", b"")

        return run

    def test_begin_empties_the_inputs_after_the_launchers_begin(self):
        got = slots.begin("rkpr1", "/l", run=self.run_as(0), inputs=str(self.top))
        self.assertEqual(got, (0, ["said"]))
        self.assertEqual(self.calls, [["sudo", "-n", "-u", "rkpr1", "/l", "begin"]])
        self.assertEqual(list((self.top / "rkpr1").iterdir()), [])
        self.assertTrue(
            (self.outside / "keep.txt").exists()
        )  # the link went, not its target

    def test_a_begin_that_fails_leaves_the_inputs(self):
        code, lines = slots.begin(
            "rkpr1", "/l", run=self.run_as(3), inputs=str(self.top)
        )
        self.assertEqual(code, 3)
        self.assertEqual(lines[-1], "PR_SLOT rkpr1: begin failed (exit 3)")
        self.assertTrue((self.top / "rkpr1" / "winsim_scan.txt").exists())

    def test_end_empties_them_too(self):
        got = slots.end("rkpr1", "/l", run=self.run_as(0), inputs=str(self.top))
        self.assertEqual(got, (0, ["said"]))
        self.assertEqual(self.calls, [["sudo", "-n", "-u", "rkpr1", "/l", "end"]])
        self.assertEqual(list((self.top / "rkpr1").iterdir()), [])
        self.assertEqual(slots.clear_inputs("rkpr2", str(self.top)), [])  # none there

    def test_keep(self):
        self.assertEqual(
            slots.keep_call("rkpr1", "/l", run=self.run_as(0)), (True, ["said"])
        )
        self.assertEqual(self.calls, [["sudo", "-n", "-u", "rkpr1", "/l", "keep"]])
        ok, lines = slots.keep_call("rkpr1", "/l", run=self.run_as(3))
        self.assertFalse(ok)
        self.assertEqual(lines[-1], "PR_SLOT rkpr1: keep failed (exit 3)")
        self.assertTrue((self.top / "rkpr1" / "winsim_scan.txt").exists())


class TestCall(unittest.TestCase):
    def test_as_the_slots_user_in_slash(self):
        seen = []

        def run(argv, **kw):
            seen.append((argv, kw))
            return subprocess.CompletedProcess(argv, 0, b"ok\n", b"")

        slots.call("rkpr2", ["begin"], "/opt/x/docker/rk_prcode.py", run=run)
        argv, kw = seen[0]
        self.assertEqual(
            argv, ["sudo", "-n", "-u", "rkpr2", "/opt/x/docker/rk_prcode.py", "begin"]
        )
        self.assertEqual(kw["cwd"], "/")
        self.assertIs(kw["stdin"], subprocess.DEVNULL)
        slots.call("rkpr2", ["run", "--", "id"], "/l", data=b"A=1\0", run=run)
        self.assertEqual(seen[1][1]["input"], b"A=1\0")
        self.assertNotIn("stdin", seen[1][1])

    def test_a_timeout_or_a_failed_start(self):
        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 5)

        def missing(argv, **kw):
            raise FileNotFoundError(2, "No such file")

        self.assertEqual(slots.call("rkpr1", ["end"], "/l", run=slow).returncode, 124)
        self.assertEqual(
            slots.call("rkpr1", ["end"], "/l", run=missing).returncode, 127
        )

    def test_the_lines_of_a_call(self):
        p = subprocess.CompletedProcess([], 3, b"one\n", b"two\n")
        self.assertEqual(
            slots.lines_of(p, "rkpr1", "end"),
            ["one", "two", "PR_SLOT rkpr1: end failed (exit 3)"],
        )

    def test_split(self):
        top = L.SLOTS_DIR
        self.assertEqual(
            slots.split(f"{top}/rkpr3/build/wt7m/core/x.json"),
            (f"{top}/rkpr3", "build/wt7m/core/x.json"),
        )
        self.assertEqual(slots.split(f"{top}/rkpr3"), (f"{top}/rkpr3", ""))
        self.assertEqual(slots.split(f"{top}/locks/x"), (f"{top}/locks/x", ""))
        self.assertEqual(slots.split("/work/build/wt7m"), ("/work/build/wt7m", ""))
        self.assertEqual(slots.split(f"{top}/rkpr3/../locks/x"), (f"{top}/locks/x", ""))
        self.assertEqual(slots.slot_of(f"{top}/rkpr3/build/wt7m"), "rkpr3")
        self.assertEqual(slots.slot_of(f"{top}/rkpr3"), "rkpr3")
        for other in (
            f"{top}/locks/x",
            f"{top}/rkpr99/x",
            f"{top}x/rkpr3",
            "/work/build",
        ):
            self.assertEqual(slots.slot_of(other), "", other)

    def test_the_folders(self):
        self.assertEqual(slots.folder("rkpr3"), f"{L.SLOTS_DIR}/rkpr3")
        self.assertEqual(slots.folder("rkpr3", "build"), f"{L.SLOTS_DIR}/rkpr3/build")
        self.assertEqual(slots.inputs("rkpr3"), f"{L.SLOT_INPUTS}/rkpr3")
        self.assertEqual(slots.launcher({"RKMON_PR_RUNNER": "/l"}), "/l")
        self.assertEqual(slots.launcher({}), "")


if __name__ == "__main__":
    unittest.main()

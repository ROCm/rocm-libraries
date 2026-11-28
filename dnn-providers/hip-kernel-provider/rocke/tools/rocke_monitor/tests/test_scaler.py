# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scaler and the RAM disk (rkmon_lib/scaler.py, rkmon_lib/ramdisk.py), and rk doctor's service checks
(rkmon_lib/reach.py): the current pipes and heavy checks move one step at a time within their fixed ranges, down on
load over the ceiling, up only with headroom and a full setting, with hysteresis; lowering stops nothing; with no
drafter the reviewer endpoint caps no pipe; the RAM disk's footprints are learned (in the container with the slots each
PR holds there) and every start needs room there; a finished review leaves the RAM disk at once; and the service checks
read fakes only (the claude login's test call a local gateway), with their results cached.
"""

import http.server
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (
    coherence,
    config,
    container_layout,
    control,
    deployment,
    file_types,
    jobs,
    paths,
    ramdisk,
    reach,
    review,
    scaler,
    secrets_registry,
    settings_file,
    slots,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

OLD = "2000-01-01T00:00:00Z"
LOCAL = {"backend": "ollama", "url": "http://127.0.0.1:11434", "model": "m:1"}


def a_cfg(work: pathlib.Path, pipes: int = 3, checks: int = 4, **extra) -> dict:
    c = config.load_defaults()
    c.pop("_doc", None)
    c.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"]},
        worker=LOCAL,
        review={"pipes": pipes},
        checks={"limit": checks},
        deployment=dict(c["deployment"], phase="review"),
        reviewer="worker",
    )
    c["paths"] = dict(c["paths"], working_dir=str(work))
    c.update(extra)
    return config.normalize(c)


def host(cpu=40.0, mem=50.0, total=76.0, limit=75):
    return {
        "limit_pct": limit,
        "cpu_pct": cpu,
        "mem_pct": mem,
        "pct": max(cpu, mem),
        "mem_total_gb": total,
        "mem_used_gb": round(total * mem / 100, 2),
    }


def inp(
    cpu=40.0, mem=50.0, pipes=3, heavy=4, cap=5, budget=None, rd=None, reserved=0.0
):
    return {
        "host": host(cpu, mem),
        "ramdisk": rd or {"enabled": False},
        "budget": budget or {},
        "running": {"pipes": pipes, "heavy": heavy},
        "pipe_cap": (cap, "why"),
        "est": dict(ramdisk.SEED_GB),
        "reserved_gb": reserved,
    }


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.store = Store(self.base / "state")
        self.st = empty_state()
        self.cfg = a_cfg(self.work)
        self.t = 1_000_000.0

    def tearDown(self):
        self.tmp.cleanup()

    def adjust(self, i, at="2026-01-01T10:00:00Z"):
        self.t += scaler.SAMPLE_S + 1
        return scaler.adjust(
            self.cfg, self.st, self.store, inp=i, at=at, clock=lambda: self.t
        )


class TestSteps(_Base):
    def test_down_after_two_samples_over_and_never_below_the_range(self):
        rec = {"current": 3}
        self.assertEqual(
            scaler.step("pipes", rec, inp(cpu=90), "2026-01-01T10:00:00Z")[0], 3
        )  # one sample: holds
        new, why = scaler.step("pipes", rec, inp(cpu=90), "2026-01-01T10:01:00Z")
        self.assertEqual(new, 2)
        self.assertIn("CPU 90% over 75%", why)
        rec = {"current": 1, "over": 5}
        new, why = scaler.step("pipes", rec, inp(mem=95), "2026-01-01T10:00:00Z")
        self.assertEqual(new, 1)
        self.assertIn("at the bottom of its range (1)", why)

    def test_up_needs_three_samples_of_headroom_a_full_setting_and_the_cool_down(self):
        rec = {"current": 3, "changed_at": "2026-01-01T09:59:00Z"}
        for k in range(1, 3):
            self.assertEqual(
                scaler.step("pipes", rec, inp(), f"2026-01-01T10:0{k}:00Z")[0], 3
            )
        new, why = scaler.step("pipes", rec, inp(), "2026-01-01T10:03:00Z")
        self.assertEqual(
            new, 3
        )  # three samples, but only 5 min after the last change may it go up
        self.assertIn("min of cool-down left", why)
        rec = {"current": 3, "under": 2, "changed_at": "2026-01-01T09:00:00Z"}
        self.assertEqual(
            scaler.step("pipes", rec, inp(), "2026-01-01T10:04:00Z"),
            (4, "headroom (3 samples in a " "row): one more"),
        )
        # Not every pipe busy: no call for more; at the top: none either; within 10 points of the ceiling: none.
        self.assertIn(
            "2 of 3 busy",
            scaler.step("pipes", {"current": 3, "under": 2}, inp(pipes=2), "x")[1],
        )
        self.assertIn(
            "top of its range (8)",
            scaler.step("heavy", {"current": 8}, inp(heavy=8), "x")[1],
        )
        self.assertIn(
            "within 10 points",
            scaler.step("heavy", {"current": 4, "under": 2}, inp(cpu=70), "x")[1],
        )

    def test_hysteresis_alternating_samples_move_nothing(self):
        rec = {"current": 3}
        for k, cpu in enumerate((90, 40, 90, 40, 90, 40)):
            self.assertEqual(
                scaler.step("pipes", rec, inp(cpu=cpu), f"2026-01-01T10:0{k}:00Z")[0], 3
            )

    def test_the_drafters_endpoint_and_the_reviewer_endpoints_cap(self):
        rec = {"current": 3}
        scaler.step("pipes", rec, inp(budget={"rate_limited": 2}), "x")
        self.assertEqual(
            scaler.step("pipes", rec, inp(budget={"rate_limited": 2}), "y")[0], 2
        )
        self.assertEqual(
            scaler.step(
                "heavy", {"current": 4, "over": 1}, inp(budget={"rate_limited": 2}), "y"
            )[0],
            4,
        )
        self.assertIn(
            "the reviewer endpoint allows 1",
            scaler.step("pipes", {"current": 1}, inp(pipes=1, cap=1), "x")[1],
        )

    def test_the_ramdisk_blocks_a_step_up_without_room(self):
        rd = {
            "enabled": True,
            "mounted": True,
            "free_gb": 1.0,
            "used_gb": 20.0,
            "total_gb": 21.0,
        }
        why = scaler.step("pipes", {"current": 3, "under": 2}, inp(rd=rd), "x")[1]
        self.assertIn("the RAM disk has no room for one more", why)
        # Not mounted, it blocks a step up too; the load sample and the memory's headroom name it the same way.
        off = dict(rd, mounted=False)
        why = scaler.step("pipes", {"current": 3, "under": 2}, inp(rd=off), "x")[1]
        self.assertIn("the RAM disk is not mounted", why)
        self.assertEqual(
            (scaler.load_text(inp(rd=rd)), scaler.load_text(inp(rd=off))),
            (
                "CPU 40%, mem 50%, RAM disk 20.0/21 GiB",
                "CPU 40%, mem 50%, RAM disk not mounted",
            ),
        )
        roomy = inp(mem=70, rd=dict(rd, free_gb=10.0))
        why = scaler.step("heavy", {"current": 4, "under": 2}, roomy, "x")[1]
        self.assertIn("with one more build on the RAM disk", why)

    def test_memory_without_headroom_blocks_a_step_up(self):
        """Memory under the ceiling but within its headroom once one more piece (and what is reserved) is counted:
        the setting holds."""
        for i, want in (
            (inp(mem=70), "memory would be 70% with one more build"),
            (inp(mem=50, reserved=12.0), "memory would be 66% with one more build"),
        ):
            new, why = scaler.step("heavy", {"current": 4, "under": 2}, i, "x")
            self.assertEqual(new, 4, why)
            self.assertIn(want, why)
        self.assertEqual(
            scaler.step("heavy", {"current": 4, "under": 2}, inp(mem=50), "x")[0], 5
        )  # with headroom it goes up


class TestAdjust(_Base):
    def test_a_change_starts_the_cool_down(self):
        """A setting that moved does not move up again until COOLDOWN_MIN minutes later, however much headroom."""
        for k in range(3):
            out = self.adjust(inp(), at=f"2026-01-01T10:0{k}:00Z")
        self.assertTrue(any("parallel builds 4 -> 5" in x for x in out), out)
        heavy = scaler.load_state(self.store)["heavy"]
        self.assertEqual(heavy["changed_at"], "2026-01-01T10:02:00Z")
        for k in range(3, 7):  # 1 to 4 minutes after the change
            out = self.adjust(inp(pipes=4, heavy=5), at=f"2026-01-01T10:0{k}:00Z")
            self.assertFalse(any("parallel builds" in x for x in out), out)
        self.assertIn(
            "min of cool-down left", scaler.load_state(self.store)["heavy"]["reason"]
        )
        out = self.adjust(inp(pipes=4, heavy=5), at="2026-01-01T10:07:00Z")
        self.assertTrue(any("parallel builds 5 -> 6" in x for x in out), out)

    def test_a_clock_stepped_back_never_stops_the_sampling(self):
        self.assertEqual(self.adjust(inp(cpu=90)), [])  # one sample, at self.t
        back = (
            self.t - 3600
        )  # the clock went back an hour: the last sample reads as in the future
        scaler.adjust(
            self.cfg, self.st, self.store, inp=inp(cpu=90), clock=lambda: back
        )
        self.assertEqual(scaler.load_state(self.store)["sampled_at"], back)

    def test_its_state_is_written_whole_and_never_through_a_link(self):
        """The scaler's state is written whole, as the state folder's other JSON files are (state_store.write_file):
        a link at the name a write once put its new file at is never followed, and the state is a file of its own at
        the mode the monitor and the writer share."""
        from rkmon_lib import state_store

        outside = self.base / "outside"
        outside.write_text("theirs\n")
        state = self.store.dir / scaler.STATE
        state.parent.mkdir(parents=True, exist_ok=True)
        (state.parent / (scaler.STATE + ".new")).symlink_to(outside)
        scaler.save_state(self.store, {"pipes": {"current": 2}})
        self.assertEqual(outside.read_text(), "theirs\n")
        self.assertFalse(state.is_symlink())
        self.assertEqual(scaler.load_state(self.store), {"pipes": {"current": 2}})
        self.assertEqual(
            state.stat().st_mode & 0o777, state_store._perms().SHARED_FILE_MODE
        )

    def test_changes_are_logged_with_inputs_and_reason_and_samples_are_spaced(self):
        self.assertEqual(self.adjust(inp(cpu=90)), [])
        self.assertEqual(
            scaler.adjust(
                self.cfg, self.st, self.store, inp=inp(cpu=90), clock=lambda: self.t
            ),
            [],
        )
        lines = self.adjust(inp(cpu=90, mem=80), at="2026-01-01T10:02:00Z")
        self.assertEqual(
            lines[0],
            "SCALER parallel reviews 3 -> 2 (of 1-5): CPU 90% over 75%; memory 80% over 75% (2 samples "
            "in a row): one fewer; CPU 90%, mem 80%",
        )
        self.assertTrue(lines[1].startswith("SCALER parallel builds 4 -> 3 (of 1-8)"))
        log = [
            json.loads(x)
            for x in (self.store.dir / scaler.LOG).read_text().splitlines()
        ]
        self.assertEqual(
            [(r["kind"], r["what"], r["from"], r["to"]) for r in log],
            [("setting", "pipes", 3, 2), ("setting", "heavy", 3 + 1, 3)],
        )
        self.assertEqual(
            set(log[0]["inputs"]),
            {"host", "ramdisk", "budget", "running", "pipe_cap", "est", "reserved_gb"},
        )
        self.assertEqual(
            (
                scaler.current(self.cfg, self.store, "pipes"),
                scaler.current(self.cfg, self.store, "heavy"),
            ),
            (2, 3),
        )
        self.assertEqual(
            review.pipes_in_effect(self.cfg, self.store), 1
        )  # a local Ollama still allows one
        self.assertIsNone(
            scaler.last(self.store)
        )  # the setting records are no draft decisions
        # A new starting value in the configuration starts again from it.
        self.cfg = a_cfg(self.work, pipes=5)
        self.assertEqual(scaler.current(self.cfg, self.store, "pipes"), 5)
        status = "\n".join(scaler.status_lines(self.cfg, self.store))
        self.assertIn("parallel builds 3 of 1–8 — CPU 90%, mem 80%", status)
        self.assertIn("parallel reviews 3 -> 2: CPU 90% over 75%", status)

    def test_lowering_stops_no_running_work(self):
        """Pipes 3 running and heavy jobs 2 running; the setting falls to 1: nothing is stopped, nothing starts."""
        for n in (1, 2, 3):
            review.pipes(self.st).append(
                {
                    "pr": n,
                    "kind": "first-round",
                    "head": str(n) * 40,
                    "since": None,
                    "started": OLD,
                    "stage": "worker",
                    "worker": {"id": f"w{n}"},
                    "checks": None,
                }
            )
            self.st["prs"][str(n)] = {
                "n": n,
                "head": str(n) * 40,
                "status": "open",
                "draft": False,
                "size": 9,
            }
        self.st["prs"]["4"] = {
            "n": 4,
            "head": "4" * 40,
            "status": "open",
            "draft": False,
            "size": 9,
        }
        self.st["jobs"]["heavy"] = [
            {"id": "c1", "started": OLD},
            {"id": "c2", "started": OLD},
        ]
        s = {"pipes": {"start": 3, "current": 1}, "heavy": {"start": 4, "current": 1}}
        scaler.save_state(self.store, s)
        cfg = a_cfg(self.work, reviewer="default")
        killed = []
        with mock.patch.object(
            jobs, "status", lambda store, job: "running" if job else "none"
        ), mock.patch.object(
            jobs, "kill", lambda store, job, *a: killed.append(job["id"])
        ), mock.patch.object(
            review, "start_review", side_effect=AssertionError("started")
        ):
            self.assertEqual(
                review._fill(
                    cfg, self.st, self.store, [{"kind": "FIRST_ROUND", "pr": 4}]
                ),
                [],
            )
            self.assertEqual(jobs.heavy_busy(cfg, self.st, self.store)["id"], "c1")
        self.assertEqual((killed, len(review.pipes(self.st))), ([], 3))
        scaler.save_state(
            self.store,
            {"pipes": {"start": 3, "current": 3}, "heavy": {"start": 4, "current": 4}},
        )
        with mock.patch.object(
            jobs, "status", lambda store, job: "running" if job else "none"
        ):
            self.assertIsNone(jobs.heavy_busy(cfg, self.st, self.store))


class TestNoDraftPipes(_Base):
    def test_with_no_drafter_the_reviewer_endpoint_caps_no_pipe(self):
        p = config.pipes(a_cfg(self.work, reviewer="default"))
        self.assertEqual((p["asked"], p["effective"]), (3, 3))
        self.assertIn("no drafter runs", p["why"])
        self.assertEqual(
            config.pipes_line(a_cfg(self.work, reviewer="default"))[:21],
            "asked 3, running 3, b",
        )
        self.assertEqual(scaler.pipe_cap(a_cfg(self.work, reviewer="default"))[0], 5)
        self.assertEqual(
            config.pipes(a_cfg(self.work))["effective"], 1
        )  # the local Ollama drafting: one
        self.assertEqual(
            review.pipes_in_effect(a_cfg(self.work, draft_stage="none")), 3
        )


class TestRamdisk(_Base):
    def rd_cfg(self, **extra):
        return a_cfg(
            self.work,
            ramdisk={"enabled": True, "path": str(self.base / "rd"), "size_gb": 24},
            **extra,
        )

    def test_paths_and_settings(self):
        cfg = self.rd_cfg()
        p = paths.resolve(cfg)
        self.assertEqual(
            (p["ramdisk"], p["tmp"]),
            (str(self.base / "rd"), str(self.base / "rd" / "tmp")),
        )
        self.assertEqual(config.env(cfg)["RKMON_RAMDISK"], str(self.base / "rd"))
        self.assertEqual(paths.resolve(self.cfg)["ramdisk"], "")
        self.assertEqual(paths.resolve(self.cfg)["tmp"], str(self.work / "tmp"))
        self.assertIn(
            "host.resources.ramdisk_gib is -1; it must be a number of GiB, 0 or more",
            config.problems(
                dict(cfg, ramdisk={"enabled": True, "path": "/r", "size_gb": -1})
            ),
        )
        # The switch is the size's own and the folder the container's: no row names a key the deploy settings file
        # does not have.
        for given in ({"enabled": True}, {"enabled": True, "path": "here"}):
            self.assertEqual(ramdisk.problems({"ramdisk": given}), [], given)

    def test_usage_needs_a_tmpfs_mounted_at_the_path(self):
        cfg, rd = self.rd_cfg(), str(self.base / "rd")
        mounts = self.base / "mounts"
        mounts.write_text(f"tmpfs {rd} tmpfs rw 0 0\n/dev/sda1 / ext4 rw 0 0\n")
        vfs = lambda p: os.statvfs_result(
            (4096, 4096, 6291456, 3932160, 3932160, 0, 0, 0, 0, 255)
        )  # noqa: E731
        u = ramdisk.usage(cfg, str(mounts), vfs)
        self.assertEqual(
            (u["mounted"], u["total_gb"], u["used_gb"], u["free_gb"]),
            (True, 24.0, 9.0, 15.0),
        )
        mounts.write_text("/dev/sda1 / ext4 rw 0 0\n")
        self.assertEqual(
            (
                ramdisk.usage(cfg, str(mounts), vfs)["mounted"],
                ramdisk.usage(cfg, str(mounts))["fs"],
            ),
            (False, "ext4"),
        )

    def test_a_prs_slots_count_toward_its_heavy_job_while_one_runs_else_its_review(
        self,
    ):
        """In the container the slots are on the RAM disk (container_layout.SLOTS_DIR, in RAMDISK): the slots a PR holds
        (its jobs', its named work's, the one kept for its review) count toward its heavy job while one runs, else
        toward its review, never both. On a host (no launcher), or with the RAM disk at another folder, they count
        toward nothing."""
        cfg = self.rd_cfg()
        top = self.base / "rd" / "slots"
        (top / "locks").mkdir(parents=True)
        rec = slots.Records(str(top / "locks"))
        rec.write("rkpr1", {"state": "kept", "pr": "7", "head": "h7", "since": 1})
        rec.write(
            "rkpr2",
            {"state": "taken", "name": "pr8-tests", "pr": "8", "head": "", "since": 1},
        )
        rec.write(
            "rkpr3", {"state": "taken", "name": "pr7-tests", "pr": "7", "since": 1}
        )

        def uses(act, slots_dir=top, launcher="/stand-in/rk_prcode.py"):
            s = {}
            env = {k: v for k, v in os.environ.items() if k != slots.RUNNER_VAR}
            if launcher:
                env[slots.RUNNER_VAR] = launcher
            with mock.patch.multiple(
                container_layout,
                SLOTS_DIR=str(slots_dir),
                SLOT_LOCKS=str(top / "locks"),
            ), mock.patch.dict(os.environ, env, clear=True):
                scaler.learn(
                    cfg,
                    s,
                    act,
                    self.store,
                    lambda folders: float(
                        sum(f.name.startswith("rkpr") for f in folders)
                    ),
                )
            return s["uses"]

        self.assertEqual(
            uses({"pipe": {7, 8}, "heavy": set()}), {"pipe:7": 2.0, "pipe:8": 1.0}
        )
        self.assertEqual(
            uses({"pipe": {7, 8}, "heavy": {7}}),
            {"pipe:7": 0.0, "heavy:7": 2.0, "pipe:8": 1.0},
        )
        self.assertEqual(uses({"pipe": set(), "heavy": {8}}), {"heavy:8": 1.0})
        self.assertEqual(
            uses({"pipe": {7}, "heavy": set()}, launcher=""), {"pipe:7": 0.0}
        )
        self.assertEqual(
            uses({"pipe": {7}, "heavy": set()}, slots_dir=self.base / "elsewhere"),
            {"pipe:7": 0.0},
        )

    def test_footprints_are_learned_and_starts_need_room(self):
        cfg = self.rd_cfg()
        s = {}
        uses = {"pipe": 2.5, "heavy": 0.6}
        measure = lambda folders: (
            uses["pipe"] if "build" not in str(folders[0]) else uses["heavy"]
        )  # noqa: E731
        scaler.learn(cfg, s, {"pipe": {7}, "heavy": {7}}, self.store, measure)
        self.assertEqual(s["peaks"], {"pipe:7": 2.5, "heavy:7": 0.6})
        uses["pipe"] = 1.0
        scaler.learn(
            cfg, s, {"pipe": {7}, "heavy": set()}, self.store, measure
        )  # the heavy job ended: folded in
        self.assertEqual(s["est"], {"heavy": round(2.0 - 1.4 * scaler.DECAY, 3)})
        scaler.learn(
            cfg, s, {"pipe": set(), "heavy": set()}, self.store, measure
        )  # a peak over the estimate wins
        self.assertEqual(s["est"]["pipe"], 2.5)
        self.assertEqual(scaler.reserved_gb(s, {"pipe": {8}, "heavy": set()}), 2.5)
        log = [
            json.loads(x)
            for x in (self.store.dir / scaler.LOG).read_text().splitlines()
        ]
        self.assertEqual(
            [(r["kind"], r["what"], r["peak_gb"]) for r in log],
            [("footprint", "heavy", 0.6), ("footprint", "pipe", 2.5)],
        )
        scaler.save_state(self.store, s)
        self.st["review_pipes"] = [
            {"pr": 8, "kind": "first-round", "worker": {"id": "w8"}}
        ]
        mounted = {
            "enabled": True,
            "mounted": True,
            "free_gb": 6.0,
            "used_gb": 18.0,
            "total_gb": 24.0,
        }
        with mock.patch.object(
            jobs, "status", lambda store, job: "running" if job else "none"
        ):
            # 6 GB free, 2.5 GB held for #8's review: a review (2.5 GB) fits, a heavy job (1.86 GB) too.
            self.assertEqual(
                scaler.room(cfg, self.st, self.store, "pipe", 9, mounted, host(mem=40)),
                (True, ""),
            )
            ok, why = scaler.room(
                cfg,
                self.st,
                self.store,
                "pipe",
                9,
                dict(mounted, free_gb=4.0),
                host(mem=40),
            )
            self.assertFalse(ok)
            self.assertIn(
                "the RAM disk has 4.0 GiB free, 2.5 GiB of it held for the work running; a review needs "
                "about 2.5 GiB",
                why,
            )
            (self.base / "rd" / "wt9").mkdir(
                parents=True
            )  # a pool draft staged its trees: it needs no more
            self.assertTrue(
                scaler.room(
                    cfg,
                    self.st,
                    self.store,
                    "pipe",
                    9,
                    dict(mounted, free_gb=4.0),
                    host(mem=40),
                )[0]
            )
            ok, why = scaler.room(
                cfg, self.st, self.store, "heavy", 9, mounted, host(mem=72)
            )
            self.assertFalse(ok)
            self.assertIn(
                "memory: 54.7 GiB used and 4.4 GiB more on the RAM disk would be 78% of 76 GiB, over 75%",
                why,
            )
            ok, why = scaler.room(
                cfg,
                self.st,
                self.store,
                "heavy",
                9,
                dict(mounted, mounted=False),
                host(),
            )
            self.assertIn("not mounted", why)
        self.assertEqual(
            scaler.room(self.cfg, self.st, self.store, "pipe", 9), (True, "")
        )  # the RAM disk is off

    def test_a_finished_review_leaves_the_ramdisk_at_once(self):
        cfg = self.rd_cfg()
        rd = self.base / "rd"
        build = pathlib.Path(paths.resolve(cfg)["build"])
        build.mkdir(parents=True)
        for n in (5, 6):
            for name in (f"wt{n}", f"wt{n}m"):
                (rd / name).mkdir(parents=True)
                (self.work / name).symlink_to(rd / name)
            (rd / "build" / f"wt{n}m").mkdir(parents=True)
            (build / f"wt{n}m").symlink_to(rd / "build" / f"wt{n}m")
        (rd / "tmp" / "v5fr").mkdir(parents=True)
        calls = []
        run = lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(
            cmd, 0, "", ""
        )  # noqa: E731
        lines = ramdisk.sweep(
            cfg, self.st, self.store, {6}, run=run, clock=lambda: time.time() + 3 * 3600
        )
        self.assertEqual(
            lines,
            [
                "RAMDISK #5: worktrees and builds removed (the review is over)",
                "RAMDISK v5fr: review directory removed (PR #5 is in no review)",
            ],
        )
        self.assertEqual(
            sorted(p.name for p in rd.iterdir()), ["build", "tmp", "wt6", "wt6m"]
        )
        self.assertFalse(
            (self.work / "wt5").is_symlink() or (build / "wt5m").is_symlink()
        )
        self.assertTrue((self.work / "wt6").is_symlink())
        self.assertIn(
            ["git", "-C", paths.resolve(cfg)["repo"], "worktree", "prune"], calls
        )
        self.assertEqual(
            ramdisk.sweep(self.cfg, self.st, self.store, set(), run=run), []
        )  # off: nothing

    def test_a_folder_that_would_not_go_is_journaled_once_and_left_only_when_gone(
        self,
    ):
        """The sweep says a review's folders left the RAM disk only once they are gone: a build that would not go (PR
        code's files, say) is one journal line, not one per sweep, and is tried again each sweep; the line that it left
        comes once it is really gone."""
        cfg = self.rd_cfg()
        rd = self.base / "rd"
        build = pathlib.Path(paths.resolve(cfg)["build"])
        build.mkdir(parents=True)
        (rd / "wt5").mkdir(parents=True)
        (self.work / "wt5").symlink_to(rd / "wt5")
        held = rd / "build" / "wt5m"
        (held / "sub").mkdir(parents=True)
        (build / "wt5m").symlink_to(held)
        real_rmtree = ramdisk._rmtree
        refuse = {str(held)}

        def rmtree(folder):
            if str(folder) in refuse:
                return f"{folder}/sub/x.o: Permission denied"
            return real_rmtree(folder)

        run = lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, "", ""
        )  # noqa: E731

        def cleanup_lines():
            return [
                ln
                for ln in self.store.journal_path.read_text().splitlines()
                if " cleanup " in ln
            ]

        with mock.patch.object(ramdisk, "_rmtree", rmtree):
            first = ramdisk.sweep(cfg, self.st, self.store, set(), run=run)
            again = ramdisk.sweep(cfg, self.st, self.store, set(), run=run)
        self.assertEqual(
            first,
            [
                "RAMDISK #5: 2 folder(s) of a finished review would not go; tried again each sweep"
            ],
        )
        self.assertEqual(again, [])
        (line,) = cleanup_lines()
        self.assertIn(
            "#5: its review is over, but", line
        )  # never "left the RAM disk" while a folder is there
        self.assertIn("did not leave the RAM disk (", line)
        self.assertIn("Permission denied); each sweep tries again", line)
        self.assertFalse((rd / "wt5").exists())  # what could go went
        self.assertTrue(held.is_dir() and (build / "wt5m").is_symlink())
        self.assertIn("5", self.st[ramdisk.STUCK])
        # Now it goes: the line that it left, once, and nothing after.
        self.assertEqual(
            ramdisk.sweep(cfg, self.st, self.store, set(), run=run),
            ["RAMDISK #5: worktrees and builds removed (the review is over)"],
        )
        self.assertEqual(ramdisk.sweep(cfg, self.st, self.store, set(), run=run), [])
        lines = cleanup_lines()
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(
            lines[1].endswith(
                "#5: its review is over; its worktrees and builds left the RAM disk"
            )
        )
        self.assertNotIn(ramdisk.STUCK, self.st)
        self.assertFalse(held.exists() or (build / "wt5m").is_symlink())

    def test_what_pr_code_left_goes_through_its_launcher(self):
        """An install step's folders (755, written as a PR runner) are not this user's to empty: with the launcher,
        `rm -rf` runs as PR code; without it (a host) nothing else is tried."""
        d = self.base / "rd" / "build" / "wt9m"
        (d / "inst").mkdir(parents=True)
        calls = []

        def pr_run(argv, **kw):
            calls.append((argv, kw["cwd"]))
            ramdisk.shutil.rmtree(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "", "")

        refused = lambda folder: (  # noqa: E731
            "" if not folder.exists() else f"{folder}/inst/x: Permission denied"
        )
        with mock.patch.object(ramdisk, "_rmtree", refused):
            self.assertEqual(
                ramdisk.remove_tree(
                    d, "", environ={"RKMON_PR_RUNNER": "/x/rk_prcode.py"}, pr_run=pr_run
                ),
                "",
            )
            self.assertEqual(calls, [(["rm", "-rf", "--", str(d)], str(d.parent))])
            self.assertFalse(d.exists())
            (d / "inst").mkdir(parents=True)
            self.assertIn(
                "Permission denied",
                ramdisk.remove_tree(d, "", environ={}, pr_run=pr_run),
            )
            self.assertEqual(len(calls), 1)

    def test_a_read_only_folder_of_its_own_is_removed(self):
        d = self.base / "rd" / "build" / "wt8m"
        (d / "ro").mkdir(parents=True)
        (d / "ro" / "f").write_text("x")
        os.chmod(d / "ro", 0o500)
        self.assertEqual(ramdisk._rmtree(d), "")
        self.assertFalse(d.exists())

    def test_reviews_awaiting_verification_hold_only_a_residual(self):
        """Three reviews awaiting verification (draft and checks done: their trees no longer grow) hold
        READY_RESERVE_GB each, not a review's whole estimate, so a fourth starts at once; a review in flight still
        holds its estimate over its use, and the whole never passes the RAM disk's size.
        """
        cfg = self.rd_cfg()
        s = {
            "est": {"pipe": 7.0},
            "uses": {"pipe:1": 1.0, "pipe:2": 1.2, "pipe:3": 0.8},
        }
        scaler.save_state(self.store, s)
        self.st["review_ready"] = [
            {
                "pr": n,
                "kind": "follow-up",
                "head": "a" * 40,
                "at": "2026-11-12T03:00:00Z",
            }
            for n in (1, 2, 3)
        ]
        mounted = {
            "enabled": True,
            "mounted": True,
            "free_gb": 12.0,
            "used_gb": 12.0,
            "total_gb": 24.0,
        }
        act = {"pipe": {1, 2, 3}, "heavy": set()}
        self.assertEqual(scaler.reserved_gb(s, act), 18.0)  # each at its full estimate
        self.assertEqual(scaler.awaiting(self.st), {1, 2, 3})
        self.assertEqual(scaler.reserved_gb(s, act, {1, 2, 3}), 1.5)
        with mock.patch.object(jobs, "status", lambda store, job: "none"):
            self.assertEqual(
                scaler.room(cfg, self.st, self.store, "pipe", 4, mounted, host(mem=40)),
                (True, ""),
            )
            self.st["review_pipes"] = [
                {"pr": 5, "kind": "first-round"}
            ]  # in flight: 7 GB more
            ok, why = scaler.room(
                cfg, self.st, self.store, "pipe", 4, mounted, host(mem=40)
            )
            self.assertFalse(ok)
            self.assertIn("12.0 GiB free, 8.5 GiB of it held for the work running", why)
        self.assertEqual(
            scaler.reserved_gb(
                s, {"pipe": set(range(10, 20)), "heavy": set()}, cap=24.0
            ),
            24.0,
        )

    def test_status_shows_the_ramdisk(self):
        cfg = self.rd_cfg()
        with mock.patch.object(
            ramdisk,
            "usage",
            lambda c: {
                "enabled": True,
                "path": "/rd",
                "mounted": True,
                "fs": "tmpfs",
                "size_gb": 24,
                "total_gb": 24.0,
                "used_gb": 9.2,
                "free_gb": 14.8,
            },
        ):
            lines = control.status_lines(
                cfg, self.st, self.store, table={}, web_running=True
            )
        self.assertEqual(
            lines[-1],
            "ramdisk: 9.2 of 24.0 GiB used at /rd (14.8 GiB free); PRs on it: none; held for the work running: 0.0 GiB",
        )
        self.assertIn("  admin dashboard's service: running", lines)
        self.assertIn(
            "  footprints on the RAM disk (running high estimates): a review 1.50 GiB, a build 2.00 GiB",
            lines,
        )


class TestHost(_Base):
    def test_a_memory_limited_container_counts_its_limit(self):
        mem = self.base / "meminfo"
        mem.write_text("MemTotal:  80000000 kB\nMemAvailable: 60000000 kB\n")
        cg = self.base / "cg"
        cg.mkdir()
        (cg / "memory.max").write_text(str(8 * 2**30))
        (cg / "memory.current").write_text(str(7 * 2**30))
        (cg / "memory.stat").write_text(f"anon 1\ninactive_file {2 ** 30}\nshmem 5\n")
        h = scaler.host_load({}, str(mem), str(cg), loadavg=lambda: (16.0, 0, 0))
        self.assertEqual(
            (h["mem_total_gb"], h["mem_used_gb"], h["mem_pct"]), (8.0, 6.0, 75.0)
        )
        (cg / "memory.max").write_text("max")
        h = scaler.host_load({}, str(mem), str(cg), loadavg=lambda: (16.0, 0, 0))
        self.assertEqual(h["mem_pct"], 25.0)
        # A CPU-limited container: its own CPU time since the last sample over its 4 CPUs.
        (cg / "cpu.max").write_text("400000 100000\n")
        (cg / "cpu.stat").write_text("usage_usec 1000000\nuser_usec 1\n")
        prev = {}
        with mock.patch.object(time, "time", lambda: 100.0):
            self.assertEqual(
                scaler.host_load({}, str(mem), str(cg), lambda: (0.0, 0, 0), prev)[
                    "cpu_pct"
                ],
                0.0,
            )
        (cg / "cpu.stat").write_text("usage_usec 121000000\n")
        with mock.patch.object(time, "time", lambda: 160.0):
            self.assertEqual(
                scaler.host_load({}, str(mem), str(cg), lambda: (0.0, 0, 0), prev)[
                    "cpu_pct"
                ],
                50.0,
            )


class FakeCLI:
    """gh and the atlassian CLI as the service checks see them: recorded, answered from a table."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, cmd, env=None, timeout=30):
        self.calls.append(cmd)
        for key, val in self.answers.items():
            if " ".join(cmd[1:]).startswith(key):
                return val
        return 1, "unknown"


class _Gateway(http.server.BaseHTTPRequestHandler):
    """A gateway for reach.http_post, at the start of a path (a gateway's address, before the Messages API's
    /v1/messages): /moved redirects to /elsewhere, /refused refuses the key with an API error whose message quotes it,
    /busy is overloaded, /odd answers an error whose type is not one, /page a page that is no API error, any other
    answers. Each request's path and key are kept (seen)."""

    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append((self.path, self.headers.get("x-api-key")))
        place = self.path.removesuffix("/v1/messages")
        if place == "/moved":
            self.send_response(307)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        code, answer = {
            "/refused": (
                401,
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key sk-test-123",
                    },
                },
            ),
            "/busy": (529, {"type": "error", "error": {"type": "overloaded_error"}}),
            "/odd": (400, {"error": {"type": "Not One: sk-test-123"}}),
            "/page": (502, "<html>sk-test-123</html>"),
        }.get(place, (200, {"content": [{"type": "text", "text": "OK"}]}))
        body = (answer if isinstance(answer, str) else json.dumps(answer)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestDoctorServices(_Base):
    def test_github(self):
        ok = FakeCLI(
            {
                "auth status": (
                    0,
                    "github.com\n  ✓ Logged in to github.com account op-login (keyring)",
                ),
                "api rate_limit": (
                    0,
                    json.dumps(
                        {
                            "resources": {
                                "core": {"limit": 5000, "remaining": 4321, "reset": 0}
                            }
                        }
                    ),
                ),
            }
        )
        c = reach.github(self.cfg, run=ok)[0]
        self.assertEqual(
            (c.status, c.found),
            (
                "PASS",
                "github.com: account op-login, auth OK; rate limit 4321 of "
                "5000 left (resets 00:00Z)",
            ),
        )
        bad = FakeCLI(
            {
                "auth status": (
                    1,
                    "X Failed to log in to github.com account op-login\n- The token is "
                    "invalid.",
                )
            }
        )
        c = reach.github(self.cfg, run=bad)[0]
        self.assertEqual(
            (c.status, c.found),
            (
                "FAIL",
                "github.com: the login is expired or invalid; it blocks every GitHub read and write",
            ),
        )
        self.assertNotIn(["gh", "api", "rate_limit"], bad.calls)

    def test_jira_reads_only_through_the_cli(self):
        cfg = dict(
            self.cfg,
            jira={"project": "ABC", "url": "https://jira.example.org"},
            atlassian={"site": "example", "enabled": True},
        )
        cli = FakeCLI(
            {
                "version": (0, "atlassian version 1.2.3"),
                "auth status": (0, "Logged in"),
                "--site example site active": (0, ""),
                "--site example --json jira project versions ABC": (0, "[]"),
            }
        )
        with mock.patch("shutil.which", lambda b: f"/usr/bin/{b}"):
            j = reach.jira(cfg, run=cli, get=lambda url, h=None: (200, ""))
        self.assertEqual(
            [(x.name, x.status) for x in j],
            [("JIRA site", "PASS"), ("JIRA project", "PASS")],
        )
        writes = [
            cmd
            for cmd in cli.calls
            if any(w in cmd for w in ("create", "update", "delete", "comment"))
        ]
        self.assertEqual(writes, [])
        # The access token expired: the CLI is still logged in, and the reads (which refresh it) run, as the CLI
        # check and the session's login check say.
        expired = FakeCLI(
            dict(
                cli.answers,
                **{
                    "auth status": (
                        1,
                        "Logged in to Atlassian Cloud\n  Token:    expired",
                    )
                },
            )
        )
        with mock.patch("shutil.which", lambda b: f"/usr/bin/{b}"):
            j = reach.jira(cfg, run=expired, get=lambda url, h=None: (200, ""))
        self.assertEqual([x.status for x in j], ["PASS", "PASS"])
        self.assertIn(
            [
                "/usr/bin/atlassian",
                "--site",
                "example",
                "--json",
                "jira",
                "project",
                "versions",
                "ABC",
            ],
            expired.calls,
        )

    def test_the_doctor_has_no_confluence_check(self):
        """The monitor never writes to Confluence, and no settings file holds a Confluence space: the doctor's
        remote checks (reach.REMOTE) hold no Confluence row."""
        self.assertNotIn("confluence", reach.REMOTE)

    def test_jira_names_the_site_to_the_cli_by_its_name(self):
        """The deploy settings file gives the site as JIRA's address: the CLI's calls and the rows name it as the CLI
        names a site, by its name (atlassian.jira_site), never by the address."""
        cfg = dict(
            self.cfg,
            jira={"project": "ABC", "url": "https://jira.example.com"},
            atlassian={"site": "https://jira.example.com", "enabled": True},
        )
        cli = FakeCLI(
            {
                "version": (0, "atlassian version 1.2.3"),
                "auth status": (0, "Logged in"),
                "--site jira --json jira project versions ABC": (0, "[]"),
            }
        )
        with mock.patch("shutil.which", lambda b: f"/usr/bin/{b}"):
            rows = reach.jira(cfg, run=cli, get=lambda url, h=None: (200, ""))
        self.assertEqual(
            [(r.name, r.status, r.found) for r in rows[1:]],
            [
                (
                    "JIRA project",
                    "PASS",
                    "ABC on jira: readable through the atlassian CLI",
                ),
            ],
        )
        # the read this row makes (the CLI's own status is atlassian.status's)
        reads = [c for c in cli.calls if "versions" in c]
        self.assertEqual([c[c.index("--site") + 1] for c in reads], ["jira"])

    @mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"})
    def test_the_jira_project_row_fixes_what_the_cli_lacks(self):
        """While the atlassian CLI cannot read, the JIRA project's row says why it is not checked, and its fix is what
        makes the CLI ready, as the CLI's own row gives it (atlassian.cli_fix): an image of this package, which carries
        the CLI, while it is not found; its login while it is not logged in.
        """
        from rkmon_lib import atlassian, doctor

        cfg = dict(
            self.cfg,
            jira={"project": "ABC", "url": "https://jira.example.org"},
            atlassian={"site": "example", "enabled": True},
        )
        out = FakeCLI(
            {
                "version": (0, "atlassian version 1.2.3"),
                "auth status": (1, "Not logged in"),
            }
        )
        login = "./rkmon login atlassian"
        for why, which, cli, fix in (
            (
                "not installed here",
                lambda b: None,
                FakeCLI({}),
                str(doctor.image_fix()),
            ),
            ("not logged in", lambda b: f"/usr/bin/{b}", out, login),
        ):
            with self.subTest(why), mock.patch("shutil.which", which):
                rows = reach.jira(cfg, run=cli, get=lambda url, h=None: (200, ""))
                line = atlassian.check_line(cfg, atlassian.status(cfg, run=cli))
                project = {r.name: r for r in rows}["JIRA project"]
                self.assertEqual(project.status, "WARN")
                self.assertIn(
                    f"ABC on example: not checked: the atlassian CLI is {why}",
                    project.found,
                )
                self.assertEqual([str(r.fix) for r in (project, line)], [fix] * 2)
        self.assertNotEqual(str(doctor.image_fix()), login)

    def test_a_service_that_does_not_answer_is_fixed_on_the_deployment_s_network(self):
        """The JIRA site and the team chat's host, when nothing answers: the fix names the deployment's network, where
        the doctor runs (in the container, the deployment's own), as the claude login's test call does.
        """
        cfg = dict(
            self.cfg,
            jira={"project": "ABC", "url": "https://jira.example.org"},
            atlassian={"site": "example", "enabled": True},
        )
        cli = FakeCLI(
            {
                "version": (0, "atlassian version 1.2.3"),
                "auth status": (0, "Logged in"),
                "--site example site active": (0, ""),
                "--site example --json jira project versions ABC": (0, "[]"),
            }
        )
        with mock.patch("shutil.which", lambda b: f"/usr/bin/{b}"):
            site = reach.jira(cfg, run=cli, get=lambda url, h=None: (0, "timed out"))[0]
        self.assertEqual((site.name, site.status), ("JIRA site", "WARN"))
        self.assertTrue(
            str(site.fix).startswith("fix the deployment's network to it, then "),
            site.fix,
        )
        hook = self.base / "hook"
        hook.write_text("https://chat.example.org/hooks/SECRET-PART\n")
        cfg = dict(self.cfg, teams={"channel": "ops", "webhook_file": str(hook)})
        (chat,) = reach.chat(cfg, reach=lambda h: "timed out")
        self.assertEqual(chat.status, "WARN")
        self.assertTrue(
            str(chat.fix).startswith(
                "fix the deployment's network to the chat service, then "
            ),
            chat.fix,
        )

    def test_chat_is_reachability_only_and_never_shows_the_webhook(self):
        hook = self.base / "hook"
        hook.write_text("https://chat.example.org/hooks/SECRET-PART\n")
        cfg = dict(self.cfg, teams={"channel": "ops", "webhook_file": str(hook)})
        seen = []
        c = reach.chat(cfg, reach=lambda h: seen.append(h) or "")[0]
        self.assertEqual((c.status, seen), ("PASS", ["chat.example.org"]))
        self.assertNotIn("SECRET", f"{c.found} {c.fix}")
        self.assertNotIn("chat.example.org", c.found)

    def test_chat_off_names_each_kind_of_alert(self):
        """Each kind of alert has its own switch: with the alerts on, neither kind on and no webhook, the row is off
        and names both kinds."""
        cfg = dict(
            self.cfg,
            features={
                "alerts": {"enabled": True, "develop_breaks": False, "needs_you": False}
            },
            teams={"channel": "", "webhook_file": str(self.base / "no-hook")},
        )
        [row] = reach.chat(cfg, reach=lambda h: "")
        self.assertEqual(row.status, "OFF")
        self.assertIn(
            "the develop break alerts and the needs-you alerts are off", row.found
        )

    def test_chat_needs_no_message_label_and_never_names_it(self):
        """The message label only labels each message (the webhook decides the channel): the row is the same
        with a blank label or with one, and names no label."""
        hook = self.base / "hook"
        hook.write_text("https://chat.example.org/hooks/SECRET-PART\n")
        rows = [
            reach.chat(
                dict(self.cfg, teams={"channel": label, "webhook_file": str(hook)}),
                reach=lambda h: "",
            )[0]
            for label in ("", "ops")
        ]
        self.assertEqual(rows[0].status, "PASS", rows[0].found)
        self.assertEqual(
            (rows[1].status, rows[1].found), (rows[0].status, rows[0].found)
        )
        self.assertNotIn("ops", rows[1].found)

    def test_gateways_check_the_key_without_showing_it(self):
        cfg = dict(
            self.cfg,
            reviewer="gw",
            model_bindings={
                "gw": {
                    "url": "https://gw.example.org/v1",
                    "api_key_secret": "gw_api_key",
                }
            },
        )
        # the key, where the secrets registry keeps the secret the binding names (bindings.binding_by_name)
        key = pathlib.Path(secrets_registry.binding_key_file(cfg, "gw_api_key"))
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text("sk-test-123\n")
        sent = []
        get = lambda url, h=None: sent.append((url, h)) or (401, "")  # noqa: E731
        c = reach.gateways(cfg, get=get)[0]
        self.assertEqual(
            (c.status, c.found),
            ("FAIL", "https://gw.example.org/v1: HTTP 401: the key is refused"),
        )
        self.assertEqual(
            sent[0],
            (
                "https://gw.example.org/v1/models",
                {"Authorization": "Bearer sk-test-123"},
            ),
        )
        self.assertNotIn("sk-test", f"{c.found} {c.fix}")
        self.assertEqual(
            reach.gateways(dict(self.cfg, reviewer="default"), get=get), []
        )
        # a reviewer no binding answers, named as the deploy settings file names it
        c = reach.gateways(dict(self.cfg, reviewer="gone"), get=get)[0]
        self.assertEqual(
            (c.name, c.status, c.found),
            (
                "model gateway (gone)",
                "WARN",
                "features.review_automation.passes.first_review_pass.model names it, but model_bindings has no such binding",
            ),
        )

    def test_the_claude_login_s_test_call_is_sent_as_the_session_sends_its_requests(
        self,
    ):
        """The claude login's test call (reach.claude_test): one message of one token to the Messages API
        at integrations.claude_code's address, on the session's model by its id, with the claude login's key in the
        headers the session sends it in (its key variable's, and the named header after its word); at Anthropic's own
        API when no address is set. Answered: a PASS that says where, and on which model. Never the key in the row.
        """
        from rkmon_lib import orchestrator

        cfg = dict(
            self.cfg,
            claude=dict(
                self.cfg.get("claude") or {},
                gateway="https://gw.example.org/anthropic/",
                api_key_header="X-Gateway-Sub",
                api_key_scheme="Token",
            ),
        )
        model = config.claude_model_id(cfg)
        sent = []

        def post(url, headers, body):
            sent.append((url, headers, json.loads(body)))
            return 200, ""

        with mock.patch.object(orchestrator, "read_key", return_value="sk-test-123"):
            row = reach.claude_test(cfg, reach.again({}), post=post)
            self.assertEqual(
                sent,
                [
                    (
                        "https://gw.example.org/anthropic/v1/messages",
                        {
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                            "x-api-key": "sk-test-123",
                            "X-Gateway-Sub": "Token sk-test-123",
                        },
                        {
                            "model": model,
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "Answer OK."}],
                        },
                    )
                ],
            )
            self.assertEqual(
                (row.name, row.status, row.found, row.fix),
                (
                    "claude",
                    "PASS",
                    f"one test call answered, at https://gw.example.org/anthropic/ (model {model})",
                    None,
                ),
            )
            sent.clear()
            row = reach.claude_test(
                dict(cfg, claude=dict(cfg["claude"], gateway="")),
                reach.again({}),
                post=post,
            )
        self.assertEqual(sent[0][0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(
            row.found, f"one test call answered, at Anthropic's own API (model {model})"
        )

    @mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"})
    def test_the_claude_login_s_row_says_why_its_test_call_did_not_pass(self):
        """A key refused (HTTP 401 or 403), or none: a FAIL whose fix is the claude login's command, which says what
        it blocks. No answer: a FAIL whose fix is the deployment's network or an edit of the address; another refusal
        (a wrong address or model): a FAIL whose fix is an edit of either. Each edit names the key and the deploy
        settings file, then rkmon confirm and rkmon apply, as every settings fix does, then the call again. An answer
        that says nothing of the key (HTTP 429, a server's error): a WARN whose fix makes the call again. Each says
        the call's status and the API error's type, never the key; with no key no call is made.
        """
        from rkmon_lib import doctor, orchestrator

        cfg = dict(
            self.cfg,
            claude=dict(self.cfg.get("claude") or {}, gateway="https://gw.example.org"),
        )
        login = str(doctor.claude_login_fix(cfg))
        self.assertIn(" login claude", login)
        blocks = "it blocks Claude's part of each tick: verification, gate decisions, posting"
        model = config.claude_model_id(cfg)
        again = str(reach.again({}))

        def row(status, why=""):
            with mock.patch.object(
                orchestrator, "read_key", return_value="sk-test-123"
            ):
                c = reach.claude_test(
                    cfg, reach.again({}), post=lambda *a: (status, why)
                )
            self.assertNotIn("sk-test", f"{c.found} {c.fix}")
            return c.status, c.found, str(c.fix)

        for status in (401, 403):
            self.assertEqual(
                row(status, "authentication_error"),
                (
                    "FAIL",
                    f"the key is refused, at https://gw.example.org (HTTP {status}, authentication_error); "
                    f"{blocks}",
                    login,
                ),
            )
        self.assertEqual(
            row(0, "URLError: [Errno 111] Connection refused"),
            (
                "FAIL",
                "the test call had no answer, at https://gw.example.org (URLError: [Errno 111] Connection refused); "
                f"{blocks}",
                "fix the deployment's network to it, or set integrations.claude_code.base_url in your deploy "
                "settings file to an address the deployment reaches, or blank for Anthropic's own API, then ./rkmon "
                f"confirm and ./rkmon apply, then {again}",
            ),
        )
        self.assertEqual(
            row(404, "not_found_error"),
            (
                "FAIL",
                f"the test call was refused, at https://gw.example.org (HTTP 404, not_found_error); {blocks}",
                "set integrations.claude_code.base_url in your deploy settings file to the address of a gateway "
                "that serves the model, or set features.review_automation.orchestrator.model in your deploy "
                f"settings file to a model the address serves (now {model}), then ./rkmon confirm and ./rkmon apply, "
                f"then {again}",
            ),
        )
        for status in (429, 529):
            self.assertEqual(
                row(status, "overloaded_error"),
                (
                    "WARN",
                    f"the test call could not be answered, at https://gw.example.org (HTTP {status}, "
                    "overloaded_error): the key is not confirmed",
                    f"try again in a few minutes: {again}",
                ),
            )
        self.assertEqual(
            row(500)[1].split(" (")[1], "HTTP 500): the key is not confirmed"
        )
        with mock.patch.object(orchestrator, "read_key", return_value=""):
            c = reach.claude_test(
                cfg, reach.again({}), post=lambda *a: self.fail("no key: no call")
            )
        self.assertEqual(
            (c.status, c.found, str(c.fix)), ("FAIL", f"no Claude key; {blocks}", login)
        )
        # rk claude-test's own: the call again is the login's command, which makes it too
        with mock.patch.object(orchestrator, "read_key", return_value="sk-test-123"):
            c = reach.claude_test(
                cfg, doctor.claude_login_fix(cfg), post=lambda *a: (529, "")
            )
        self.assertEqual(str(c.fix), f"try again in a few minutes: {login}")

    def test_the_claude_login_s_row_is_the_container_s(self):
        """In the container the claude login has one row among the remote checks (reach.claude_login): its test call
        while review automation is on, OFF while it is off, for then no Claude session runs to read the key. None on a
        host, where the Claude session is a person's Claude Code, with its own login (the Claude Code row). A change of
        the session's address or model checks again (the cache key)."""
        from rkmon_lib import features, orchestrator

        post = lambda *a: (200, "")  # noqa: E731
        with mock.patch.object(orchestrator, "read_key", return_value="sk-test-123"):
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):
                self.assertEqual(reach.claude_login(self.cfg, post=post), [])
            with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
                with mock.patch.object(features, "uses_claude", lambda cfg: True):
                    (row,) = reach.claude_login(self.cfg, post=post)
                self.assertEqual((row.name, row.status), ("claude", "PASS"))
                with mock.patch.object(features, "uses_claude", lambda cfg: False):
                    (row,) = reach.claude_login(
                        self.cfg, post=lambda *a: self.fail("off: no call")
                    )
                self.assertEqual(
                    (row.name, row.status, row.found),
                    (
                        "claude",
                        "OFF",
                        "review automation is off: no Claude session runs, so none reads the key",
                    ),
                )
        self.assertIn("claude_login", reach.REMOTE)
        a = reach._digest(self.cfg)
        moved = dict(
            self.cfg,
            claude=dict(self.cfg.get("claude") or {}, gateway="https://gw.example.org"),
        )
        self.assertNotEqual(reach._digest(moved), a)
        other = dict(
            self.cfg, orchestrator=dict(self.cfg.get("orchestrator") or {}, model="m-x")
        )
        self.assertNotEqual(reach._digest(other), a)

    def test_the_test_call_follows_no_redirect_and_keeps_only_an_error_s_type(self):
        """reach.http_post, the test call's POST: a redirect is its answer, never followed (the key goes to its own
        address alone); of an answer only the API error's type is kept, never its message, which may quote the
        request; nothing answering is status 0 and why, never the key: a closed port, a key http.client refuses to
        send, an address that answers no HTTP, which raises nothing. reach.http_get says why the same way.
        """
        _Gateway.seen = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        h = {"x-api-key": "sk-test-123", "content-type": "application/json"}
        self.assertEqual(reach.http_post(f"{base}/ok", h, b"{}"), (200, ""))
        self.assertEqual(reach.http_post(f"{base}/moved", h, b"{}"), (307, ""))
        self.assertEqual(
            reach.http_post(f"{base}/refused", h, b"{}"), (401, "authentication_error")
        )
        self.assertEqual(reach.http_post(f"{base}/odd", h, b"{}"), (400, ""))
        self.assertEqual(reach.http_post(f"{base}/page", h, b"{}"), (502, ""))
        self.assertEqual(
            [path for path, _key in _Gateway.seen],
            ["/ok", "/moved", "/refused", "/odd", "/page"],
        )  # never /elsewhere
        self.assertEqual({key for _path, key in _Gateway.seen}, {"sk-test-123"})
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        status, why = reach.http_post(f"http://127.0.0.1:{port}/", h, b"{}", timeout=5)
        self.assertEqual(status, 0)
        self.assertTrue(why.startswith("URLError: "), why)
        self.assertNotIn("sk-test", why)
        # A key http.client refuses to send (a line break in a header's value) and an address that answers no HTTP:
        # nothing answered, the reason the error's type alone, which never quotes the key.
        bad = dict(h, **{"x-api-key": "sk-invented-000\nX"})
        junk = socket.socket()
        junk.bind(("127.0.0.1", 0))
        junk.listen(4)
        self.addCleanup(junk.close)

        def no_http():
            while True:
                try:
                    c, _ = junk.accept()
                except OSError:
                    return
                with c:
                    c.recv(65536)
                    c.sendall(b"not http\r\n")

        threading.Thread(target=no_http, daemon=True).start()
        odd = f"http://127.0.0.1:{junk.getsockname()[1]}/"
        for call in (
            lambda url, hd: reach.http_post(url, hd, b"{}", timeout=5),
            lambda url, hd: reach.http_get(url, hd, timeout=5),
        ):
            self.assertEqual(call(f"http://127.0.0.1:{port}/", bad), (0, "ValueError"))
            self.assertEqual(call(odd, h), (0, "BadStatusLine"))

    def test_rk_claude_test_prints_the_row_and_fails_on_a_fail(self):
        """`rk claude-test`, which `rkmon login claude` runs once the key is set: the test call's row, here from a
        gateway on this machine; exit 4 on a FAIL, else 0. Its fix, and its call again, is the claude login's command,
        for before the first start no doctor runs. The key is never printed."""
        import contextlib
        import io

        import rkmon
        from rkmon_lib import doctor, orchestrator

        _Gateway.seen = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def claude_test(gateway):
            cfg = dict(
                self.cfg, claude=dict(self.cfg.get("claude") or {}, gateway=gateway)
            )
            out = io.StringIO()
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch.object(
                orchestrator, "read_key", return_value="sk-test-123"
            ), contextlib.redirect_stdout(
                out
            ):
                code = rkmon.main(["claude-test"])
            self.assertNotIn("sk-test", out.getvalue())
            return (
                code,
                " ".join(out.getvalue().split()),
                str(doctor.claude_login_fix(cfg)),
            )

        code, out, login = claude_test(base)
        self.assertEqual(code, 0, out)
        self.assertIn(f"PASS claude: one test call answered, at {base} (model ", out)
        code, out, login = claude_test(f"{base}/refused")
        self.assertEqual(code, 4, out)
        self.assertIn(
            f"FAIL claude: the key is refused, at {base}/refused (HTTP 401, authentication_error); it blocks ",
            out,
        )
        self.assertIn(f"fix: {login}", out)
        code, out, login = claude_test(f"{base}/busy")
        self.assertEqual(code, 0, out)
        self.assertIn(
            f"WARN claude: the test call could not be answered, at {base}/busy (HTTP 529, overloaded_error)",
            out,
        )
        self.assertIn(f"fix: try again in a few minutes: {login}", out)
        self.assertEqual(
            [path for path, _key in _Gateway.seen],
            ["/v1/messages", "/refused/v1/messages", "/busy/v1/messages"],
        )

    def test_the_ramdisk_check(self):
        cfg = a_cfg(self.work, ramdisk={"enabled": True, "path": "/rd", "size_gb": 24})
        u = {"enabled": True, "path": "/rd", "mounted": False, "fs": "overlay"}
        self.assertEqual(reach.ramdisk_check(cfg, usage=u)[0].status, "FAIL")
        u = dict(u, mounted=True, total_gb=24.0, used_gb=1.0, free_gb=23.0)
        c = reach.ramdisk_check(cfg, usage=u, cgroup=str(self.base))
        self.assertEqual([x.status for x in c], ["PASS"])
        # The room left, not the size: 23 GiB free holds 15 more reviews at 1.5 GiB
        self.assertIn(
            "23.0 GiB free: room for about 15 more reviews at 1.5 GiB", c[0].found
        )
        full = dict(
            u, used_gb=22.0, free_gb=2.0
        )  # most of it in use: only the room left counts
        c = reach.ramdisk_check(cfg, usage=full, cgroup=str(self.base))
        self.assertIn("room for about 1 more reviews", c[0].found)
        c = reach.ramdisk_check(
            cfg, usage=dict(full, free_gb=1.0), cgroup=str(self.base)
        )
        self.assertEqual(c[0].status, "WARN")
        c = reach.ramdisk_check(cfg, usage=dict(u, total_gb=12.0))
        self.assertEqual(c[0].status, "WARN")  # under 24 GiB
        # the size as the deploy settings file names it
        self.assertTrue(
            c[0].found.endswith("; host.resources.ramdisk_gib is 24"), c[0].found
        )
        small = a_cfg(self.work, ramdisk={"enabled": True, "path": "/rd", "size_gb": 2})
        self.assertEqual(
            reach.ramdisk_check(small, usage=dict(u, total_gb=2.0, used_gb=0.0))[
                0
            ].found,
            "/rd: tmpfs of 2.0 GiB, 0.0 GiB used; one review (1.5 GiB) and one build (2.0 GiB) do not fit; it blocks "
            "reviews: they wait for room",
        )
        (self.base / "memory.max").write_text(str(8 * 2**30))
        c = reach.ramdisk_check(cfg, usage=u, cgroup=str(self.base))
        self.assertEqual(c[-1].name, "ramdisk and the memory limit")
        self.assertEqual(reach.ramdisk_check(self.cfg)[0].status, "OFF")

    def test_the_memory_limit_holds_the_ramdisk_and_the_room_for_the_work(self):
        """The RAM disk (a tmpfs) counts against the container's memory: a limit under its size plus the room for the
        work warns, and its fix asks for that sum in the deploy settings file, wherever this runs. The room has one home
        (deployment.WORK_GB): this row's threshold and fix, and confirm's check of the deploy settings file's memory
        (coherence.py), the container's memory limit, both read it."""
        cfg = a_cfg(self.work, ramdisk={"enabled": True, "path": "/rd", "size_gb": 24})
        u = {
            "enabled": True,
            "path": "/rd",
            "mounted": True,
            "fs": "tmpfs",
            "total_gb": 24.0,
            "used_gb": 1.0,
            "free_gb": 23.0,
        }
        here = {"RKMON_CONTAINER": ""}
        box = {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "./rkmon"}

        def row(gib, environ=here):
            (self.base / "memory.max").write_text(str(gib * 2**30))
            with mock.patch.dict(os.environ, environ):
                rows = reach.ramdisk_check(
                    cfg, usage=u, cgroup=str(self.base), exec_check=lambda p: ""
                )
            return [c for c in rows if c.name == "ramdisk and the memory limit"]

        for gib in (27, 29, 31):
            with self.subTest(limit=gib):
                [host], [inside] = row(gib), row(gib, box)
                self.assertEqual((host.status, inside.status), ("WARN", "WARN"))
                for fix in (host.fix, inside.fix):
                    self.assertIn(
                        "in your deploy settings file to at least 32: the RAM disk's size plus room for the work",
                        str(fix),
                    )
                # the change comes to run as every setting's does: no stop and start
                self.assertTrue(
                    str(inside.fix).endswith(
                        "the work, then ./rkmon confirm and ./rkmon apply"
                    ),
                    inside.fix,
                )
        self.assertEqual((row(32), row(32, box)), ([], []))
        with mock.patch.object(deployment, "WORK_GB", 6):
            self.assertEqual(row(30), [])
            self.assertIn("to at least 30:", str(row(29)[0].fix))
            for gib, status in ((30, "PASS"), (29, "FAIL")):
                values = {"host": {"resources": {"memory_gib": gib, "ramdisk_gib": 24}}}
                doc = settings_file.Document(
                    file_types.DEPLOY,
                    file_types.DEPLOY.version,
                    values,
                    path=self.work / "rkmon-deploy.yaml",
                )
                rows = coherence.resources(doc, 64.0)
                self.assertEqual(
                    [c.status for c in rows if c.name == "memory"], [status], gib
                )

    def test_the_ramdisk_must_run_programs(self):
        """Docker's tmpfs default is noexec: a helper a PR's test writes to its temporary folder there cannot run, so
        rk doctor writes a tiny script there and runs it."""
        cfg = a_cfg(self.work, ramdisk={"enabled": True, "path": "/rd", "size_gb": 24})
        u = {
            "enabled": True,
            "path": "/rd",
            "mounted": True,
            "fs": "tmpfs",
            "total_gb": 24.0,
            "used_gb": 1.0,
            "free_gb": 23.0,
        }
        c = reach.ramdisk_check(
            cfg,
            usage=u,
            cgroup=str(self.base),
            exec_check=lambda p: "the mount is noexec",
        )
        bad = [x for x in c if x.name == "ramdisk executes programs"]
        self.assertEqual((bad[0].status, "exec" in str(bad[0].fix)), ("FAIL", True))
        self.assertEqual(
            reach.ramdisk_exec(str(self.base)), ""
        )  # a folder that runs programs
        self.assertEqual(
            reach.ramdisk_exec("/nonexistent-ramdisk"), ""
        )  # not there: the mount check says so
        with mock.patch.object(
            reach.subprocess, "run", side_effect=PermissionError("noexec")
        ):
            self.assertIn("noexec", reach.ramdisk_exec(str(self.base)))
        self.assertEqual(
            list(self.base.glob(".rkmon-exec-*")), []
        )  # the probe is removed

    def test_the_cache_key_names_the_code(self):
        """A remote check cached by other code (the code before an update's swap) is not this code's result."""
        a = reach._digest(self.cfg)
        with mock.patch.object(reach, "code_id", lambda root=None: "another"):
            self.assertNotEqual(reach._digest(self.cfg), a)
        self.assertEqual(reach._digest(self.cfg), a)

    def test_a_login_or_a_secret_set_since_a_cached_run_checks_again(self):
        """Every login and secret the remote checks read is part of the cache key (reach.marks): the Claude session's
        key, gh's read login and the atlassian CLI's login (a login, a token refresh), each set after a cached run,
        check again; the cache never holds a value."""
        from rkmon_lib import secrets_registry as sr

        cfg = dict(
            self.cfg,
            paths=dict(
                self.cfg.get("paths") or {}, secrets_dir=str(self.base / "secrets")
            ),
        )
        runs = []
        checks = {
            "github": lambda cfg: runs.append(1)
            or [reach.Check("GitHub", "PASS", "ok")]
        }
        now = [1000.0]

        def run():
            now[0] += 10
            reach.remote_checks(
                cfg, self.store.dir, checks=checks, clock=lambda: now[0]
            )
            return len(runs)

        self.assertEqual((run(), run()), (1, 1))  # the second from the cache
        for name in ("claude", "github_read", "atlassian"):
            entry = sr.entry(cfg, name)
            place = pathlib.Path(sr.path(cfg, entry))
            self.assertTrue(str(place).startswith(str(self.base)), place)
            files = sr.held_files(entry)
            target = place / files[0] if files else place
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("not-a-real-value\n")
            before = len(runs)
            self.assertEqual((run(), run()), (before + 1, before + 1), name)
        self.assertNotIn("not-a-real-value", (self.store.dir / reach.CACHE).read_text())

    def test_the_login_mark_is_the_credentials_files_time_and_inode_never_its_content_or_size(
        self,
    ):
        d = self.base / "secrets" / "atlassian"
        d.mkdir(parents=True)
        cfg = dict(
            self.cfg, paths=dict(self.cfg.get("paths") or {}, secrets_dir=str(d.parent))
        )
        self.assertEqual(reach.atlassian.login_mark(cfg), "")
        (d / "credentials").write_text("not-a-real-token")
        s = (d / "credentials").stat()
        mark = reach.atlassian.login_mark(cfg)
        self.assertEqual(mark, f"{s.st_mtime_ns}:{s.st_ino}")
        self.assertNotIn("token", mark)

    def test_results_are_cached_a_few_minutes(self):
        runs = []
        checks = {
            "github": lambda cfg: runs.append(1)
            or [reach.Check("GitHub", "PASS", "ok")]
        }
        now = [1000.0]
        a, said = reach.remote_checks(
            self.cfg, self.store.dir, checks=checks, clock=lambda: now[0]
        )
        self.assertEqual((len(runs), said[:19]), (1, "service checks run "))
        now[0] += 60
        b, said = reach.remote_checks(
            self.cfg, self.store.dir, checks=checks, clock=lambda: now[0]
        )
        self.assertEqual(
            (len(runs), [x.found for x in b], said[:25]),
            (1, ["ok"], "service checks from 1 min"),
        )
        self.assertEqual(
            oct(os.stat(self.store.dir / reach.CACHE).st_mode & 0o777), "0o600"
        )
        reach.remote_checks(
            self.cfg, self.store.dir, fresh=True, checks=checks, clock=lambda: now[0]
        )
        now[0] += reach.CACHE_S
        reach.remote_checks(
            self.cfg, self.store.dir, checks=checks, clock=lambda: now[0]
        )
        reach.remote_checks(
            dict(self.cfg, operator={"github_login": "other"}),
            self.store.dir,
            checks=checks,
            clock=lambda: now[0],
        )
        self.assertEqual(
            len(runs), 4
        )  # --fresh, an old cache and another configuration each check again
        failing = {"github": lambda cfg: 1 / 0}
        c, _ = reach.remote_checks(self.cfg, None, checks=failing)
        self.assertEqual(
            (c[0].status, c[0].found[:30]), ("WARN", "the check failed: ZeroDivision")
        )

    def test_the_cache_is_written_whole_and_never_through_a_link(self):
        """The service checks' cache is written whole (whole_file.write), its maker's alone: a link at the name a write
        once put its new file at is never followed."""
        outside = self.base / "outside"
        outside.write_text("theirs\n")
        cache = self.store.dir / reach.CACHE
        cache.parent.mkdir(parents=True, exist_ok=True)
        (cache.parent / (reach.CACHE + ".new")).symlink_to(outside)
        checks = {"github": lambda cfg: [reach.Check("GitHub", "PASS", "ok")]}
        reach.remote_checks(self.cfg, self.store.dir, checks=checks, clock=lambda: 1.0)
        self.assertEqual(outside.read_text(), "theirs\n")
        self.assertFalse(cache.is_symlink())
        self.assertEqual(oct(cache.stat().st_mode & 0o777), "0o600")
        rows, said = reach.remote_checks(
            self.cfg, self.store.dir, checks={}, clock=lambda: 2.0
        )
        self.assertEqual(
            ([r.found for r in rows], said[:21]), (["ok"], "service checks from 0")
        )

    def test_a_cached_row_is_the_row_that_ran(self):
        """The cache keeps each row's fix by its kind (a command; an edit with its value, a described value or the
        deletion; steps): a row read from it is the row that ran, and checks.render prints it the same.
        """
        from rkmon_lib import checks as k

        rows = [
            k.Check("GitHub", "PASS", "ok"),
            k.Check("gateway", "FAIL", "401", k.Command(("rk", "login", "gh"))),
            k.Check("budget", "WARN", "low", k.Edit("a.b", "config.json", 7)),
            k.Check(
                "model",
                "WARN",
                "gone",
                k.Edit(
                    "models.planner", "config.json", k.Described("a model it lists")
                ),
            ),
            k.Check("old key", "WARN", "set", k.Edit("a.c", "config.json", k.DELETE)),
            k.Check(
                "mount",
                "FAIL",
                "missing",
                k.Steps(
                    (
                        "mount it:",
                        k.Edit("paths.x", "config.json", "/srv/x"),
                        ", then",
                        reach.again(),
                        "checks again",
                    )
                ),
            ),
        ]
        now = [1000.0]
        ran, _ = reach.remote_checks(
            self.cfg,
            self.store.dir,
            checks={"github": lambda cfg: rows},
            clock=lambda: now[0],
        )
        now[0] += 60
        cached, said = reach.remote_checks(
            self.cfg,
            self.store.dir,
            checks={"github": lambda cfg: 1 / 0},
            clock=lambda: now[0],
        )
        self.assertEqual((ran, said[:25]), (rows, "service checks from 1 min"))
        self.assertEqual(cached, rows)
        self.assertEqual(k.render(cached), k.render(rows))


if __name__ == "__main__":
    unittest.main()

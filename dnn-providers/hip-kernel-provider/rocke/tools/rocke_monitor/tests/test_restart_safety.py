# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a stop, a restart or a busy machine must not cost: an interrupted review starts again (mirrored or not), the
verification already written comes back from its disk copy, two stagings never race on the shared clone, a network
blip is retried, a mirrored review stays blind and paired, a killed lock holder's children hold no lock, and the
status, board, scaler and tick filter read right."""

import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))

from rkmon_lib import (  # noqa: E402
    control,
    gitops,
    gh,
    mirror,
    post,
    procs,
    prtree,
    report,
    review,
    review_keep,
    scaler,
    tickfilter,
)
from rkmon_lib.store import Store, empty_state  # noqa: E402

import steps  # noqa: E402

H, H2 = "5" * 40, "6" * 40


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = pathlib.Path(self.d.name)
        self.store = Store(self.root / "state")
        self.feed = self.root / "live" / "mirror_feed.jsonl"
        self.feed.parent.mkdir()
        self.cfg = {
            "repo": "ROCm/r",
            "operator": {"github_login": "dev-op"},
            "rocke_path": "rk",
            "features": {"merge_queue": {"develop_guard": False}},
            "merge_authority": "propose",
            "review_first_line": "Automated review: Model - qwen, effort: max",
            "paths": {
                "working_dir": str(self.root),
                "root": str(self.root),
                "workers": str(self.root / "workers"),
            },
        }
        self.st = empty_state()
        self.st["prs"]["5"] = {
            "n": 5,
            "head": H,
            "status": "open",
            "draft": False,
            "ours": None,
        }

    def mirrored(self):
        return dict(
            self.cfg, mirror={"enabled": True, "feed": str(self.feed), "pairs": 5}
        )


class TestInterrupted(Base):
    """A stop records each pipe as interrupted; the first advances after rk start start it again at its head."""

    def pipe(self):
        review.pipes(self.st).append(
            {"pr": 5, "kind": "follow-up", "head": H, "again": True, "worker": None}
        )

    def test_a_stop_keeps_the_review_and_the_next_advance_starts_it_again(self):
        self.pipe()
        lines = control.cleanup(self.cfg, self.st, self.store)
        self.assertEqual(
            lines[0], "#5's review (follow-up) stopped: it starts again after rk start"
        )
        self.assertEqual(review.pipes(self.st), [])
        self.assertEqual(review.interrupted(self.st, 5)["head"], H)
        calls = []

        def start(cfg, st, store, n, kind, force=False, again=False):
            calls.append((n, kind, force, again))
            return [f"REVIEW_START #{n} ({kind}): worker started"]

        with mock.patch.object(review, "start_review", start), mock.patch.object(
            review.scaler, "room", lambda *a: (True, "")
        ):
            out, started = review.resume_interrupted(self.cfg, self.st, self.store, 3)
        self.assertEqual(calls, [(5, "follow-up", True, True)])
        self.assertEqual(started, {5})
        self.assertTrue(out[0].startswith("REVIEW_RESUME #5: the follow-up"))
        self.assertIsNone(review.interrupted(self.st, 5))

    def test_a_pr_that_moved_or_closed_meanwhile_is_not_started_again(self):
        self.pipe()
        control.cleanup(self.cfg, self.st, self.store)
        self.st["prs"]["5"]["head"] = H2
        with mock.patch.object(review, "start_review", lambda *a, **k: 1 / 0):
            out, started = review.resume_interrupted(self.cfg, self.st, self.store, 3)
        self.assertEqual(started, set())
        self.assertIn("not started again", out[0])
        self.assertIsNone(review.interrupted(self.st, 5))

    def test_a_start_that_waits_keeps_the_record_and_room_bounds_it(self):
        self.pipe()
        control.cleanup(self.cfg, self.st, self.store)
        with mock.patch.object(review.scaler, "room", lambda *a: (True, "")):
            with mock.patch.object(
                review, "start_review", lambda *a, **k: ["REVIEW_WAIT #5: not read"]
            ):
                review.resume_interrupted(self.cfg, self.st, self.store, 3)
            self.assertIsNotNone(review.interrupted(self.st, 5))
            out, _ = review.resume_interrupted(self.cfg, self.st, self.store, 0)
        self.assertEqual(out, [])  # no room: nothing tried
        self.assertTrue(review.forget_interrupted(self.st, 5))  # rk review-skip

    def test_a_mirrored_review_survives_the_stop_and_keeps_its_feed_line_and_arm(self):
        cfg = self.mirrored()
        key = mirror.key_of(5, H)
        mirror.state(self.st)["started"][key] = {
            "pr": 5,
            "head": H,
            "round": "fu2",
            "kind": "follow-up",
            "arm": "B",
            "again": True,
            "feed_at": ago(30),
            "at": ago(20),
        }
        self.pipe()
        control.cleanup(cfg, self.st, self.store)
        mirror.tidy(
            cfg, self.st, self.store
        )  # 20 minutes old, not in flight: still not dropped
        self.assertIn(key, mirror.state(self.st)["started"])
        self.assertEqual(mirror.state(self.st)["dropped"], {})
        seen = []

        def start(cfg, st, store, n, kind, force=False, again=False):
            seen.append((n, kind, force, again))
            review.pipes(st).append({"pr": n, "kind": kind, "head": H})
            return [f"REVIEW_START #{n}"]

        with mock.patch.object(
            gh, "rest", lambda *a, **k: {"state": "open", "head": H}
        ), mock.patch.object(review, "start_review", start), mock.patch.object(
            review.scaler, "room", lambda *a: (True, "")
        ):
            out, started = review.resume_interrupted(cfg, self.st, self.store, 3)
        self.assertEqual((started, seen), ({5}, [(5, "follow-up", True, True)]))
        s = mirror.state(self.st)["started"][key]
        self.assertEqual((s["arm"], s["round"]), ("B", "fu2"))
        # Without the record (the review ended), tidy drops it as before.
        mirror.state(self.st)["started"][key]["at"] = ago(10)
        review.drop_pipe(self.st, 5)
        mirror.tidy(cfg, self.st, self.store)
        self.assertIn(key, mirror.state(self.st)["dropped"])


class TestKeep(Base):
    """A review directory on the RAM disk is kept on the disk and comes back after a restart emptied it."""

    def setUp(self):
        super().setUp()
        self.rd = self.root / "rd"
        self.rd.mkdir()
        self.cfg["ramdisk"] = {"enabled": True, "path": str(self.rd), "size_gb": 4}
        self.st["review_ready"] = [{"pr": 5, "kind": "follow-up", "head": H}]

    def test_kept_restored_and_pruned(self):
        v = self.rd / "tmp" / "v5fu2"
        v.mkdir(parents=True)
        (v / "notes.md").write_text("premises: ...\n")
        (v / "pr5_fu2.md").write_text("the post\n")
        (v / "big.bin").write_bytes(b"0" * (review_keep.MAX_FILE_MB * 2**20 + 1))
        self.assertEqual(review_keep.sync(self.cfg, self.st, self.store), [])
        kept = self.root / review_keep.KEEP / "v5fu2"
        self.assertEqual(
            sorted(p.name for p in kept.iterdir()), ["notes.md", "pr5_fu2.md"]
        )
        import shutil

        shutil.rmtree(self.rd / "tmp")  # a restart emptied the RAM disk
        (self.rd / "tmp").mkdir()
        lines = review_keep.sync(self.cfg, self.st, self.store)
        self.assertTrue(lines[0].startswith("REVIEW_RESTORED #5"), lines)
        self.assertIn("rk spec-post 5 --round fu2", lines[0])
        self.assertEqual((v / "notes.md").read_text(), "premises: ...\n")
        self.assertEqual(review_keep.sync(self.cfg, self.st, self.store), [])  # once
        self.st["review_ready"] = []  # posted: its copy goes once quiet
        review_keep.sync(self.cfg, self.st, self.store)
        self.assertTrue(kept.is_dir())
        review_keep.sync(
            self.cfg,
            self.st,
            self.store,
            clock=lambda: time.time() + (review_keep.quiet_hours(self.cfg) + 1) * 3600,
        )
        self.assertFalse(kept.exists())

    def test_the_kept_copys_hours_are_a_setting(self):
        """retention.kept_reviews_h: how long a finished review's copy stays (24 h by default)."""
        self.assertEqual(review_keep.quiet_hours({}), 24)
        self.assertEqual(
            review_keep.quiet_hours({"retention": {"kept_reviews_h": "x"}}), 24
        )
        self.assertEqual(
            review_keep.quiet_hours({"retention": {"kept_reviews_h": -1}}), 24
        )
        kept = self.root / review_keep.KEEP / "v5fu2"
        kept.mkdir(parents=True)
        (kept / "notes.md").write_text("x\n")
        self.st["review_ready"] = []
        later = time.time() + 3 * 3600
        self.cfg["retention"] = {"kept_reviews_h": 6}
        review_keep.sync(self.cfg, self.st, self.store, clock=lambda: later)
        self.assertTrue(kept.is_dir())  # 3 h quiet: kept under a 6 h setting
        self.cfg["retention"] = {"kept_reviews_h": 2}
        review_keep.sync(self.cfg, self.st, self.store, clock=lambda: later)
        self.assertFalse(kept.exists())  # gone under a 2 h one

    def test_without_the_ramdisk_nothing_is_copied(self):
        del self.cfg["ramdisk"]
        self.assertEqual(review_keep.sync(self.cfg, self.st, self.store), [])
        self.assertFalse((self.root / review_keep.KEEP).exists())


class TestClone(Base):
    """Stagings and fetches take the clone's lock; a fetch that meets the shallow race runs once more; a staging that
    failed on the race is started again, never skipped (in a mirror its feed line is taken again).
    """

    def test_the_lock_is_one_file_for_the_stagings_and_the_toolkit(self):
        repo = self.root / "clone"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        self.assertEqual(steps.clone_lock_path(repo), gitops.clone_lock_path(str(repo)))
        with steps.clone_lock(repo) as held:
            self.assertTrue(held)
            with gitops.clone_lock(str(repo), wait_s=0.3) as other:
                self.assertFalse(other)  # waited, then went on without it
        with gitops.clone_lock(str(repo), wait_s=0.3) as other:
            self.assertTrue(other)
        with steps.clone_lock(self.root / "not-a-repo") as none:
            self.assertFalse(none)
        self.assertFalse((self.root / "not-a-repo").exists())

    def test_a_fetch_that_meets_the_shallow_race_runs_once_more(self):
        count = self.root / "n"
        fake = self.root / "fakegit"
        fake.write_text(
            "#!/bin/sh\nn=$(cat "
            f"{count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}\n"
            '[ "$n" = 1 ] && { echo "fatal: shallow file has changed since we read it" >&2; exit 128; }\n'
            "exit 0\n"
        )
        fake.chmod(0o755)
        p = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import steps; "
                "steps.must_fetch([sys.argv[2]], retry_s=0); print('ok')",
                str(ROOT / "worker"),
                str(fake),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual((p.returncode, p.stdout.split()[-1]), (0, "ok"), p.stderr)
        self.assertEqual(count.read_text().strip(), "2")

    def test_a_staging_lost_to_the_race_starts_again_not_skipped(self):
        cfg = self.mirrored()
        key = mirror.key_of(5, H)
        mirror.state(self.st)["started"][key] = {"pr": 5, "head": H, "at": ago(1)}
        w = self.root / "workers" / "pr5"
        w.mkdir(parents=True)
        (w / "mkwt.log").write_text(
            "fatal: shallow file has changed since we read it\nmkwt failed\n"
        )
        job = {"id": "worker-pr5-1", "pid": 2**22 + 1, "started": ago(9)}
        review.pipes(self.st).append(
            {
                "pr": 5,
                "kind": "first-round",
                "head": H,
                "worker": job,
                "checks": None,
                "started": ago(9),
            }
        )
        with mock.patch.object(
            review.jobs, "status", lambda s, j: "done" if j else "none"
        ), mock.patch.object(
            review.config, "reviewer_is_local", lambda cfg: False
        ), mock.patch.object(
            review.jobs, "heavy_busy", lambda *a: None
        ), mock.patch.object(
            review.scaler, "room", lambda *a: (True, "")
        ):
            lines = review._advance_pipe(
                cfg, self.st, self.store, review.pipes(self.st)[0]
            )
        self.assertIn("starts again, not skipped", lines[-1])
        self.assertEqual(
            tickfilter.classify(lines[-1])[0], tickfilter.ROUTINE
        )  # the tick filter does not send it to the orchestrator as a skipped PR
        self.assertNotIn(5, self.st.get("review_skip") or [])
        self.assertEqual(review.pipes(self.st), [])
        self.assertNotIn(key, mirror.state(self.st)["started"])
        self.assertNotIn(key, mirror.state(self.st)["dropped"])


class TestMirrorBlind(Base):
    """Our last post is pinned to a PR's feed line from the moment the line is in the feed; a review at a head no feed
    line started is refused, and a refused post leaves nothing in the mirror folder."""

    def feed_line(self, at):
        with open(self.feed, "a") as f:
            f.write(
                json.dumps(
                    {"pr": 5, "head": H, "round": "fu2", "kind": "follow-up", "at": at}
                )
                + "\n"
            )

    def test_a_waiting_feed_line_already_pins_our_last_post(self):
        cfg = self.mirrored()
        mirror.state(self.st)["since"] = ago(60)
        at = ago(10)
        self.feed_line(at)
        self.assertEqual(mirror.pinned_at(cfg, self.st, 5), at)
        self.assertTrue(
            mirror.hold_ours(cfg, self.st, 5, {"at": ago(2)})
        )  # live's later post
        self.assertFalse(
            mirror.hold_ours(cfg, self.st, 5, {"at": ago(20)})
        )  # the post it follows up on
        self.assertFalse(mirror.hold_ours(cfg, self.st, 6, {"at": ago(2)}))
        mirror.state(self.st)["done"][mirror.key_of(5, H)] = {"pr": 5}
        self.assertFalse(
            mirror.hold_ours(cfg, self.st, 5, {"at": ago(2)})
        )  # written: free again
        self.assertFalse(
            mirror.hold_ours(self.cfg, self.st, 5, {"at": ago(2)})
        )  # mirror off

    def test_no_review_is_written_at_a_head_no_feed_line_started(self):
        cfg = self.mirrored()
        with self.assertRaisesRegex(
            post.PostRefused, "no feed line started a review of #5"
        ):
            mirror.divert(cfg, self.st, self.store, 5, {"text": "x\n", "sha": H}, None)
        self.assertFalse(mirror.out_dir(cfg).exists())

    def test_a_refused_post_leaves_nothing_behind(self):
        cfg = self.mirrored()
        mirror.state(self.st)["started"][mirror.key_of(5, H)] = {
            "pr": 5,
            "head": H,
            "round": "fu2",
        }

        def refuse(*a, **k):
            raise post.PostRefused("line 1 must be ...")

        with mock.patch.object(post, "preflight", refuse), self.assertRaises(
            post.PostRefused
        ):
            mirror.divert(cfg, self.st, self.store, 5, {"text": "x\n", "sha": H}, None)
        self.assertFalse((mirror.out_dir(cfg) / f"pr5_fu2_{H[:10]}").exists())


class TestLocksAcrossAKill(unittest.TestCase):
    """A lock holder killed while its children run: no child holds the state lock or the write lock."""

    def test_the_children_of_a_killed_holder_hold_no_lock(self):
        with tempfile.TemporaryDirectory() as d:
            child = (
                "import os, subprocess, sys, time\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from rkmon_lib.store import Store\n"
                "from rkmon_lib import control, jobs\n"
                "s = Store(__import__('pathlib').Path(sys.argv[1]))\n"
                "with control.write_lock(s.dir), s.locked():\n"
                "    a = subprocess.Popen(['sleep', '60'])\n"
                "    b = jobs.start(s, 'checks', 'pr5', 'sleep 61', timeout_min=5)\n"
                "    print(a.pid, b['pid'], flush=True)\n"
                "    time.sleep(60)\n"
            )
            p = subprocess.Popen(
                [sys.executable, "-c", child, d],
                stdout=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            a, b = map(int, p.stdout.readline().split())
            try:
                os.kill(p.pid, signal.SIGKILL)
                p.wait(10)
                for lock in ("lock", "write.lock"):
                    free = subprocess.run(["flock", "-n", f"{d}/{lock}", "true"])
                    self.assertEqual(free.returncode, 0, f"{lock} is still held")
                for pid in (a,):
                    fds = [
                        os.readlink(f"/proc/{pid}/fd/{x}")
                        for x in os.listdir(f"/proc/{pid}/fd")
                    ]
                    self.assertFalse([f for f in fds if f.endswith("lock")], fds)
            finally:
                p.stdout.close()
                for pid in (a, b):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass


class TestNetwork(Base):
    """A check section that meets a network failure runs once more; the archive prefetches its blobs at once."""

    def net(self, script: str, env: dict | None = None):
        return subprocess.run(
            ["bash", "-c", review.NET_FN + "\n" + script],
            capture_output=True,
            text=True,
            env={**os.environ, "RKMON_NET_RETRY_S": "0", **(env or {})},
        )

    def test_a_network_failure_runs_once_more(self):
        n = self.root / "n"
        p = self.net(
            f'f() {{ c=$(cat {n} 2>/dev/null || echo 0); echo $((c+1)) > {n}; [ "$c" = 0 ] && '
            '{ echo "NOT RUN: Could not resolve host: github.com"; return 1; }; echo "CI at head: 3 success"; }\n'
            "net f"
        )
        self.assertEqual(p.stdout.strip(), "CI at head: 3 success")
        self.assertIn("a network error", p.stderr)
        self.assertEqual(n.read_text().strip(), "2")
        p = self.net(
            f'g() {{ echo "FAIL: a real failure"; return 1; }}\nnet g; echo rc=$?'
        )
        self.assertEqual(p.stdout.split(), ["FAIL:", "a", "real", "failure", "rc=1"])

    def test_a_run_that_succeeds_runs_once_whatever_it_says(self):
        """A run that succeeds (its exit status) is never run again, though its errors name a network failure: sudo
        warns on each command when it cannot resolve the host's own name."""
        n = self.root / "n"
        p = self.net(
            f"f() {{ c=$(cat {n} 2>/dev/null || echo 0); echo $((c+1)) > {n}; "
            'echo "sudo: unable to resolve host <host>: Temporary failure in name resolution" >&2; '
            'echo "CI at head: 3 success"; }\nnet f; echo rc=$?'
        )
        self.assertEqual(p.stdout.splitlines(), ["CI at head: 3 success", "rc=0"])
        self.assertIn("unable to resolve host", p.stderr)
        self.assertNotIn("a network error", p.stderr)
        self.assertEqual(n.read_text().strip(), "1")

    def test_the_review_checks_wrap_their_network_sections(self):
        lines = review.status_lines("python -I x.py", "r.txt", "TAG", "k", 5, "scan")
        self.assertTrue(lines[0].startswith("net python -I x.py"))

    def test_the_archive_fetches_its_missing_blobs_in_one_request(self):
        src, clone = self.root / "src", self.root / "clone"
        run = lambda *a, cwd=None: subprocess.run(  # noqa: E731
            ["git", *a], cwd=cwd, check=True, capture_output=True, text=True
        )
        run("init", "-q", "-b", "main", str(src))
        (src / "rk").mkdir()
        for i in range(3):
            (src / "rk" / f"f{i}.py").write_text(f"x = {i}\n")
        run("-C", str(src), "add", ".")
        run(
            "-C",
            str(src),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "-m",
            "c",
        )
        run("-C", str(src), "config", "uploadpack.allowFilter", "true")
        run("-C", str(src), "config", "uploadpack.allowAnySHA1InWant", "true")
        run(
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            f"file://{src}",
            str(clone),
        )
        rev = run("-C", str(clone), "rev-parse", "origin/main").stdout.strip()
        self.assertEqual(len(prtree.missing_blobs(str(clone), rev, ["rk"])), 3)
        self.assertEqual(prtree.prefetch(str(clone), rev, ["rk"]), 3)
        self.assertEqual(prtree.missing_blobs(str(clone), rev, ["rk"]), [])
        prtree.extract(str(clone), rev, ["rk"], self.root / "out")
        self.assertEqual((self.root / "out" / "rk" / "f2.py").read_text(), "x = 2\n")


class TestReadings(Base):
    def test_the_review_driver_under_the_orchestrator_reads_as_running(self):
        """A registered process started by another (the driver under the orchestrator) is listed as itself."""
        run = self.root / "run"
        (run / "procs").mkdir(parents=True)
        uid = os.getuid()
        table = {
            100: {
                "pid": 100,
                "ppid": 1,
                "pgid": 100,
                "sid": 100,
                "start": 7,
                "state": "S",
                "comm": "o",
                "uid": uid,
            },
            200: {
                "pid": 200,
                "ppid": 100,
                "pgid": 100,
                "sid": 100,
                "start": 8,
                "state": "S",
                "comm": "d",
                "uid": uid,
            },
            300: {
                "pid": 300,
                "ppid": 200,
                "pgid": 100,
                "sid": 100,
                "start": 9,
                "state": "S",
                "comm": "a",
                "uid": uid,
            },
        }
        for pid, kind, start in ((100, "orchestrator", 7), (200, "driver", 8)):
            (run / "procs" / f"{pid}.json").write_text(
                json.dumps({"pid": pid, "start": start, "kind": kind, "what": ""})
            )
        got = procs.ours(run, self.store.dir, table, env=lambda pid: {}, me=99999)
        kinds = {e["pid"]: (e["kind"], e["what"]) for e in got}
        self.assertEqual(kinds[200], ("driver", ""))
        self.assertEqual(kinds[300], ("driver", "child of 200"))

    def test_the_board_says_develop_issues_were_not_read_without_the_develop_guard(
        self,
    ):
        self.assertEqual(
            report.issues_read(self.cfg, self.st, []),
            "not read (features.merge_queue.develop_guard is off)",
        )
        self.assertEqual(
            report.issues_read(
                dict(self.cfg, features={"merge_queue": {"develop_guard": True}}),
                self.st,
                [],
            ),
            "not read yet",
        )
        self.st["post_merge"] = {"open": [], "at": ago(1)}
        self.assertEqual(report.issues_read(self.cfg, self.st, []), "none")

    def test_the_cpu_reads_at_most_full(self):
        """A load past the cores this process may use reads 100%, never 140%."""
        mem = self.root / "meminfo"
        mem.write_text("MemTotal: 1000 kB\nMemAvailable: 500 kB\n")
        with mock.patch.object(scaler, "cpu_count", lambda: 32):
            h = scaler.host_load(
                {}, str(mem), str(self.root / "nocg"), loadavg=lambda: (45.0, 0, 0)
            )
        self.assertEqual(h["cpu_pct"], 100.0)

    def test_the_second_line_of_a_gh_network_error_is_not_an_act_line(self):
        text = "error connecting to api.github.com\ncheck your internet connection or the GitHub status page\n"
        self.assertEqual(
            [ln for ln, _ in tickfilter.needed(text, 0)],
            ["error connecting to api.github.com"],
        )


if __name__ == "__main__":
    unittest.main()

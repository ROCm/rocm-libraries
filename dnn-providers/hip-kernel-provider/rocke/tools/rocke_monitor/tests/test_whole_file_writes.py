# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The files the monitor keeps for its runs, its caches and its CI tools, each written whole (rkmon_lib/whole_file.py):
the state, the incidents and the copy of those they replace, the overlaps' cache and decisions, a process's record,
the review requests' cache, a develop's clash cache, a run's lane listing (rkmon_lib/lanes.py and tools/lane_of.py),
the develop outcomes' cache (tools/installed_tests.py) and a CI job's log (tools/ci_sha.py). A link planted beside the
file, at a fixed temporary name or one with this process's id, is never followed and never stops the write: the file
is a regular one at its kind's mode whatever the umask (worker/perms.py: the shared files', or a tree's), and the file
the link points at is as it was. Temporary folders only; gh is faked."""

import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

import ci_sha  # noqa: E402
import installed_tests  # noqa: E402
import lane_of  # noqa: E402
from rkmon_lib import clashes, gh, guard, lanes, overlaps, paths  # noqa: E402
from rkmon_lib import procs, snapshot, store  # noqa: E402

PERMS = paths.worker_module("perms")
ELSEWHERE = "the file the link points at\n"


class Case(unittest.TestCase):
    """A temporary folder, the umask 077 (so a mode the write does not set shows), and the file a planted link points
    at."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.elsewhere = self.dir / "elsewhere"
        self.elsewhere.write_text(ELSEWHERE, encoding="utf-8")
        umask = os.umask(0o077)
        self.addCleanup(os.umask, umask)

    def plant(self, path) -> None:
        """A link at PATH to the file elsewhere."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(self.elsewhere)

    def assertWhole(self, path, mode: int) -> None:
        """PATH is a regular file at MODE, and the file the link points at is as it was."""
        path = pathlib.Path(path)
        self.assertFalse(path.is_symlink(), path)
        self.assertTrue(path.is_file(), path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode, path)
        self.assertEqual(self.elsewhere.read_text(encoding="utf-8"), ELSEWHERE)


class TestStateFolder(Case):
    """The files in the state folder: the monitor and the writer share them."""

    def setUp(self):
        super().setUp()
        self.store = store.Store(self.dir / "state")

    def test_the_state(self):
        self.plant(self.store.dir / "state.json.tmp")
        self.store.save(dict(store.empty_state(), x=1))
        self.assertWhole(self.store.state_path, PERMS.SHARED_FILE_MODE)
        self.assertEqual(self.store.load()["x"], 1)

    def test_the_incidents_and_the_copy_of_those_they_replace(self):
        incidents = self.store.dir / guard.INCIDENTS_FILE
        self.plant(self.store.dir / (guard.INCIDENTS_FILE + ".tmp"))
        guard.save_incidents(self.store, [{"pr": 1}])
        self.assertWhole(incidents, PERMS.SHARED_FILE_MODE)
        prev = self.store.dir / (guard.INCIDENTS_FILE + ".prev")
        self.plant(prev)
        guard.save_incidents(self.store, [{"pr": 2}])
        self.assertWhole(prev, PERMS.SHARED_FILE_MODE)
        self.assertEqual(json.loads(prev.read_text(encoding="utf-8")), [{"pr": 1}])
        self.assertEqual(guard.incidents(self.store), [{"pr": 2}])

    def test_the_overlaps_cache(self):
        self.plant(self.store.dir / (overlaps.CACHE_FILE + ".tmp"))
        cache = overlaps.load_cache(self.store)
        cache["named"]["k"] = 1
        overlaps.save_cache(self.store, cache)
        self.assertWhole(self.store.dir / overlaps.CACHE_FILE, PERMS.SHARED_FILE_MODE)
        self.assertEqual(overlaps.load_cache(self.store), cache)

    def test_the_overlap_decisions(self):
        self.plant(self.store.dir / "overlaps_decided.json.tmp")
        overlaps.decide({}, self.store, "k", 1, [1, 2], "", "tester")
        self.assertWhole(self.store.dir / overlaps.DECIDED_FILE, PERMS.SHARED_FILE_MODE)
        self.assertEqual(overlaps.decisions(self.store)["k"]["owner"], 1)


class TestRecords(Case):
    """A process's record: a file the monitor shares."""

    def test_a_process_record(self):
        run = self.dir / "run"
        self.plant(run / "procs" / f".{os.getpid()}.json.new")
        f = procs.register(run, "tick")
        self.assertIsNotNone(f)
        self.assertWhole(f, PERMS.SHARED_FILE_MODE)
        self.assertEqual(json.loads(f.read_text(encoding="utf-8"))["kind"], "tick")


class TestCaches(Case):
    """The review requests' cache and a develop's clash cache: files the monitor shares."""

    def test_the_review_requests_cache(self):
        cfg = {"repo": "o/r", "paths": {"cache": str(self.dir / "cache")}}
        cache = self.dir / "cache" / snapshot.REVIEW_REQUESTS_CACHE
        self.plant(cache.with_name(f"{cache.name}.{os.getpid()}.tmp"))
        with mock.patch.object(gh, "rest", return_value=["dev-b"]):
            self.assertEqual(
                snapshot.rest_review_requests(cfg, 7, "2026-11-11T00:00:00Z"),
                ["dev-b"],
            )
        self.assertWhole(cache, PERMS.SHARED_FILE_MODE)
        self.assertEqual(json.loads(cache.read_text())["7"]["reqs"], ["dev-b"])

    def test_a_develop_clash_cache(self):
        folder = str(self.dir / "clash")
        dc = clashes.DevCache(folder, "a" * 40)
        self.plant(dc.path.with_suffix(f".{os.getpid()}.tmp"))
        dc.put("names", "k", 1)
        dc.save()
        self.assertWhole(dc.path, PERMS.SHARED_FILE_MODE)
        self.assertEqual(clashes.DevCache(folder, "a" * 40).get("names", "k"), 1)


class TestTrees(Case):
    """A run's lane listing and the develop outcomes' cache: a tree's files, which the runners read."""

    SHA = "ab" * 20

    class Slot:
        """A slot's `out`, read as JSON."""

        def __init__(self, out: pathlib.Path):
            self.out = str(out)

        def read_json(self, rel: str):
            return json.loads((pathlib.Path(self.out) / rel).read_text())

    def test_a_listing_the_monitor_keeps(self):
        out = self.dir / "out"
        out.mkdir()
        (out / f"{self.SHA}.json").write_text(json.dumps({"version": 1}))
        cache = self.dir / "cache"
        self.plant(cache / f".{self.SHA}.json.{os.getpid()}.tmp")
        self.assertEqual(lanes.keep_listing(str(cache), self.SHA, self.Slot(out)), "")
        self.assertWhole(cache / f"{self.SHA}.json", PERMS.TREE_FILE_MODE)

    def test_a_listing_lane_of_keeps(self):
        cache = self.dir / "cache"
        self.plant(cache / f".{self.SHA}.json.tmp")
        lane_of.keep_listing(cache, self.SHA, {"version": 1})
        kept = cache / f"{self.SHA}.json"
        self.assertWhole(kept, PERMS.TREE_FILE_MODE)
        self.assertEqual(json.loads(kept.read_text()), {"version": 1})

    def test_the_develop_outcomes_cache(self):
        cache = self.dir / "installed_develop_abc.tsv"
        src = self.dir / installed_tests.OUTCOMES_FILE
        src.write_text("e\tt.py::a\tpassed\n")
        self.plant(f"{cache}.{os.getpid()}.new")
        self.assertEqual(installed_tests.cache_add(str(src), str(cache)), 1)
        self.assertWhole(cache, PERMS.TREE_FILE_MODE)
        self.assertEqual(
            installed_tests.load_cache(str(cache)), {("e", "t.py::a"): "passed"}
        )


class TestCiLog(Case):
    """A CI job's log ci_sha.py keeps in the scratch folder: a file the monitor and the writer share."""

    LOG = b"FAILED tests/test_a.py::test_x\nok\n"

    def test_a_job_log(self):
        logs = self.dir / "ci_sha"
        self.plant(logs / "job_101.log.tmp")
        cfg = {
            "repo": "o/r",
            "develop_watched_workflows": ["Build and test"],
            "develop_rocke_job_pattern": "unit",
            "paths": {"scratch": str(self.dir)},
        }

        def ghget(*args):
            if "/actions/runs?" in args[0]:
                return b"11\tBuild and test\tcompleted\tfailure\t2026-11-01T00:00:00Z"
            if "/jobs?" in args[0]:
                return b"101\tLinux / unit tests (shard 1)"
            return b"a job [failure, attempt 1]"

        prints = [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.stdout.buffer.write({self.LOG!r})",
        ]
        said = []
        with mock.patch.object(
            ci_sha.store, "load_config", return_value=cfg
        ), mock.patch.object(ci_sha, "ghget", ghget), mock.patch.object(
            ci_sha, "gh", return_value=prints
        ), mock.patch.object(
            ci_sha, "say", said.append
        ):
            self.assertEqual(ci_sha.main(["ci_sha.py", "a" * 40, "FAILED "]), 0)
        self.assertWhole(logs / "job_101.log", PERMS.SHARED_FILE_MODE)
        self.assertEqual((logs / "job_101.log").read_bytes(), self.LOG)
        self.assertIn(b"     FAILED tests/test_a.py::test_x", said)


if __name__ == "__main__":
    unittest.main()

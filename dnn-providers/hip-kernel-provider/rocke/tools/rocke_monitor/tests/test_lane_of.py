# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/lane_of.py (`rk lane-of`): where a test runs in CI, from the tree. The test file's own facts (marks,
module-level importorskips, imports and skips), the `-m` expression, the verdict over a tree's entries and the CI lanes,
the CI-condition collect, the shard log's lines, and the tool end to end on a small provider tree configured with cmake
(an entry per tier, an integration-marked module, a torch-guarded module, a laneless file). Temp git repos, no
network."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import lane_of as lo  # noqa: E402
from rkmon_lib import lanes  # noqa: E402
from test_ctest_labels import FAKE_CATEGORIES, cmake_dir, g  # noqa: E402

K = "prov/rocke"
LANES = [
    {
        "jobs": "PR shards",
        "tree": "installed",
        "tier": "standard",
        "exclude": ["ex_gpu"],
    }
]
STD = ["ckc", "python", "quick", "standard", "comprehensive", "full"]


def pytest_entry(*args, labels=STD) -> dict:
    return {
        "labels": sorted(labels),
        "cmd": ["python", "-m", "pytest", *args],
        "cwd": "",
    }


LISTING = {
    "build": {
        "rocke_parity": {
            "labels": ["ckc", "python"],
            "cmd": ["python", "-m", "pytest", "platform/tests/core/test_parity.py"],
            "cwd": "platform/tests",
        }
    },
    "installed": {
        "rocke_pytest_core": pytest_entry(
            "./tests/core", "-m", "not integration", "-v"
        ),
        "rocke_vis": pytest_entry(
            "./tests/vis", labels=["ckc", "python", "comprehensive", "full"]
        ),
        "rocke_gpu": pytest_entry("./tests/gpu", labels=STD + ["ex_gpu"]),
        "rocke_untiered": pytest_entry("./tests/bare", labels=["ckc", "python"]),
    },
    "suites": None,
}
ABSENT = {
    "torch",
    "matplotlib",
    "rocke_engine",
    "rocke_library_engine",
}  # the modules this CI lacks


def verdict(
    target: str, source: str | None = "def test_x():\n    pass\n", lanes=LANES
) -> dict:
    return lo.verdict(lo.Target(target, K), LISTING, lanes, ABSENT, source)


class TestTarget(unittest.TestCase):
    def test_a_file_a_node_id_and_an_entry(self):
        t = lo.Target(f"{K}/platform/tests/core/test_a.py", K)
        self.assertEqual(
            (t.kind, t.file, t.test), ("file", "platform/tests/core/test_a.py", "")
        )
        t = lo.Target("platform/tests/core/test_a.py::TestX::test_y[1]", K)
        self.assertEqual(
            (t.kind, t.file, t.test),
            ("node", "platform/tests/core/test_a.py", "TestX::test_y[1]"),
        )
        t = lo.Target("rocke_pytest_core", K)
        self.assertEqual((t.kind, t.entry), ("entry", "rocke_pytest_core"))


class TestFileFacts(unittest.TestCase):
    def test_module_marks_skips_and_imports(self):
        src = textwrap.dedent(
            """\
            import os
            import pytest
            from rocke_engine import x
            torch = pytest.importorskip("torch")
            pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.name == "nt", reason="posix")]
            try:
                import matplotlib
            except ImportError:
                matplotlib = None
            """
        )
        f = lo.file_facts(src)
        self.assertEqual(f["importorskip"], ["torch"])
        self.assertEqual([m for m, _ in f["marks"]], ["integration", "skipif"])
        self.assertIn("rocke_engine", f["imports"])
        self.assertNotIn("matplotlib", f["imports"])  # guarded: no collection error
        self.assertFalse(f["skip"])
        self.assertTrue(
            lo.file_facts(
                'import pytest\npytest.skip("no", allow_module_level=True)\n'
            )["skip"]
        )

    def test_a_node_reads_its_class_and_function(self):
        src = textwrap.dedent(
            """\
            import pytest
            class TestA:
                pytestmark = pytest.mark.slow
                @pytest.mark.skip(reason="broken")
                def test_b(self):
                    pytest.importorskip("rocke_library_engine")
            """
        )
        f = lo.file_facts(src, "TestA::test_b[p0]")
        self.assertTrue(f["test_found"])
        self.assertEqual(sorted(m for m, _ in f["test_marks"]), ["skip", "slow"])
        self.assertEqual(f["test_importorskip"], ["rocke_library_engine"])
        self.assertFalse(lo.file_facts(src, "TestA::test_gone")["test_found"])
        self.assertIn("error", lo.file_facts("def (:\n"))


class TestMarkers(unittest.TestCase):
    def test_the_entrys_expression_and_its_selection(self):
        self.assertEqual(
            lo.marker_expr(["py", "-m", "pytest", "./t", "-m", "not integration"]),
            "not integration",
        )
        self.assertEqual(lo.marker_expr(["py", "-m", "pytest", "./t"]), "")
        self.assertTrue(lo.selected("not integration", set()))
        self.assertFalse(lo.selected("not integration", {"integration"}))
        self.assertTrue(lo.selected("gpu or (slow and not integration)", {"slow"}))
        self.assertTrue(lo.selected("", {"integration"}))
        self.assertIsNone(lo.selected("not integration; rm", set()))
        self.assertIsNone(lo.selected("__import__('os')", set()))


class TestVerdict(unittest.TestCase):
    def test_a_test_a_lane_selects_runs_with_its_entry_and_tier(self):
        v = verdict("platform/tests/core/test_a.py")
        self.assertEqual(
            (v["verdict"], v["entry"], v["tier"]),
            ("runs", "rocke_pytest_core", "standard"),
        )
        self.assertEqual(
            lo.line(v),
            "platform/tests/core/test_a.py: CI: runs (rocke_pytest_core, standard)",
        )
        self.assertEqual(verdict("rocke_pytest_core")["verdict"], "runs")

    def test_comprehensive_only_runs_nowhere(self):
        v = verdict("platform/tests/vis/test_v.py")
        self.assertEqual(v["verdict"], "no lane")
        self.assertIn("comprehensive full only", v["why"])
        self.assertIn("no CI job selects those tiers", lo.line(v))

    def test_the_build_tree_an_excluded_label_no_tier_and_no_entry(self):
        v = verdict("platform/tests/core/test_parity.py")
        self.assertEqual(
            v["verdict"], "runs"
        )  # the installed entry runs it; the build-tree one alone would not
        self.assertEqual(
            verdict("rocke_parity")["why"],
            "rocke_parity is a build-tree entry; CI runs the installed tree",
        )
        self.assertIn(
            "ex_gpu, which the lanes leave out",
            verdict("platform/tests/gpu/test_g.py")["why"],
        )
        self.assertIn("no tier label", verdict("platform/tests/bare/test_b.py")["why"])
        v = verdict("library/tests/test_lonely.py")
        self.assertEqual(v["verdict"], "no lane")
        self.assertIn("in no CTest entry", v["why"])
        self.assertIn("no CTest entry has that name", verdict("rocke_nothing")["why"])

    def test_without_lanes_a_collected_test_is_not_run(self):
        # No PR-CI lane configured: a test an entry collects has no verdict (exit 3); one in no entry runs nowhere.
        v = verdict("platform/tests/core/test_a.py", lanes=[])
        self.assertEqual((v["verdict"], v["why"]), ("not run", lo.NO_LANE_KNOWN))
        self.assertEqual(
            lo.line(v), f"platform/tests/core/test_a.py: NOT RUN: {lo.NO_LANE_KNOWN}"
        )
        self.assertEqual(lo.code({"error": None, "results": [v]}), 3)
        self.assertEqual(
            verdict("library/tests/test_lonely.py", lanes=[])["verdict"], "no lane"
        )

    def test_markers_and_module_level_skips_in_a_lane(self):
        v = verdict(
            "platform/tests/core/test_i.py",
            "import pytest\npytestmark = pytest.mark.integration\ndef test_x(): pass\n",
        )
        self.assertEqual(v["verdict"], "no lane")
        self.assertIn(
            'deselected by rocke_pytest_core\'s -m "not integration"', v["why"]
        )
        v = verdict(
            "platform/tests/core/test_t.py",
            'import pytest\ntorch = pytest.importorskip("torch")\n',
        )
        self.assertEqual(
            (v["verdict"], v["why"]),
            ("skips", 'module-level importorskip("torch"); CI has no torch'),
        )
        v = verdict(
            "platform/tests/core/test_t.py", "import torch.nn\ndef test_x(): pass\n"
        )
        self.assertEqual(v["verdict"], "errors")
        self.assertIn("CI has no torch", v["why"])
        v = verdict(
            "platform/tests/core/test_t.py::test_x",
            'import pytest\ndef test_x():\n    pytest.importorskip("rocke_engine")\n',
        )
        self.assertEqual(v["verdict"], "skips")
        # A module CI has (numpy) guards nothing.
        self.assertEqual(
            verdict(
                "platform/tests/core/test_n.py",
                'import pytest\npytest.importorskip("numpy")\n',
            )["verdict"],
            "runs",
        )
        v = verdict(
            "platform/tests/core/test_s.py",
            'import pytest, os\npytestmark = pytest.mark.skipif(os.name == "nt", reason="x")\n',
        )
        self.assertEqual(v["verdict"], "runs")
        self.assertTrue(any("decided at run time" in d for d in v["detail"]))


class TestCollectAndLog(unittest.TestCase):
    def test_the_collect_overrides_a_runs(self):
        base = verdict("platform/tests/core/test_a.py")
        got = {
            "rc": 5,
            "run": [],
            "skipped": [("core/test_a.py", "could not import 'torch'")],
            "errors": [],
        }
        v = lo.apply_collect(dict(base, detail=[]), got)
        self.assertEqual(v["verdict"], "skips")
        self.assertIn("could not import 'torch'", v["why"])
        got = {"rc": 2, "run": [], "skipped": [], "errors": ["ERROR core/test_a.py"]}
        self.assertEqual(
            lo.apply_collect(dict(base, detail=[]), got)["verdict"], "errors"
        )
        got = {"rc": 0, "run": ["core/test_a.py::test_x"], "skipped": [], "errors": []}
        v = lo.apply_collect(dict(base, detail=[]), got)
        self.assertEqual(v["verdict"], "runs")
        self.assertIn("1 would run, 0 skip", v["detail"][0])
        v = lo.apply_collect(dict(base, detail=[]), {"error": "timed out"})
        self.assertEqual(
            (v["verdict"], v["detail"]), ("runs", ["collect NOT RUN: timed out"])
        )

    def test_the_shard_logs_lines_for_the_entry(self):
        log = (
            "2026-01-01T00:00:00Z  Start 12: rocke_pytest_core\n"
            "12/90 Test #12: rocke_pytest_core ................   Passed  41.20 sec\n"
            "13/90 Test #13: rocke_pytest_core_extra ..........   Passed   1.00 sec\n"
            "a line that names rocke_pytest_core in passing\n"
        )
        got = lo.log_lines(log, "rocke_pytest_core")
        self.assertEqual(len(got), 2)
        self.assertIn("Test #12", got[1])


class TestWiring(unittest.TestCase):
    def test_rk_passes_the_ci_facts_the_cache_and_the_rev(self):
        cfg = {
            "rocke_path": K,
            "paths": {"working_dir": "/w"},
            "ci_test_tiers": LANES,
            "ci_absent_modules": ["torch", "mpmath"],
        }
        argv = lanes.lane_of_argv(
            cfg, ["a.py"], rev="abc", collect=True, log="/l", as_json=True
        )
        self.assertEqual(argv[-2:], ["--", "a.py"])
        self.assertEqual(argv[argv.index("--absent") + 1], "torch,mpmath")
        self.assertEqual(json.loads(argv[argv.index("--ci-tiers") + 1]), LANES)
        self.assertEqual(argv[argv.index("--rev") + 1], "abc")
        self.assertEqual(argv[argv.index("--cache") + 1], "/w/cache/lane_of")
        self.assertIn("--collect", argv)
        self.assertEqual(lanes.absent_modules({}), [])  # none configured: none absent
        env = lanes.lane_of_env(cfg)
        self.assertTrue(env["PATH"].startswith("/w/venv/bin:"))
        self.assertNotIn("GH_TOKEN", env)

    def test_the_rk_command_runs_the_tool_at_the_rev_or_the_prs_head(self):
        import rkmon
        from rkmon_lib import config
        from rkmon_lib.store import Store, empty_state

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = pathlib.Path(tmp.name)
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path=K,
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        c["paths"] = dict(c["paths"], working_dir=str(base / "work"))
        cfg = config.normalize(c)
        store = Store(base / "state")
        st = empty_state()
        st["prs"]["7"] = {"n": 7, "head": "f" * 40}
        store.save(st)
        with mock.patch.object(rkmon, "load_config", lambda: cfg), mock.patch.object(
            rkmon, "Store", lambda: store
        ), mock.patch.object(rkmon.subprocess, "run") as run:
            # Each command fails, as a real one would (its output, empty, captured).
            run.side_effect = lambda argv, **kw: subprocess.CompletedProcess(
                argv, 1, "", ""
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(rkmon.main(["lane-of", "a.py", "--rev", "abc"]), 1)
            argv = run.call_args.args[0]
            self.assertIn("lane_of.py", " ".join(argv))
            # its --fetch first, as the monitor: the targets after it, before the run in the slot
            calls = [c.args[0] for c in run.call_args_list]
            fetch = [a for a in calls if "--fetch" in a]
            self.assertEqual(len(fetch), 1)
            self.assertLess(calls.index(fetch[0]), len(calls) - 1)
            self.assertEqual(
                fetch[0][fetch[0].index("--fetch") :], ["--fetch", "--", "a.py"]
            )
            # the clone, the one tree its git reads
            self.assertEqual(
                run.call_args.kwargs["env"]["RKMON_TREES"],
                argv[argv.index("--repo") + 1],
            )
            self.assertEqual((argv[argv.index("--rev") + 1], argv[-1]), ("abc", "a.py"))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(rkmon.main(["lane-of", "a.py", "--pr", "7"]), 1)
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index("--rev") + 1], "f" * 40)
            with mock.patch("builtins.print"):
                self.assertEqual(rkmon.main(["lane-of", "a.py", "--pr", "8"]), 4)
                self.assertEqual(
                    rkmon.main(["lane-of", "a.py", "--pr", "7", "--rev", "x"]), 4
                )


def _venv_python() -> str | None:
    """A python with pytest: $RKMON_VENV's, else this one when it has pytest."""
    v = os.environ.get("RKMON_VENV")
    if v and pathlib.Path(v, "bin", "python").exists():
        return str(pathlib.Path(v, "bin", "python"))
    try:
        import pytest  # noqa: F401

        return sys.executable
    except ImportError:
        return None


class TestKeepListing(unittest.TestCase):
    """lanes.keep_listing: the listing a run made for its commit, read as JSON by the result reader, kept in the cache
    as a new file the runners read; anything else is not kept, with why, never a traceback.
    """

    SHA = "ab" * 20

    class Slot:
        """A slot's `out`, read as prcode.Slot reads it."""

        def __init__(self, out: str):
            self.out = out

        def read_json(self, rel: str):
            sys.path.insert(0, str(ROOT / "worker"))
            import pr_results

            return pr_results.read_json(self.out, rel)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = pathlib.Path(tmp.name) / "out"
        self.out.mkdir()
        self.cache = pathlib.Path(tmp.name) / "cache"
        self.slot = self.Slot(str(self.out))

    def keep(self, data: bytes) -> str:
        (self.out / f"{self.SHA}.json").write_bytes(data)
        return lanes.keep_listing(str(self.cache), self.SHA, self.slot)

    def test_a_listing_is_kept_as_a_file_the_runners_read(self):
        self.assertEqual(self.keep(json.dumps(LISTING).encode()), "")
        kept = self.cache / f"{self.SHA}.json"
        self.assertEqual(json.loads(kept.read_text()), LISTING)
        self.assertEqual(kept.stat().st_mode & 0o777, 0o640)

    def test_no_listing_needs_nothing(self):
        self.assertEqual(lanes.keep_listing(str(self.cache), self.SHA, self.slot), "")
        self.assertFalse(self.cache.exists())

    def test_what_is_no_json_object_is_not_kept(self):
        for data, why in (
            (b"[" * 200000 + b"]" * 200000, "no JSON"),
            (b"{", "no JSON"),
            (b"[1, 2]", "not a JSON object"),
        ):
            got = self.keep(data)
            self.assertTrue(
                got.startswith(f"the listing for {self.SHA[:10]} was not kept"), got
            )
            self.assertIn(why, got)
        self.assertFalse((self.cache / f"{self.SHA}.json").exists())


@unittest.skipUnless(cmake_dir(), "the tool configures with cmake and lists with ctest")
class TestEndToEnd(unittest.TestCase):
    """A provider tree whose rocKE stages installed pytest entries: rocke_pytest_core (standard, `-m "not
    integration"`) over tests/core, rocke_vis (comprehensive only) over tests/vis."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        py = sys.executable
        platform = textwrap.dedent(
            f"""\
            cmake_minimum_required(VERSION 3.16)
            project(rocke LANGUAGES NONE)
            enable_testing()
            if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
              set_property(GLOBAL APPEND_STRING PROPERTY
                "EXTERNAL_TEST_INSTALL_STAGING_${{ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}}"
                "add_test(rocke_pytest_core \\"{py}\\" \\"-m\\" \\"pytest\\" \\"./tests/core\\" \\"-m\\" \\"not integration\\")\\nadd_test(rocke_vis \\"{py}\\" \\"-m\\" \\"pytest\\" \\"./tests/vis\\")\\n")
            endif()
            """
        )
        files = {
            "shared/ctest/TestCategories.cmake": FAKE_CATEGORIES,
            "prov/CMakeLists.txt": 'set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")\n',
            "prov/src/CMakeLists.txt": '"${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"\n',
            "prov/ROCKE_ENGINE_test_categories_external.yaml": "quick: rocke_pytest_core\n"
            "standard: rocke_pytest_core\ncomprehensive: rocke_pytest_core rocke_vis\n",
            "prov/rocke/CMakeLists.txt": "add_subdirectory(platform)\n",
            "prov/rocke/platform/CMakeLists.txt": platform,
            "prov/rocke/platform/tests/pytest.ini": "[pytest]\nmarkers =\n    integration: slow\n",
            "prov/rocke/platform/tests/core/test_a.py": "def test_a():\n    assert True\n",
            "prov/rocke/platform/tests/core/test_t.py": "import pytest\n\ntorch = pytest.importorskip('torch')\n\n\n"
            "def test_t():\n    assert torch\n",
            "prov/rocke/platform/tests/core/test_i.py": "import pytest\n\npytestmark = pytest.mark.integration\n\n\n"
            "def test_i():\n    pass\n",
            "prov/rocke/platform/tests/vis/test_v.py": "def test_v():\n    pass\n",
            "prov/rocke/library/tests/test_lonely.py": "def test_l():\n    pass\n",
        }
        for path, text in files.items():
            p = r / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        g(r, "init", "-q", "-b", "develop")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        self.tmp = pathlib.Path(self.d.name) / "tmp"
        self.tmp.mkdir()
        self.cache = pathlib.Path(self.d.name) / "cache"

    def run_tool(self, *targets, collect=False, python=None) -> dict:
        env = {
            "TMPDIR": str(self.tmp),
            "PATH": f"{cmake_dir()}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        with mock.patch.dict(os.environ, env):
            out = lo.run(
                str(self.repo),
                list(targets),
                rev="develop",
                K=K,
                lanes=LANES,
                absent={"torch"},
                cache=self.cache,
                do_collect=collect,
                collect_python=python,
            )
        self.assertEqual(list(self.tmp.iterdir()), [])  # the scratch tree is gone
        self.assertIsNone(out["error"], out)
        return {r["target"]: r for r in out["results"]}

    def test_the_verdicts_from_a_configured_tree(self):
        got = self.run_tool(
            "platform/tests/core/test_a.py",
            "platform/tests/core/test_t.py",
            "platform/tests/core/test_i.py",
            "platform/tests/vis/test_v.py",
            "library/tests/test_lonely.py",
        )
        self.assertEqual(
            {t: r["verdict"] for t, r in got.items()},
            {
                "platform/tests/core/test_a.py": "runs",
                "platform/tests/core/test_t.py": "skips",
                "platform/tests/core/test_i.py": "no lane",
                "platform/tests/vis/test_v.py": "no lane",
                "library/tests/test_lonely.py": "no lane",
            },
        )
        self.assertEqual(
            got["platform/tests/core/test_a.py"]["entry"], "rocke_pytest_core"
        )
        self.assertIn("comprehensive", got["platform/tests/vis/test_v.py"]["why"])
        # The entries are cached by commit: the next call needs no cmake.
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        with mock.patch.object(lo.shutil, "which", return_value=None):
            again = lo.run(
                str(self.repo),
                ["platform/tests/core/test_a.py"],
                rev="develop",
                K=K,
                lanes=LANES,
                cache=self.cache,
            )
        self.assertEqual(again["results"][0]["verdict"], "runs")
        # A bare file name is the tree's only file of that name; a file the tree lacks reads NOT RUN (exit 3).
        with mock.patch.object(lo.shutil, "which", return_value=None):
            out = lo.run(
                str(self.repo),
                ["test_t.py", "platform/tests/core/test_new.py"],
                rev="develop",
                K=K,
                lanes=LANES,
                absent={"torch"},
                cache=self.cache,
            )
        bare, new = out["results"]
        self.assertEqual(bare["verdict"], "skips")
        self.assertIn("read as platform/tests/core/test_t.py", bare["detail"][0])
        self.assertEqual(new["verdict"], "not run")
        self.assertIn(
            "NOT RUN: platform/tests/core/test_new.py is not in", lo.line(new)
        )
        self.assertEqual(lo.code(out), 3)

    @unittest.skipUnless(_venv_python(), "the collect needs a python with pytest")
    def test_the_collect_as_ci(self):
        got = self.run_tool(
            "platform/tests/core/test_a.py",
            "platform/tests/core/test_t.py",
            collect=True,
            python=_venv_python(),
        )
        a, t = (
            got["platform/tests/core/test_a.py"],
            got["platform/tests/core/test_t.py"],
        )
        self.assertEqual(a["verdict"], "runs")
        self.assertTrue(any("1 would run, 0 skip" in d for d in a["detail"]), a)
        self.assertEqual(t["verdict"], "skips")
        self.assertTrue(any("0 would run, 1 skip" in d for d in t["detail"]), t)


if __name__ == "__main__":
    unittest.main()

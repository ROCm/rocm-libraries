# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ctest_labels.py: the CTest entries read from ctest's JSON with their
add_test() and labels, the comparison of the merge base with the head and its leads, the report, the checks wiring,
and the check end to end on a small provider tree where the PR moves a test out of the directory that applies the YAML
tiers (as #11111 moved two host tests to library/cpp/tests), deletes one and adds one with no tier.
Temp git repos, no network; the end-to-end test needs cmake and ctest (it configures a few CMake files, no
compiler)."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ctest_labels.py"
sys.path.insert(0, str(TOOL.parent))
import ctest_labels as cl  # noqa: E402

ROOT = pathlib.Path("/w/base")
K = "prov/rocke"


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def entry(labels, file=None, line=None, d=None) -> dict:
    return {
        "labels": sorted(labels),
        "at": {"file": file, "line": line, "dir": d} if file else None,
    }


class TestEntries(unittest.TestCase):
    def test_labels_and_the_add_test_of_each_entry(self):
        data = {
            "kind": "ctestInfo",
            "backtraceGraph": {
                "commands": ["add_test", "include"],
                "files": [
                    "/w/shim/CMakeLists.txt",
                    f"{ROOT}/{K}/platform/tests/CMakeLists.txt",
                    f"{ROOT}/{K}/platform/tests/cmake/InstanceTests.cmake",
                ],
                "nodes": [
                    {"file": 1},
                    {"file": 1, "line": 13, "command": 1, "parent": 0},
                    {"file": 2, "parent": 1},
                    {"file": 2, "line": 107, "command": 0, "parent": 2},
                ],
            },
            "tests": [
                {
                    "name": "host",
                    "backtrace": 3,
                    "properties": [
                        {"name": "LABELS", "value": ["ckc", "unit", "quick"]}
                    ],
                },
                {
                    "name": "py",
                    "properties": [
                        {"name": "LABELS", "value": ["ckc;python;quick", "standard"]}
                    ],
                },
                {"name": "bare"},
            ],
        }
        got = cl.entries(json.dumps(data), ROOT, K)
        self.assertEqual(
            got["host"],
            {
                "labels": ["ckc", "quick", "unit"],
                "at": {
                    "file": "platform/tests/cmake/InstanceTests.cmake",
                    "line": 107,
                    "dir": "platform/tests",
                },
                "cmd": [],
                "cwd": "",
            },
        )
        # An escaped-semicolon label (the installed text's "ckc\;python\;quick") reads as its parts, as `-L` does.
        self.assertEqual(
            got["py"],
            {
                "labels": ["ckc", "python", "quick", "standard"],
                "at": None,
                "cmd": [],
                "cwd": "",
            },
        )
        self.assertEqual(got["bare"], {"labels": [], "at": None, "cmd": [], "cwd": ""})
        with self.assertRaises(cl.prtree.NotRun):
            cl.entries("not json", ROOT, K)


class TestTestMap(unittest.TestCase):
    """Each changed test file -> its CTest entries -> their tiers -> the PR-CI lanes that run it; a test no lane runs
    is a lead (agent_script_overlap.md: the test -> entry -> tier -> lane lookup by hand in 29 of 39 reviews).
    """

    STD = ["ckc", "python", "quick", "standard"]
    HEAD = {
        "build": {
            "rocke_host": {"labels": ["ckc", "host"], "cmd": ["/x/host"], "cwd": ""},
            "rocke_parity": {
                "labels": ["ckc", "python"],
                "cwd": "platform/tests",
                "cmd": [
                    "python",
                    "-m",
                    "pytest",
                    "platform/tests/core/test_parity.py",
                    "-q",
                ],
            },
        },
        "installed": {
            "rocke_pytest_other": {
                "labels": STD,
                "cwd": "",
                "cmd": [
                    "python",
                    "-m",
                    "pytest",
                    "./tests",
                    "--ignore",
                    "./tests/library",
                    "--ignore",
                    "./tests/core",
                    "--ignore",
                    "./tests/tools",
                    "--ignore",
                    "./tests/in_tree",
                    "-m",
                    "not integration",
                    "-v",
                ],
            },
            "rocke_pytest_core": {
                "labels": STD,
                "cwd": "",
                "cmd": ["python", "-m", "pytest", "./tests/core"],
            },
            "rocke_pytest_runtime_tools": {
                "labels": STD,
                "cwd": "",
                "cmd": [
                    "python",
                    "-m",
                    "pytest",
                    "./tests/tools",
                    "./tests/in_tree",
                    "-v",
                ],
            },
            "rocke_library_dsa_pytest": {
                "labels": ["ckc", "python", "quick"],
                "cwd": "",
                "cmd": [
                    "python",
                    "-m",
                    "pytest",
                    "./tests/library/tests/test_dsa.py",
                    "-v",
                ],
            },
            "rocke_library_gpu_pytest": {
                "labels": STD + ["ex_gpu"],
                "cwd": "",
                "cmd": [
                    "python",
                    "-m",
                    "pytest",
                    "./tests/library/tests/test_numeric.py",
                    "-c",
                    "./tests/pytest.ini",
                ],
            },
        },
    }
    SUITES = {"platform/python/rocke/benchmark/perf/tests": {"test_pmc.py"}}
    # The PR-CI lanes of this test's config ci_test_tiers.
    LANES = [
        {
            "jobs": "PR test shards",
            "tree": "installed",
            "tier": "standard",
            "exclude": ["ex_gpu"],
        }
    ]

    def test_each_file_maps_to_its_entries_tiers_and_lanes_and_one_no_lane_runs_is_a_lead(
        self,
    ):
        files = [
            "library/tests/test_dsa.py",
            "library/tests/test_numeric.py",
            "library/tests/test_nowhere.py",
            "platform/python/rocke/benchmark/perf/tests/test_cli.py",
            "platform/python/rocke/benchmark/perf/tests/test_pmc.py",
            "platform/tests/core/test_parity.py",
            "platform/tests/test_engine_linkage.py",
        ]
        rows, leads = cl.test_map(files, self.HEAD, self.SUITES, self.LANES)
        got = {
            r["file"]: [
                (x["tree"], x["entry"], x["how"], x["tiers"], x["lanes"])
                for x in r["entries"]
            ]
            for r in rows
        }
        lane = [self.LANES[0]["jobs"]]
        self.assertEqual(
            got["platform/tests/test_engine_linkage.py"],
            [
                (
                    "installed",
                    "rocke_pytest_other",
                    "direct",
                    ["quick", "standard"],
                    lane,
                )
            ],
        )
        self.assertEqual(
            got[
                "platform/tests/core/test_parity.py"
            ],  # not rocke_pytest_other: it ignores tests/core
            [
                ("build", "rocke_parity", "direct", [], []),
                (
                    "installed",
                    "rocke_pytest_core",
                    "direct",
                    ["quick", "standard"],
                    lane,
                ),
            ],
        )
        self.assertEqual(
            got["platform/python/rocke/benchmark/perf/tests/test_cli.py"],
            [
                (
                    "installed",
                    "rocke_pytest_runtime_tools",
                    "in-tree",
                    ["quick", "standard"],
                    lane,
                )
            ],
        )
        self.assertEqual(
            got["platform/python/rocke/benchmark/perf/tests/test_pmc.py"], []
        )  # the runner leaves it out
        self.assertEqual(
            got["library/tests/test_dsa.py"],
            [("installed", "rocke_library_dsa_pytest", "direct", ["quick"], [])],
        )
        self.assertEqual(
            {x["name"]: x["why"] for x in leads},
            {
                "library/tests/test_dsa.py": "rocke_library_dsa_pytest (installed) has tiers quick",
                "library/tests/test_numeric.py": "rocke_library_gpu_pytest (installed) has the label ex_gpu, which the "
                "lanes leave out",
                "library/tests/test_nowhere.py": "no CTest entry in either tree runs it",
                "platform/python/rocke/benchmark/perf/tests/test_pmc.py": "no CTest entry in either tree runs it",
            },
        )
        self.assertEqual(
            cl.describe(leads[0]),
            "no PR-CI lane runs library/tests/test_dsa.py, which the PR adds or "
            "changes: rocke_library_dsa_pytest (installed) has tiers quick",
        )
        self.assertTrue(
            cl.summary(leads).startswith(
                "4 leads: no PR-CI lane runs library/tests/test_dsa.py"
            )
        )
        lines = cl.map_lines(rows, self.LANES)
        self.assertEqual(
            lines[0],
            "test map (test -> CTest entry -> tiers -> PR-CI lane): lanes PR test shards run the installed tree, "
            "-L standard -LE ex_gpu",
        )
        self.assertIn("   library/tests/test_nowhere.py: in no entry -> no lane", lines)
        self.assertIn(
            "   library/tests/test_numeric.py: installed rocke_library_gpu_pytest [quick standard; ex_gpu] "
            "-> no lane",
            lines,
        )

    def test_without_lanes_the_map_stops_at_the_tiers(self):
        # No PR-CI lane configured: each file still maps to its entries and tiers, and no lead says no lane runs it.
        files = ["library/tests/test_dsa.py", "library/tests/test_nowhere.py"]
        rows, leads = cl.test_map(files, self.HEAD, None, [])
        self.assertEqual(
            [(x["entry"], x["tiers"], x["lanes"]) for x in rows[0]["entries"]],
            [("rocke_library_dsa_pytest", ["quick"], [])],
        )
        self.assertEqual(rows[1]["entries"], [])
        self.assertEqual(leads, [])
        self.assertEqual(
            cl.map_lines(rows, [])[0],
            "test map (test -> CTest entry -> tiers -> PR-CI lane): lanes unknown (config ci_test_tiers names none)",
        )

    def test_config_names_the_lanes(self):
        lanes = [{"jobs": "quick shard", "tree": "installed", "tier": "quick"}]
        rows, leads = cl.test_map(["library/tests/test_dsa.py"], self.HEAD, None, lanes)
        self.assertEqual(rows[0]["entries"][0]["lanes"], ["quick shard"])
        self.assertEqual(leads, [])

    def test_installed_paths_map_back_to_the_source(self):
        self.assertEqual(
            cl.source_path("./tests/library/tests/test_a.py::t", "installed"),
            "library/tests/test_a.py",
        )
        self.assertEqual(cl.source_path("./tests", "installed"), "platform/tests")
        self.assertEqual(
            cl.source_path("./rocke/tests/helpers", "installed"),
            "platform/python/rocke/tests/helpers",
        )
        self.assertIsNone(cl.source_path("../elsewhere/test_x.py", "installed"))
        self.assertEqual(
            cl.source_path("platform/tests/core/test_x.py", "build"),
            "platform/tests/core/test_x.py",
        )


class TestCompare(unittest.TestCase):
    TIERS = ["quick", "standard", "comprehensive", "full"]

    def trees(self):
        base = {
            "build": {
                "moved": entry(
                    ["ckc", "unit"] + self.TIERS + ["smoke"],
                    "platform/tests/x.cmake",
                    7,
                    "platform/tests",
                ),
                "gone": entry(
                    ["ckc", "quick"], "platform/tests/x.cmake", 9, "platform/tests"
                ),
                "same": entry(
                    ["ckc", "quick"], "platform/tests/x.cmake", 11, "platform/tests"
                ),
                "relabel": entry(
                    ["ckc", "quick", "gpu"],
                    "platform/tests/x.cmake",
                    12,
                    "platform/tests",
                ),
            },
            "installed": {
                "moved": entry(["ckc"] + self.TIERS),
                "gone": entry(["quick"]),
                "same": entry(["quick"]),
                "inst_only": entry(["quick"]),
            },
        }
        head = {
            "build": {
                "moved": entry(
                    ["ckc", "unit"],
                    "library/cpp/tests/CMakeLists.txt",
                    17,
                    "library/cpp/tests",
                ),
                "same": entry(
                    ["ckc", "quick"], "platform/tests/x.cmake", 11, "platform/tests"
                ),
                "relabel": entry(
                    ["ckc", "quick", "host"],
                    "platform/tests/x.cmake",
                    12,
                    "platform/tests",
                ),
                "new": entry(
                    ["ckc", "unit"], "platform/tests/x.cmake", 20, "platform/tests"
                ),
            },
            "installed": {
                "moved": entry(["ckc"] + self.TIERS),
                "same": entry(["quick"]),
                "new": entry(["ckc", "quick"]),
            },
        }
        return base, head

    def test_tier_losses_departures_and_untiered_additions_are_leads(self):
        base, head = self.trees()
        res = cl.compare(base, head)
        leads = [(x["kind"], x["tree"], x["name"]) for x in res["leads"]]
        self.assertEqual(
            leads,
            [
                ("tiers", "build", "moved"),
                ("removed", "build", "gone"),
                ("untiered", "build", "new"),
                ("removed", "installed", "gone"),
                ("removed", "installed", "inst_only"),
            ],
        )
        self.assertEqual(
            res["trees"]["build"]["relabelled"],
            [
                {
                    "name": "moved",
                    "lost": ["comprehensive", "full", "quick", "smoke", "standard"],
                    "gained": [],
                },
                {"name": "relabel", "lost": ["gpu"], "gained": ["host"]},
            ],
        )
        text = cl.describe(res["leads"][0])
        self.assertEqual(
            text,
            "(build tree) moved loses its tier labels quick standard comprehensive full (and "
            "smoke); its add_test() moved from platform/tests/x.cmake:7 to "
            "library/cpp/tests/CMakeLists.txt:17 (directory platform/tests -> library/cpp/tests); "
            "the installed tree keeps quick standard comprehensive full",
        )
        self.assertEqual(
            cl.describe(res["leads"][1]),
            "(build tree) gone is an entry at the base but not at the head; the installed tree "
            "has none either",
        )
        self.assertIn(
            "new is new with no tier label (ckc unit, add_test() at platform/tests/x.cmake:20)",
            cl.describe(res["leads"][2]),
        )
        self.assertEqual(
            cl.describe(res["leads"][4]),
            "(installed tree) inst_only is an entry at the base but not at the head; the build "
            "tree has none either",
        )
        res.update(
            head="h" * 40,
            mb="m" * 40,
            yaml={"base": True, "head": True},
            notes=[],
            secs=1.0,
        )
        lines = cl.render(res)
        self.assertEqual(
            lines[-1],
            "LABELS fail 5 leads: moved lose quick standard comprehensive full in the build "
            "tree (add_test moved; the installed tree keeps them); +4 more",
        )
        self.assertIn("relabelled (build tree): relabel: -gpu +host", lines)
        self.assertIn(
            "added (installed tree): new", lines
        )  # it has a tier there: listed, not a lead

    def test_nothing_lost_reads_clean_and_nothing_run_never_does(self):
        same = {
            "build": {"a": entry(["quick"])},
            "installed": {"a": entry(["quick"]), "b": entry([])},
        }
        res = dict(cl.compare(same, same), head="h" * 40, mb="m" * 40)
        self.assertEqual(
            cl.verdict(res),
            (
                "pass",
                "no entry loses a tier, leaves a tree or comes without one (build "
                "1 -> 1, installed 2 -> 2 entries)",
            ),
        )
        self.assertEqual(
            cl.verdict({"error": "no cmake or ctest on PATH", "leads": []}),
            ("skip", "NOT RUN: no cmake or ctest on PATH"),
        )
        self.assertEqual(
            cl.render({"error": "x", "leads": []}),
            ["NOT RUN: x", "LABELS skip NOT RUN: x"],
        )
        bad = {
            "error": None,
            "head_error": "CMake Error at a/CMakeLists.txt:3 (message): boom",
            "leads": [],
        }
        self.assertEqual(cl.verdict(bad)[0], "fail")


class TestChecksWiring(unittest.TestCase):
    PLAN = {
        "families": [],
        "tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_every_review_runs_it_at_the_head_and_a_gate_does_not(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(
            cfg,
            7,
            "a" * 40,
            "follow-up",
            self.PLAN,
            None,
            reviewed="b" * 40,
            merge_only=True,
        )
        sec = script.split(f"sec '{review.LABELS_SECTION}'", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            f"nice -n 10 timeout {review.LABELS_BUDGET_S + 120} python -I -B $M/tools/ctest_labels.py --repo "
            f"$R0 --head {'a' * 40} --out $SO/labels --rocke-path $K --base $BREF --budget {review.LABELS_BUDGET_S}",
            sec,
        )  # a merge-only round too: a conflict resolution can drop an entry
        self.assertIn(
            '\nprr labels "$OUT/labels"\n', sec
        )  # the results copied back, as data
        self.assertIn("st labels ", sec)
        self.assertIn(
            "LABELS skip no result", sec
        )  # a check that wrote nothing never reads clean
        self.assertIn("labels", review.LEAD_SECTIONS)
        self.assertGreater(
            script.index(review.LABELS_SECTION),
            script.rindex('\nif [ "$MERGED" = 1 ]; then\n'),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            cfg, 7, "a" * 40, "first-round", self.PLAN, None, mode="gate"
        )
        self.assertNotIn("ctest_labels.py", gate)


def cmake_dir() -> str | None:
    """A directory with cmake and ctest: PATH's, else the checks' venv ($RKMON_VENV, or beside this checkout)."""
    found = shutil.which("cmake")
    if found and shutil.which("ctest"):
        return str(pathlib.Path(found).parent)
    for venv in (
        [pathlib.Path(os.environ["RKMON_VENV"]) / "bin"]
        if os.environ.get("RKMON_VENV")
        else []
    ) + [pathlib.Path(__file__).resolve().parents[2] / "venv" / "bin"]:
        if (venv / "cmake").exists() and (venv / "ctest").exists():
            return str(venv)
    return None


FAKE_CATEGORIES = """\
# A stand-in for rocm-libraries' shared/ctest/TestCategories.cmake: "<tier>: name name" rows label tests by name, in the
# caller's directory scope (the build tree) or in an installed file, for the tests it adds.
function(apply_ctest_category_labels yaml_file)
  set(install_file "${ARGV1}")
  file(STRINGS "${yaml_file}" _rows)
  set(_all "")
  foreach(_row IN LISTS _rows)
    if(_row MATCHES "^([a-z]+): (.*)$")
      set(_tier "${CMAKE_MATCH_1}")
      string(REPLACE " " ";" _names "${CMAKE_MATCH_2}")
      foreach(_t IN LISTS _names)
        list(APPEND _tiers_${_t} "${_tier}")
        list(APPEND _all "${_t}")
      endforeach()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _all)
  if(install_file)
    file(READ "${install_file}" _have)
  endif()
  foreach(_t IN LISTS _all)
    if(install_file)
      string(FIND "${_have}" "add_test(${_t} " _at)
      if(NOT _at EQUAL -1)
        file(APPEND "${install_file}" "set_tests_properties(${_t} PROPERTIES LABELS \\"${_tiers_${_t}}\\")\\n")
      endif()
    elseif(TEST ${_t})
      set_property(TEST ${_t} APPEND PROPERTY LABELS "${_tiers_${_t}}")
    endif()
  endforeach()
endfunction()
"""

PLATFORM = """\
cmake_minimum_required(VERSION 3.16)
project(rocke LANGUAGES NONE)
enable_testing()
add_subdirectory(tests)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../library/cpp/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../library/cpp" "${CMAKE_CURRENT_BINARY_DIR}/library_cpp")
endif()
set(_entries "")
get_property(_targets GLOBAL PROPERTY ROCKE_INSTALL_TEST_TARGETS)
foreach(_t IN LISTS _targets)
  string(APPEND _entries "add_test(${_t} \\"../${_t}\\")\\n"
         "set_tests_properties(${_t} PROPERTIES LABELS \\"ckc;unit;host;quick\\")\\n")
endforeach()
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set_property(GLOBAL APPEND_STRING PROPERTY
               "EXTERNAL_TEST_INSTALL_STAGING_${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}" "${_entries}")
endif()
"""


def host_test(name: str) -> str:
    return (
        f'add_test(NAME {name} COMMAND "${{CMAKE_COMMAND}}" -E true)\n'
        f'set_tests_properties({name} PROPERTIES LABELS "ckc;unit;host")\n'
        f"set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS {name})\n"
    )


@unittest.skipUnless(
    cmake_dir(), "the check configures with cmake and lists with ctest"
)
class TestCheck(unittest.TestCase):
    """A provider tree (`prov`, rocKE at prov/rocke) whose base registers host_a, host_b and host_gone in
    platform/tests, where the YAML tiers apply; the PR moves host_b to library/cpp/tests, deletes host_gone and adds
    host_new, which the YAML does not name."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        self.write(
            {
                "shared/ctest/TestCategories.cmake": FAKE_CATEGORIES,
                "prov/CMakeLists.txt": 'set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")\n',
                "prov/src/CMakeLists.txt": 'apply_ctest_category_labels("${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}" '
                '"${f}")\n',
                "prov/ROCKE_ENGINE_test_categories_external.yaml": "quick: host_a host_b\n"
                "standard: host_a host_b\n",
                "prov/rocke/CMakeLists.txt": "add_subdirectory(platform)\n",
                "prov/rocke/platform/CMakeLists.txt": PLATFORM,
                "prov/rocke/platform/tests/cmake/TestCategoryLabels.cmake": textwrap.dedent(
                    """\
                        if(COMMAND apply_ctest_category_labels AND HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
                           AND ROCKE_ENGINE_CTEST_CATEGORIES_YAML)
                          apply_ctest_category_labels("${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}")
                        endif()
                        """
                ),
                "prov/rocke/platform/tests/CMakeLists.txt": host_test("host_a")
                + host_test("host_b")
                + host_test("host_gone")
                + "include(cmake/TestCategoryLabels.cmake)\n",
            }
        )
        g(r, "init", "-q", "-b", "develop")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        self.write(
            {
                "prov/rocke/platform/tests/CMakeLists.txt": host_test("host_a")
                + host_test("host_new")
                + "include(cmake/TestCategoryLabels.cmake)\n",
                "prov/rocke/library/cpp/CMakeLists.txt": "add_subdirectory(tests)\n",
                "prov/rocke/library/cpp/tests/CMakeLists.txt": host_test("host_b"),
            }
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")
        self.tmp = pathlib.Path(self.d.name) / "tmp"
        self.tmp.mkdir()

    def write(self, files: dict) -> None:
        for path, text in files.items():
            p = self.repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def run_check(self) -> dict:
        env = {
            "TMPDIR": str(self.tmp),
            "PATH": f"{cmake_dir()}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        with mock.patch.dict(os.environ, env):
            res = cl.run(
                str(self.repo),
                self.head,
                pathlib.Path(self.d.name) / "out",
                K="prov/rocke",
                base="develop",
            )
        self.assertEqual(
            list(self.tmp.iterdir()), []
        )  # the scratch trees and builds are gone
        self.assertEqual(g(self.repo, "status", "--porcelain"), "")
        return res

    def test_a_test_moved_out_of_the_labelled_directory_loses_its_tiers_in_the_build_tree_only(
        self,
    ):
        res = self.run_check()
        self.assertIsNone(res["error"], res)
        self.assertEqual(res["yaml"], {"base": True, "head": True})
        self.assertEqual(res["notes"], [])
        leads = [(x["kind"], x["tree"], x["name"]) for x in res["leads"]]
        self.assertEqual(
            leads,
            [
                ("tiers", "build", "host_b"),
                ("removed", "build", "host_gone"),
                ("untiered", "build", "host_new"),
                ("removed", "installed", "host_gone"),
            ],
        )
        moved = res["leads"][0]
        self.assertEqual(moved["tiers"], ["quick", "standard"])
        self.assertEqual(
            (moved["was"]["file"], moved["was"]["dir"]),
            ("platform/tests/CMakeLists.txt", "platform/tests"),
        )
        self.assertEqual(
            moved["now"],
            {
                "file": "library/cpp/tests/CMakeLists.txt",
                "line": 1,
                "dir": "library/cpp/tests",
            },
        )
        self.assertEqual(
            moved["other"], ["quick", "standard"]
        )  # the installed tree keeps them, by name
        self.assertEqual(
            cl.verdict(res)[1],
            "4 leads: host_b lose quick standard in the build tree (add_test moved; "
            "the installed tree keeps them); +3 more",
        )
        out = pathlib.Path(self.d.name) / "out"
        self.assertTrue((out / "configure_head.log").exists())
        self.assertEqual(
            sorted(json.loads((out / "head_entries.json").read_text())["build"]),
            ["host_a", "host_b", "host_new"],
        )

    def test_a_head_that_does_not_configure_is_a_lead_and_the_cli_says_so(self):
        self.write(
            {
                "prov/rocke/library/cpp/tests/CMakeLists.txt": 'message(FATAL_ERROR "boom")\n'
            }
        )
        g(self.repo, "commit", "-qam", "broken")
        self.head = g(self.repo, "rev-parse", "HEAD")
        res = self.run_check()
        self.assertIn("boom", res["head_error"])
        self.assertEqual(cl.verdict(res)[0], "fail")
        env = {
            **os.environ,
            "TMPDIR": str(self.tmp),
            "PATH": f"{cmake_dir()}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--repo",
                str(self.repo),
                "--head",
                self.head,
                "--out",
                str(pathlib.Path(self.d.name) / "cli"),
                "--rocke-path",
                "prov/rocke",
                "--base",
                "develop",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(
            p.stdout.splitlines()[-1].startswith(
                "LABELS fail the head does not configure"
            )
        )


if __name__ == "__main__":
    unittest.main()

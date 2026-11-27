#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for run_dbsync_rocjitsu.py, the GPU-free dbsync runner.

The runner exits 0 when it validates nothing: an unknown family is skipped, and gtest reports
success when every StaticFDBSync param skips. These tests feed it the gtest JSON report and
output of a run that checked the wrong thing, and the family tables a bad merge can produce,
and require that it fails. They need neither rocjitsu nor a GPU.

Written against the standard library's unittest rather than pytest, like the other Python tests
under test/gtest. pre-commit runs it (see the miopen-dbsync-runner-tests hook).

    python3 -m unittest test_run_dbsync_rocjitsu
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_dbsync_rocjitsu as runner

GTEST_DIR = Path(__file__).resolve().parents[1]
KERNELS_DIR = GTEST_DIR.parents[1] / "src" / "kernels"
DB_SYNC_CPP = GTEST_DIR / "db_sync.cpp"


def declared_params():
    """Return the StaticFDBSync param names db_sync.cpp instantiates, in order."""
    text = DB_SYNC_CPP.read_text(encoding="utf-8")
    values = text.split("CPU_DBSync_NONE,", 1)[1].split("StaticFDBSyncParamName);", 1)[
        0
    ]
    return [
        f"{arch}_{num_cu}"
        for arch, num_cu in re.findall(r'std::make_pair\("(gfx\w+)",\s*(\d+)\)', values)
    ]


def all_runs():
    """Yield (family, arch, run spec) for every run FAMILY_MAP asks for."""
    for family, targets in runner.FAMILY_MAP.items():
        for target in targets:
            for expected in target["runs"]:
                yield family, target["arch"], expected


def gtest_report(directory, ran, params=None):
    """Write a gtest JSON report in which only the params named in ran completed."""
    params = params or declared_params()
    tests = []
    for name in params:
        arch, num_cu = name.split("_")
        tests.append(
            {
                "name": f"StaticFDBSync/{name}",
                "value_param": f'("{arch}", {num_cu})',
                "status": "RUN",
                "result": "COMPLETED" if name in ran else "SKIPPED",
            }
        )
    path = Path(directory) / "report.json"
    path.write_text(
        json.dumps(
            {"testsuites": [{"name": "Quick/CPU_DBSync_NONE", "testsuite": tests}]}
        )
    )
    return path


def fdb(db):
    return f"/work/build/share/miopen/db/{db}.HIP.fdb.txt"


class CheckOneParamRan(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def check(self, arch, expected, ran, opened):
        report = gtest_report(self.tmp, ran)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.check_one_param_ran(
                report, arch, expected, [fdb(db) for db in opened]
            )

    def fails(self, arch, expected, ran, opened):
        with self.assertRaises(SystemExit) as caught:
            self.check(arch, expected, ran, opened)
        return str(caught.exception.code)

    def test_every_run_passes_with_its_own_param_and_db(self):
        for family, arch, expected in all_runs():
            with self.subTest(family=family, param=expected["param"]):
                self.check(arch, expected, [expected["param"]], [expected["db"]])

    def test_right_arch_wrong_cu_param_fails(self):
        """gfx942_304 ran for the 228 CU run: the MI300X DB was checked, MI300A was not."""
        expected = {"cu": 228, "param": "gfx942_228", "db": "gfx942e4"}
        message = self.fails("gfx942", expected, ["gfx942_304"], ["gfx942130"])
        self.assertIn("StaticFDBSync/gfx942_228", message)
        self.assertIn("StaticFDBSync/gfx942_304", message)

    def test_right_param_on_the_wrong_db_fails(self):
        expected = {"cu": 228, "param": "gfx942_228", "db": "gfx942e4"}
        message = self.fails("gfx942", expected, ["gfx942_228"], ["gfx942130"])
        self.assertIn("gfx942e4.HIP.fdb.txt", message)
        self.assertIn("gfx942130.HIP.fdb.txt", message)

    def test_params_that_do_not_name_their_db_need_the_basename(self):
        """gfx1100_0 and gfx1200_0 run at any CU count, so only the opened file tells them apart."""
        for param, arch, db, wrong in (
            ("gfx1100_0", "gfx1100", "gfx110060", "gfx110036"),
            ("gfx1200_0", "gfx1200", "gfx120020", "gfx120040"),
        ):
            with self.subTest(param=param):
                expected = {"cu": 1, "param": param, "db": db}
                self.check(arch, expected, [param], [db])
                message = self.fails(arch, expected, [param], [wrong])
                self.assertIn(f"{db}.HIP.fdb.txt", message)

    def test_no_param_ran_fails(self):
        expected = {"cu": 256, "param": "gfx1250_256", "db": "gfx1250100"}
        self.assertIn("got []", self.fails("gfx1250", expected, [], []))

    def test_two_params_ran_fails(self):
        expected = {"cu": 228, "param": "gfx942_228", "db": "gfx942e4"}
        self.fails("gfx942", expected, ["gfx942_228", "gfx942_304"], ["gfx942e4"])

    def test_missing_or_repeated_fdb_line_fails(self):
        expected = {"cu": 228, "param": "gfx942_228", "db": "gfx942e4"}
        self.assertIn("got []", self.fails("gfx942", expected, ["gfx942_228"], []))
        self.fails("gfx942", expected, ["gfx942_228"], ["gfx942e4", "gfx942e4"])

    def test_missing_report_fails(self):
        expected = {"cu": 256, "param": "gfx950_256", "db": "gfx950100"}
        with self.assertRaises(SystemExit):
            runner.check_one_param_ran(
                Path(self.tmp) / "absent.json", "gfx950", expected, [fdb("gfx950100")]
            )


class RunCaptureFdb(unittest.TestCase):
    def run_python(self, code):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            paths = runner.run_capture_fdb([sys.executable, "-c", code])
        return paths, out.getvalue()

    def test_reads_the_find_db_path_gtest_prints_and_echoes_the_output(self):
        line = (
            'FDB: "/work/share/miopen/db/gfx110060.HIP.fdb.txt", '
            'PDB: "/work/share/miopen/db/gfx110060.db.txt", KDB: "/work/gfx1100.kdb"'
        )
        paths, echoed = self.run_python(
            f"print('Handle CU count: 48'); print({line!r})"
        )
        self.assertEqual(paths, ["/work/share/miopen/db/gfx110060.HIP.fdb.txt"])
        self.assertIn(line, echoed)
        self.assertIn("Handle CU count: 48", echoed)

    def test_nonzero_exit_raises(self):
        with self.assertRaises(runner.subprocess.CalledProcessError):
            self.run_python("import sys; sys.exit(3)")


class FamilyTables(unittest.TestCase):
    def test_shipped_tables_validate(self):
        runner.validate_tables()

    def test_gfx1250_is_a_one_element_target_list(self):
        self.assertEqual(
            runner.FAMILY_MAP["gfx125X-dcgpu"],
            [
                {
                    "arch": "gfx1250",
                    "rj_config": "gfx1250_mi455x_kmd_4gpu.json",
                    "runs": [{"cu": 256, "param": "gfx1250_256", "db": "gfx1250100"}],
                }
            ],
        )
        self.assertNotIn("gfx125X-dcgpu", runner.NOT_VALIDATED)

    def test_params_are_declared_and_dbs_are_shipped(self):
        params = declared_params()
        self.assertIn("gfx1250_256", params)
        for family, _, expected in all_runs():
            with self.subTest(family=family, param=expected["param"]):
                self.assertIn(expected["param"], params)
                for suffix in ("HIP.fdb.txt.bz2", "db.txt.bz2"):
                    shipped = KERNELS_DIR / f"{expected['db']}.{suffix}"
                    self.assertTrue(shipped.exists(), f"{shipped} is not shipped")

    def test_no_param_or_db_is_checked_twice(self):
        for key in ("param", "db"):
            values = [expected[key] for _, _, expected in all_runs()]
            with self.subTest(key=key):
                self.assertEqual(sorted(values), sorted(set(values)))

    def test_dict_entry_merged_into_not_validated_fails(self):
        """The old single-target FAMILY_MAP shape landing in NOT_VALIDATED, as a clean merge does."""
        misplaced = {"arch": "gfx1250", "rj_config": "x.json", "cus": [256]}
        with mock.patch.dict(runner.NOT_VALIDATED, {"gfx125X-dcgpu": misplaced}):
            with self.assertRaises(SystemExit) as caught:
                runner.validate_tables()
        self.assertIn("NOT_VALIDATED['gfx125X-dcgpu']", str(caught.exception.code))

    def test_old_shape_family_map_entry_fails(self):
        old = {"arch": "gfx1250", "rj_config": "x.json", "cus": [256]}
        with mock.patch.dict(runner.FAMILY_MAP, {"gfx125X-dcgpu": old}):
            with self.assertRaises(SystemExit) as caught:
                runner.validate_tables()
        self.assertIn("FAMILY_MAP['gfx125X-dcgpu']", str(caught.exception.code))

    def test_arch_in_both_tables_fails(self):
        with mock.patch.dict(
            runner.NOT_VALIDATED, {"gfx125X-dcgpu": {"gfx1250": "not ready"}}
        ):
            with self.assertRaises(SystemExit):
                runner.validate_tables()


class Main(unittest.TestCase):
    """main() must select a target for every FAMILY_MAP family. With no miopen_gtest it then
    stops on that, which is as far as a GPU-free test can follow it."""

    def run_main(self, family, test_type="standard"):
        with tempfile.TemporaryDirectory() as empty:
            env = {
                "AMDGPU_FAMILIES": family,
                "TEST_TYPE": test_type,
                "THEROCK_BIN_DIR": empty,
                "OUTPUT_ARTIFACTS_DIR": empty,
            }
            out = io.StringIO()
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
                try:
                    code = runner.main()
                except SystemExit as exit_:
                    code = exit_.code
        return code, out.getvalue()

    def test_every_family_gets_past_target_selection(self):
        for family in runner.FAMILY_MAP:
            with self.subTest(family=family):
                code, out = self.run_main(family)
                self.assertNotIn("no target to validate", out)
                self.assertIn("miopen_gtest not found", str(code))

    def test_gfx1250_standard_is_not_skipped(self):
        code, out = self.run_main("gfx125X-dcgpu")
        self.assertNotIn("skipping dbsync", out)
        self.assertNotEqual(code, 0)

    def test_unknown_family_still_skips_with_success(self):
        code, out = self.run_main("gfx1010")
        self.assertEqual(code, 0)
        self.assertIn("no target to validate", out)

    def test_quick_tier_still_skips_with_success(self):
        code, out = self.run_main("gfx125X-dcgpu", test_type="quick")
        self.assertEqual(code, 0)
        self.assertIn("TEST_TYPE=quick", out)

    def test_a_misplaced_entry_does_not_become_a_green_skip(self):
        misplaced = {"arch": "gfx1250", "rj_config": "x.json", "cus": [256]}
        with mock.patch.dict(runner.NOT_VALIDATED, {"gfx125X-dcgpu": misplaced}):
            with mock.patch.dict(runner.FAMILY_MAP):
                del runner.FAMILY_MAP["gfx125X-dcgpu"]
                code, _ = self.run_main("gfx125X-dcgpu")
        self.assertIn("NOT_VALIDATED", str(code))


if __name__ == "__main__":
    unittest.main()

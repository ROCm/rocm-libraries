#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for run_dbsync_rocjitsu.py, the GPU-free dbsync runner.

The runner exits 0 when it validates nothing: an unknown family is skipped, and gtest reports
success when every StaticFDBSync param skips. These tests feed it the gtest JSON report and
output of a run that checked the wrong thing, and the family tables a bad merge can produce,
and require that it fails. They need neither rocjitsu nor a GPU, so they run in pre-commit, where
there is no compiled binary. What the binary registers is the runner's job at run time (it checks
the binary's `--gtest_list_tests` before each run). The tests here cover that check with a fake
listing. They also read db_sync.cpp as text, with comments removed, as a quick early warning.

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


def declared_params(text=None):
    """Return the StaticFDBSync param names db_sync.cpp instantiates, in order.

    This is a text scan, so it is only an approximation of what gets compiled. Comments are removed
    first, because a value that was edited but left behind in a comment is not registered.
    """
    if text is None:
        text = DB_SYNC_CPP.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
    values = text.split("CPU_DBSync_NONE,", 1)[1].split("StaticFDBSyncParamName);", 1)[
        0
    ]
    return [
        f"{arch}_{num_cu}"
        for arch, num_cu in re.findall(r'std::make_pair\("(gfx\w+)",\s*(\d+)\)', values)
    ]


def gtest_listing(params):
    """Return what `miopen_gtest --gtest_list_tests --gtest_filter=*StaticFDBSync*` prints."""
    lines = ["Quick/CPU_DBSync_NONE."]
    for name in params:
        arch, num_cu = name.split("_")
        lines.append(f'  StaticFDBSync/{name}  # GetParam() = ("{arch}", {num_cu})')
    return "\n".join(lines) + "\n"


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

    def test_a_comma_in_the_quoted_path_is_part_of_the_path(self):
        """MIOPEN_SYSTEM_DB_PATH=/tmp/artifacts,local/db is a valid setting."""
        root = "/tmp/artifacts,local/db"
        line = (
            f'FDB: "{root}/gfx1250100.HIP.fdb.txt", '
            f'PDB: "{root}/gfx1250100.db.txt", KDB: "{root}/gfx1250.kdb"'
        )
        paths, echoed = self.run_python(f"print({line!r})")
        self.assertEqual(paths, [f"{root}/gfx1250100.HIP.fdb.txt"])
        self.assertIn(line, echoed)

    def test_the_quoted_path_is_taken_as_one_unit(self):
        """std::quoted escapes a quote and a backslash with a backslash, nothing else."""
        for printed, path in (
            ('"/with space/x.HIP.fdb.txt"', "/with space/x.HIP.fdb.txt"),
            ('"/a,b,c/x.HIP.fdb.txt"', "/a,b,c/x.HIP.fdb.txt"),
            (r'"/quo\"te/x.HIP.fdb.txt"', '/quo"te/x.HIP.fdb.txt'),
            (r'"/back\\slash/x.HIP.fdb.txt"', r"/back\slash/x.HIP.fdb.txt"),
            (r'"/a\", PDB: \"b/x.HIP.fdb.txt"', '/a", PDB: "b/x.HIP.fdb.txt'),
        ):
            with self.subTest(printed=printed):
                line = f'FDB: {printed}, PDB: "/p/x.db.txt", KDB: "/p/gfx1.kdb"'
                self.assertEqual(runner.fdb_path_from_line(line), path)

    def test_lines_that_are_not_an_fdb_announcement_give_no_path(self):
        for line in (
            "Handle CU count: 256 Parameter Value: 256",
            'PDB: "/p/x.db.txt", KDB: "/p/x.kdb"',
            'FDB: "/p/unterminated, PDB: "/p/x.db.txt"',
            "",
        ):
            with self.subTest(line=line):
                self.assertIsNone(runner.fdb_path_from_line(line))


class CommaInDatabasePath(unittest.TestCase):
    """GTest can pass with MIOPEN_SYSTEM_DB_PATH=/tmp/artifacts,local/db. The runner has to take the
    database name from that path and still fail when the name is the wrong one."""

    ROOT = "/tmp/artifacts,local/db"

    def announcement(self, db):
        return (
            f'FDB: "{self.ROOT}/{db}.HIP.fdb.txt", '
            f'PDB: "{self.ROOT}/{db}.db.txt", KDB: "{self.ROOT}/gfx1.kdb"'
        )

    def check(self, db):
        expected = {"cu": 256, "param": "gfx1250_256", "db": "gfx1250100"}
        with tempfile.TemporaryDirectory() as tmp:
            report = gtest_report(tmp, ["gfx1250_256"])
            paths = [runner.fdb_path_from_line(self.announcement(db))]
            with contextlib.redirect_stdout(io.StringIO()):
                runner.check_one_param_ran(report, "gfx1250", expected, paths)

    def test_the_expected_database_in_a_comma_path_passes(self):
        self.check("gfx1250100")

    def test_a_wrong_database_in_a_comma_path_still_fails(self):
        with self.assertRaises(SystemExit) as caught:
            self.check("gfx942130")
        self.assertIn("gfx1250100.HIP.fdb.txt", str(caught.exception.code))
        self.assertIn("gfx942130.HIP.fdb.txt", str(caught.exception.code))


STALE_REGISTRATION_CPP = """
INSTANTIATE_TEST_SUITE_P(Quick,
                         CPU_DBSync_NONE,
                         testing::Values(std::make_pair("gfx942", 304),
                                         // was std::make_pair("gfx1250", 256),
                                         /* std::make_pair("gfx1250", 256) */
                                         std::make_pair("gfx1250", 255)),
                         StaticFDBSyncParamName);
"""


class RegisteredParams(unittest.TestCase):
    """The runner checks the binary's `--gtest_list_tests` before a run. The binary is faked with
    the text gtest prints, so these run without miopen_gtest, rocjitsu or a GPU."""

    EXPECTED = {"cu": 256, "param": "gfx1250_256", "db": "gfx1250100"}

    def check(self, listing, expected=None):
        registered = runner.static_fdbsync_params_from_listing(listing)
        runner.check_param_registered(registered, "gfx1250", expected or self.EXPECTED)

    def test_every_run_is_found_in_the_listing_of_the_declared_params(self):
        listing = gtest_listing(declared_params())
        for family, arch, expected in all_runs():
            with self.subTest(family=family, param=expected["param"]):
                runner.check_param_registered(
                    runner.static_fdbsync_params_from_listing(listing), arch, expected
                )

    def test_listing_is_read_the_way_gtest_prints_it(self):
        listing = (
            "Quick/CPU_DBSync_NONE.\n"
            '  StaticFDBSync/gfx942_304  # GetParam() = ("gfx942", 304)\n'
            '  StaticFDBSync/gfx1250_256  # GetParam() = ("gfx1250", 256)\n'
            "Quick/CPU_DBSyncTarget_NONE.\n"
            "  KDBTargetID/0  # GetParam() = 0\n"
            "Other.StaticFDBSync/not_a_suite_line\n"
        )
        self.assertEqual(
            runner.static_fdbsync_params_from_listing(listing),
            ["StaticFDBSync/gfx942_304", "StaticFDBSync/gfx1250_256"],
        )
        self.check(listing)

    def test_the_suite_prefix_does_not_matter(self):
        """The Navi artifacts of an older build name the suite Smoke, not Quick."""
        self.check(gtest_listing(["gfx1250_256"]).replace("Quick/", "Smoke/"))

    def test_a_changed_num_cu_is_not_registered(self):
        listing = gtest_listing(["gfx942_304", "gfx1250_255"])
        with self.assertRaises(SystemExit) as caught:
            self.check(listing)
        message = str(caught.exception.code)
        self.assertIn("StaticFDBSync/gfx1250_256 is not registered", message)
        self.assertIn("StaticFDBSync/gfx1250_255", message)

    def test_a_stale_name_in_a_comment_is_not_registered(self):
        """The registration changed to 255 and the old 256 value is left in comments. The source
        scan and the compiled listing both have to report 256 as missing."""
        params = declared_params(STALE_REGISTRATION_CPP)
        self.assertEqual(params, ["gfx942_304", "gfx1250_255"])
        with self.assertRaises(SystemExit) as caught:
            self.check(gtest_listing(params))
        self.assertIn(
            "StaticFDBSync/gfx1250_256 is not registered", str(caught.exception.code)
        )

    def test_an_empty_listing_fails(self):
        with self.assertRaises(SystemExit) as caught:
            self.check("")
        self.assertIn("lists 0 StaticFDBSync params", str(caught.exception.code))

    def test_a_prefix_of_a_registered_param_is_not_enough(self):
        with self.assertRaises(SystemExit):
            self.check(gtest_listing(["gfx1250_2560"]))

    def run_python(self, code):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            registered = runner.list_registered_params([sys.executable, "-c", code])
        return registered, out.getvalue()

    def test_the_listing_command_is_run_and_echoed(self):
        listing = gtest_listing(["gfx942_304", "gfx1250_256"])
        registered, echoed = self.run_python(f"print({listing!r}, end='')")
        self.assertEqual(
            registered, ["StaticFDBSync/gfx942_304", "StaticFDBSync/gfx1250_256"]
        )
        self.assertIn("StaticFDBSync/gfx1250_256", echoed)

    def test_a_binary_that_cannot_list_fails(self):
        """miopen_gtest aborts without a device even when it only lists tests."""
        code = "import sys; print('terminate called: No device', file=sys.stderr); sys.exit(134)"
        with self.assertRaises(runner.subprocess.CalledProcessError):
            self.run_python(code)


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


class MainChecksRegistrationBeforeTheRun(unittest.TestCase):
    """main() with the rocjitsu build, the listing and the gtest run replaced by fakes, to see
    the order of the steps and what a binary that does not register the param leads to.
    """

    def run_main(self, listed):
        calls = []
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "dist"
            (dist / "bin").mkdir(parents=True)
            (dist / "bin" / "miopen_gtest").write_text("")
            (dist / "lib").mkdir()
            (dist / "lib" / "libMIOpenCKGroupedConv_gfx1250.so").write_text("")
            env = {
                "AMDGPU_FAMILIES": "gfx125X-dcgpu",
                "TEST_TYPE": "standard",
                "THEROCK_BIN_DIR": str(dist / "bin"),
                "OUTPUT_ARTIFACTS_DIR": str(dist),
            }

            def fake_list(cmd, **kw):
                calls.append(("list", cmd))
                return listed

            def fake_run(cmd, **kw):
                calls.append(("run", cmd))
                return ["/db/gfx1250100.HIP.fdb.txt"]

            def fake_check(report, arch, expected, fdb_paths):
                calls.append(("check", expected["param"]))

            patches = [
                mock.patch.dict(os.environ, env),
                mock.patch.object(runner, "ensure_build_tools"),
                mock.patch.object(
                    runner,
                    "build_rocjitsu",
                    return_value=(Path(tmp) / "rocjitsu", Path(tmp) / "src"),
                ),
                mock.patch.object(runner, "write_config"),
                mock.patch.object(runner, "list_registered_params", fake_list),
                mock.patch.object(runner, "run_capture_fdb", fake_run),
                mock.patch.object(runner, "check_one_param_ran", fake_check),
            ]
            os.chdir(tmp)
            try:
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    try:
                        code = runner.main()
                    except SystemExit as exit_:
                        code = exit_.code
            finally:
                os.chdir(cwd)
        return code, calls

    def test_a_registered_param_is_listed_then_run_then_checked(self):
        code, calls = self.run_main(["StaticFDBSync/gfx1250_256"])
        self.assertEqual(code, 0)
        self.assertEqual([kind for kind, _ in calls], ["list", "run", "check"])
        listing_cmd = calls[0][1]
        self.assertIn("--gtest_list_tests", listing_cmd)
        self.assertIn("--config", listing_cmd)

    def test_an_unregistered_param_fails_before_the_run_starts(self):
        code, calls = self.run_main(["StaticFDBSync/gfx1250_255"])
        self.assertIn("StaticFDBSync/gfx1250_256 is not registered", str(code))
        self.assertEqual([kind for kind, _ in calls], ["list"])


if __name__ == "__main__":
    unittest.main()

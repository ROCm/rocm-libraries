# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The prerequisite checks: one table (rkmon_lib/prerequisites.txt) behind rk doctor's rows (as rkmon_lib/checks.py's
render prints them). Every missing tool comes with what it blocks and its fix, an image of this package, which carries
every one; nothing is installed.
"""

import ast
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rkmon_lib import checks, doctor  # noqa: E402


class TestTable(unittest.TestCase):
    def test_every_row_is_complete(self):
        rows = doctor.table()
        self.assertGreaterEqual(len(rows), 12)
        for key, p in rows.items():
            self.assertIn(p.level, ("core", "review", "optional"), key)
            self.assertTrue(p.name and p.look and p.blocks, key)
            self.assertNotIn("|", p.blocks, key)
        self.assertEqual(
            {k for k, p in rows.items() if p.level == "core"},
            {"python", "git", "gh", "claude"},
        )

    def test_exit_codes(self):
        """3: a row of prerequisites.txt fails (by its name); 4: only other rows fail; 0: none fails (a WARN is
        advice)."""
        C, fix = checks.Check, doctor.check_again({})
        git = doctor.PREREQS["git"].name
        self.assertEqual(
            doctor.exit_code(
                [C("configuration", "PASS", "ok"), C(git, "WARN", "2.30", fix)]
            ),
            0,
        )
        self.assertEqual(
            doctor.exit_code(
                [C(git, "FAIL", "missing", fix), C("configuration", "FAIL", "bad", fix)]
            ),
            3,
        )
        self.assertEqual(
            doctor.exit_code(
                [C(git, "PASS", "2.43"), C("configuration", "FAIL", "bad", fix)]
            ),
            4,
        )


class TestReport(unittest.TestCase):
    def test_rk_doctor_prints_its_rows_as_checks_render_does_then_the_counts(self):
        again = doctor.check_again({})
        rows = [
            checks.Check("configuration", "PASS", "config.json"),
            checks.Check(
                "cmake", "WARN", doctor.blocking("missing", "the native builds"), again
            ),
            checks.Check("monitor state", "FAIL", "state.json does not read", again),
            checks.Check("team dashboard", "OFF", "off"),
            checks.Check("Ollama", "SKIP", "not needed"),
        ]
        self.assertEqual(
            doctor.report(rows),
            checks.render(rows) + "\n\n1 pass, 1 fail, 1 warn, 1 off, 1 skipped\n",
        )
        self.assertEqual(
            doctor.report(rows),
            "  PASS  configuration: config.json\n"
            "  WARN  cmake: missing; it blocks the native builds\n"
            "        fix: rkmon doctor\n"
            "  FAIL  monitor state: state.json does not read\n"
            "        fix: rkmon doctor\n"
            "  OFF   team dashboard: off\n"
            "  SKIP  Ollama: not needed\n"
            "\n"
            "1 pass, 1 fail, 1 warn, 1 off, 1 skipped\n",
        )

    def test_a_disk_short_of_room_says_what_it_blocks(self):
        with mock.patch.object(
            doctor.shutil,
            "disk_usage",
            return_value=types.SimpleNamespace(free=20 * 2**30),
        ), mock.patch.object(
            doctor, "claude_code", lambda *a, **k: ("1.0", "logged in")
        ):
            row = next(c for c in doctor.prerequisites(None) if c.name == "free disk")
        self.assertEqual(row.status, "WARN")
        self.assertRegex(
            row.found,
            r"^20 GiB free at \S+; the clone, worktrees and builds need tens of GiB \(60 GiB is comfortable\); it "
            r"blocks the clone, the worktrees and the builds$",
        )

    def test_a_missing_prerequisite_s_fix_is_an_image_of_this_package(self):
        """Every prerequisite is a tool every image of this package carries: a missing one's fix is such an image
        (doctor.image_fix), wherever the doctor runs."""
        with mock.patch.object(
            doctor, "run", return_value=(127, "")
        ), mock.patch.object(doctor, "claude_code", lambda *a, **k: ("", "unknown")):
            rows = {c.name: c for c in doctor.prerequisites(None)}
        for key in ("python", "git"):
            with self.subTest(key):
                row = rows[doctor.PREREQS[key].name]
                self.assertEqual(row.status, "FAIL", row)
                self.assertEqual(str(row.fix), str(doctor.image_fix()), row)

    def test_what_a_row_blocks_is_said_through_doctor_blocking(self):
        """A row says what it blocks one way, doctor.blocking's: no string of the toolkit spells "; it blocks" out by
        hand."""
        root, by_hand = HERE.parent, []
        for f in [root / "rkmon.py", *sorted((root / "rkmon_lib").glob("*.py"))]:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            home = {
                id(n)
                for fn in ast.walk(tree)
                if f.name == "doctor.py"
                and isinstance(fn, ast.FunctionDef)
                and fn.name == "blocking"
                for n in ast.walk(fn)
            }
            by_hand += [
                f"{f.relative_to(root)}:{n.lineno}"
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and "; it blocks" in n.value
                and id(n) not in home
            ]
        self.assertEqual(by_hand, [], "say what a row blocks with doctor.blocking")


class TestInAContainer(unittest.TestCase):
    """A container is told by its marker file, its variable or its cgroup, whatever its kernel: on Docker Desktop
    every container's /proc/version names WSL's kernel."""

    def tree(
        self, base: pathlib.Path, version: str, cgroup: str = "0::/\n", marker: str = ""
    ) -> pathlib.Path:
        (base / "proc" / "1").mkdir(parents=True)
        (base / "proc" / "version").write_text(version)
        (base / "proc" / "1" / "cgroup").write_text(cgroup)
        if marker:
            (base / marker).parent.mkdir(parents=True, exist_ok=True)
            (base / marker).write_text("")
        return base

    def test_a_container_is_told_whatever_its_kernel(self):
        wsl_kernel = "Linux version 6.18.54.1-microsoft-standard-WSL2 #1 SMP"
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            distro = self.tree(base / "distro", wsl_kernel)
            self.assertFalse(doctor.in_container(distro, environ={}))
            for marker, cgroup, env in (
                (".dockerenv", "0::/\n", {}),
                ("run/.containerenv", "0::/\n", {}),
                ("", "12:pids:/docker/abc\n", {}),
                ("", "0::/\n", {"container": "podman"}),
            ):
                name = marker or cgroup.strip() or str(env)
                with self.subTest(name):
                    root = self.tree(
                        base / name.replace("/", "_").replace(":", "_"),
                        wsl_kernel,
                        cgroup,
                        marker,
                    )
                    self.assertTrue(doctor.in_container(root, environ=env))


if __name__ == "__main__":
    unittest.main()

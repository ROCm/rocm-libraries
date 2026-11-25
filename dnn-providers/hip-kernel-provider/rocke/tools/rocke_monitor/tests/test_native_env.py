# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The native comparisons' switches a checks job exports (tools/native_env.py).

- A switch of run_all.py's NATIVE_PARITY_TESTS whose binary the build holds is printed as an export line, its path
  quoted; a switch whose binary the build lacks is left out.
- Only rocKE's own switches (ROCKE_ names) are printed: the names are the PR's and the job script that evaluates the
  lines is the monitor's, so PATH, LD_PRELOAD, BASH_ENV and a name that is no variable name never are.
- The build is PR code's, read by the monitor: a link at its root, or a root its owner (the slot's runner) does not
  own, is refused with nothing exported; the walk follows no link, goes no deeper than the result reader's DEPTH_MAX,
  and a binary is a regular file with an x bit, by lstat.
- A checks job's line (review.NATIVE_ENV, the one every section that runs pytest evaluates) names the slot's runner as
  the build's owner, and none on a host, where the build is the monitor's own.
"""

import os
import pathlib
import pwd
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "native_env.py"
sys.path.insert(0, str(ROOT / "worker"))
import pr_results  # noqa: E402

sys.path.insert(0, str(ROOT))
from rkmon_lib import review  # noqa: E402


def run(
    run_all: pathlib.Path, build: pathlib.Path, *owner: str
) -> subprocess.CompletedProcess:
    """The tool, as a checks job runs it."""
    return subprocess.run(
        [sys.executable, "-I", "-B", str(TOOL), str(run_all), str(build), *owner],
        capture_output=True,
        text=True,
    )


class TestNativeEnv(unittest.TestCase):
    def exports(self, pairs: list[tuple[str, str]], binaries: list[str]) -> list[str]:
        """The tool's lines for a run_all.py naming `pairs` (switch, binary) and a build holding `binaries`, with the
        build's folder read as BUILD."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            entries = tuple((v, n, "a label") for v, n in pairs)
            (root / "run_all.py").write_text(f"NATIVE_PARITY_TESTS = {entries!r}\n")
            build = root / "build" / "bin"
            build.mkdir(parents=True)
            for name in binaries:
                (build / name).write_text("#!/bin/sh\n")
                (build / name).chmod(0o755)
            r = run(root / "run_all.py", root / "build")
        self.assertEqual(r.returncode, 0, r.stderr)
        return [ln.replace(str(build), "BUILD") for ln in r.stdout.splitlines()]

    def test_a_switch_whose_binary_the_build_holds_is_exported(self):
        self.assertEqual(
            self.exports(
                [("ROCKE_BARRIER_EMIT", "barrier_emit"), ("ROCKE_GONE_EMIT", "gone")],
                ["barrier_emit"],
            ),
            ["export ROCKE_BARRIER_EMIT=BUILD/barrier_emit"],
        )

    def test_only_rockes_own_switches_are_exported(self):
        names = [
            "PATH",
            "LD_PRELOAD",
            "BASH_ENV",
            "rocke_lower",
            "ROCKE_",
            "XROCKE_A",
            "ROCKE_A;echo injected",
            "ROCKE_OK_EMIT",
        ]
        binaries = [f"b{i}" for i in range(len(names))]
        self.assertEqual(
            self.exports(list(zip(names, binaries)), binaries),
            [f"export ROCKE_OK_EMIT=BUILD/{binaries[-1]}"],
        )


class TestTheBuildIsPrCodes(unittest.TestCase):
    """A build PR code wrote, with links and files of other kinds planted in it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.run_all = self.root / "run_all.py"
        pairs = (
            ("ROCKE_A_EMIT", "a_emit", "a label"),
            ("ROCKE_B_EMIT", "b_emit", "a label"),
        )
        self.run_all.write_text(f"NATIVE_PARITY_TESTS = {pairs!r}\n")
        self.build = self.root / "build"
        self.build.mkdir()
        # The monitor's folder, which holds both binaries
        self.mine = self.root / "monitor"
        self.mine.mkdir()
        for name in ("a_emit", "b_emit"):
            self.program(self.mine / name)

    @staticmethod
    def program(path: pathlib.Path, mode: int = 0o755) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(mode)

    def test_a_link_at_the_build_root_is_refused(self):
        link = self.root / "core"
        link.symlink_to(self.mine)
        r = run(self.run_all, link)
        self.assertEqual((r.returncode, r.stdout), (1, ""))
        self.assertIn(f"native_env: {link}: a link or no folder", r.stderr)

    def test_a_link_in_the_build_is_never_followed(self):
        (self.build / "bin").symlink_to(self.mine)  # to a folder that holds a binary
        (self.build / "a_emit").symlink_to(self.mine / "a_emit")  # to a binary
        r = run(self.run_all, self.build)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))
        self.program(self.build / "lib" / "b_emit")
        r = run(self.run_all, self.build)
        self.assertEqual(r.stdout, f"export ROCKE_B_EMIT={self.build}/lib/b_emit\n")

    def test_a_build_its_owner_does_not_own_is_refused(self):
        self.program(self.build / "a_emit")
        other = next(p.pw_name for p in pwd.getpwall() if p.pw_uid != os.geteuid())
        r = run(self.run_all, self.build, other)
        self.assertEqual((r.returncode, r.stdout), (1, ""))
        self.assertIn(f"{self.build}: not {other}'s build", r.stderr)
        me = pwd.getpwuid(os.geteuid()).pw_name
        r = run(self.run_all, self.build, me)
        self.assertEqual(r.stdout, f"export ROCKE_A_EMIT={self.build}/a_emit\n")
        r = run(self.run_all, self.build, "no-such-user-here")
        self.assertEqual((r.returncode, r.stdout), (1, ""))
        self.assertIn("no user no-such-user-here", r.stderr)

    def test_a_binary_is_a_regular_file_with_an_x_bit(self):
        self.program(self.build / "a_emit", 0o644)
        os.mkfifo(self.build / "b_emit")
        (self.build / "b_emit").chmod(0o755)  # x bits, on no regular file
        self.assertEqual(run(self.run_all, self.build).stdout, "")
        (self.build / "a_emit").chmod(0o654)  # an x bit, its group's
        self.assertEqual(
            run(self.run_all, self.build).stdout,
            f"export ROCKE_A_EMIT={self.build}/a_emit\n",
        )

    def test_the_walk_goes_no_deeper_than_depth_max(self):
        cap = pr_results.DEPTH_MAX
        self.program(self.build.joinpath(*["d"] * cap) / "a_emit")
        self.program(self.build.joinpath(*["d"] * (cap + 1)) / "b_emit")
        r = run(self.run_all, self.build)
        self.assertEqual(
            r.stdout,
            f"export ROCKE_A_EMIT={self.build.joinpath(*['d'] * cap)}/a_emit\n",
        )


class TestTheChecksJobsLine(unittest.TestCase):
    def test_the_build_is_the_slots_runners(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "platform" / "tests").mkdir(parents=True)
            pairs = (("ROCKE_A_EMIT", "a_emit", "a label"),)
            (root / "platform" / "tests" / "run_all.py").write_text(
                f"NATIVE_PARITY_TESTS = {pairs!r}\n"
            )
            TestTheBuildIsPrCodes.program(root / "build" / "wt7" / "core" / "a_emit")

            def exported(slot: str) -> str:
                env = {
                    "PATH": os.environ["PATH"],
                    "M": str(ROOT),
                    "P": str(root / "platform"),
                    "SB": str(root / "build"),
                    "WTN": "wt7",
                    "SLOT": slot,
                }
                r = subprocess.run(
                    ["bash", "-c", f'{review.NATIVE_ENV}; echo "${{ROCKE_A_EMIT:-}}"'],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return r.stdout.strip()

            self.assertEqual(exported(""), f"{root}/build/wt7/core/a_emit")
            other = next(p.pw_name for p in pwd.getpwall() if p.pw_uid != os.geteuid())
            self.assertEqual(exported(other), "")  # the runner's build only

    def test_every_section_evaluates_the_one_line(self):
        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
            "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
        }
        files = [
            "rk/platform/python/rocke/core/foo.py",
            "rk/platform/tests/test_foo.py",
        ]
        plan = review.plan_checks(cfg, {"n": 7}, files, {})
        plan["_dirs"] = {}
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", plan, None)
        lines = [ln.strip() for ln in script.splitlines() if "native_env.py" in ln]
        self.assertEqual(
            len(lines), 3, lines
        )  # the PR's tests, the importers and develop's tests
        self.assertEqual(set(lines), {review.NATIVE_ENV})


if __name__ == "__main__":
    unittest.main()

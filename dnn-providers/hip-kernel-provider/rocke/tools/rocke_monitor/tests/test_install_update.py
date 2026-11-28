# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The install paths that must hold on a real deployment: worker tools copied writable and kept current (rk
provision, `rk worker-tools`, the doctor), the build's job count inside a CPU- and memory-limited cgroup, the installed
agentic files kept current, the doctor's RAM disk and Python development checks, and addresses shown without their
credentials."""

import contextlib
import io
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))

import rkmon  # noqa: E402
import steps  # noqa: E402
from rkmon_lib import (
    agentic,
    config,
    container_layout,
    deploy,
    doctor,
    paths,
    prcode,
)  # noqa: E402


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = pathlib.Path(self.d.name)


class TestWorkerTools(Tmp):
    """The image's code is read-only (0444); the copies the jobs run must be writable and current."""

    def setUp(self):
        super().setUp()
        self.pkg = self.root / "image"
        (self.pkg / "worker").mkdir(parents=True)
        self.write("mkwt.py", "v1\n", 0o555)
        self.write("steps.py", "s1\n", 0o444)
        self.cfg = {"paths": {"working_dir": str(self.root / "work")}}
        self.dest = pathlib.Path(paths.get(self.cfg, "worker_tools"))

    def write(self, name, text, mode):
        p = self.pkg / "worker" / name
        if p.exists():
            p.chmod(0o644)
        p.write_text(text)
        p.chmod(mode)

    def mode(self, name):
        return stat.S_IMODE((self.dest / name).stat().st_mode)

    def test_copies_are_writable_and_a_new_version_replaces_them(self):
        deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual((self.mode("mkwt.py"), self.mode("steps.py")), (0o755, 0o644))
        self.write("mkwt.py", "v2\n", 0o555)  # a new image's code, read-only again
        self.assertEqual(
            deploy.stale_worker_tools(self.cfg, self.pkg), [("mkwt.py", "differs")]
        )
        lines = deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual((self.dest / "mkwt.py").read_text(), "v2\n")
        self.assertFalse(
            any(x.startswith("back up") for x in lines), lines
        )  # as installed: no backup
        self.assertEqual(list(self.dest.glob("*.bak-*")), [])
        self.assertEqual(deploy.stale_worker_tools(self.cfg, self.pkg), [])

    def test_a_read_only_copy_from_an_earlier_install_is_replaced(self):
        self.dest.mkdir(parents=True)
        (self.dest / "mkwt.py").write_text("old\n")
        (self.dest / "mkwt.py").chmod(0o444)  # what copy2 made of the image's file
        lines = deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual((self.dest / "mkwt.py").read_text(), "v1\n")
        self.assertEqual(self.mode("mkwt.py"), 0o755)
        # It is not in the manifest (an install from before it): backed up once, never twice.
        self.assertEqual(sum(x.startswith("back up") for x in lines), 1)
        (self.dest / "mkwt.py").write_text("old\n")  # the same old content again
        lines = deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual(sum(x.startswith("back up") for x in lines), 0)
        self.assertEqual(len(list(self.dest.glob("mkwt.py.bak-*"))), 1)

    def test_the_pr_code_users_values_go_beside_the_tools(self):
        """The installed pr_user.py takes the PR runners' group, a slot's home and the package's rk from beside it and
        runs no file it finds (tests/test_container_layout.py), so the install writes them first; `rk worker-tools`
        and the doctor find them missing or not this code's, and an install writes them again.
        """
        values = self.dest / prcode.VALUES_FILE
        lines = deploy.install_worker_tools(self.cfg, self.pkg, dry=True)
        self.assertEqual((lines[0], self.dest.exists()), (f"write {values}", False))
        lines = deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual(lines[0], f"write {values}")
        self.assertEqual(
            json.loads(values.read_text()),
            {
                "group": container_layout.PR_GROUP,
                "home": container_layout.SLOT_HOME,
                "rk": str(self.pkg / "rk"),
            },
        )
        self.assertEqual(self.mode(prcode.VALUES_FILE), 0o644)
        self.assertEqual(deploy.stale_worker_tools(self.cfg, self.pkg), [])
        self.assertNotIn(
            f"write {values}", deploy.install_worker_tools(self.cfg, self.pkg)
        )
        values.write_text(
            paths.worker_module("pr_user").values_text("other", "home", "/x/rk")
        )
        self.assertEqual(
            deploy.stale_worker_tools(self.cfg, self.pkg),
            [(prcode.VALUES_FILE, "differs")],
        )
        values.unlink()
        self.assertEqual(
            deploy.stale_worker_tools(self.cfg, self.pkg),
            [(prcode.VALUES_FILE, "missing")],
        )
        deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual(deploy.stale_worker_tools(self.cfg, self.pkg), [])

    def test_a_locally_edited_tool_is_backed_up_first(self):
        deploy.install_worker_tools(self.cfg, self.pkg)
        (self.dest / "steps.py").write_text("my edit\n")
        lines = deploy.install_worker_tools(self.cfg, self.pkg)
        self.assertEqual(sum(x.startswith("back up") for x in lines), 1)
        self.assertEqual(
            [b.read_text() for b in self.dest.glob("steps.py.bak-*")], ["my edit\n"]
        )

    def test_rk_worker_tools_and_the_doctor_compare_contents(self):
        deploy.install_worker_tools(self.cfg)  # the running code's worker/
        (self.dest / "mkwt.py").write_text("stale\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.cmd_worker_tools(mock.Mock(install=False), self.cfg)
        self.assertEqual(code, 1)
        self.assertIn("differs  mkwt.py", out.getvalue())
        with mock.patch.object(paths, "resolve", wraps=paths.resolve):
            checks = [
                c
                for c in doctor.configuration_checks(self.cfg, "x")
                if c.name == "worker tools"
            ]
        self.assertEqual(
            (checks[0].status, str(checks[0].fix)),
            ("WARN", "rkmon rk worker-tools --install"),
        )
        self.assertIn("not the running code's: mkwt.py", checks[0].found)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                rkmon.cmd_worker_tools(mock.Mock(install=True), self.cfg), 0
            )
        checks = [
            c
            for c in doctor.configuration_checks(self.cfg, "x")
            if c.name == "worker tools"
        ]
        self.assertEqual(checks[0].status, "PASS")


class TestBuildJobs(Tmp):
    """A container with --cpus 4 --memory 8g on a 32-CPU host builds with 4 jobs at most, fewer when memory is short."""

    def cg(self, files: dict) -> str:
        d = pathlib.Path(
            tempfile.mkdtemp(dir=self.root)
        )  # a cgroup folder of its own each time
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text)
        d.mkdir(exist_ok=True)
        return str(d)

    def jobs(self, files, cpus=32, meminfo="MemAvailable:   64000000 kB\n"):
        (self.root / "meminfo").write_text(meminfo)
        with mock.patch.dict(os.environ, {"RKMON_JOBS": ""}):
            return steps.build_jobs(
                self.cg(files), str(self.root / "meminfo"), cpus=cpus
            )

    def test_the_cgroup_quota_and_memory_bound_the_jobs(self):
        self.assertEqual(self.jobs({"cpu.max": "400000 100000\n"}), 4)  # v2 quota
        self.assertEqual(
            self.jobs(
                {"cpu/cpu.cfs_quota_us": "250000", "cpu/cpu.cfs_period_us": "100000"}
            ),
            2,
        )  # v1
        self.assertEqual(
            self.jobs({"cpu.max": "max 100000\n"}), 32
        )  # no quota: the affinity
        gb = 2**30
        mem = {
            "cpu.max": "max 100000",
            "memory.max": str(8 * gb),
            "memory.current": str(2 * gb),
        }
        self.assertEqual(self.jobs(mem), 4)  # 6 GB free / 1.5 GB
        self.assertEqual(
            self.jobs(dict(mem, **{"memory.current": str(8 * gb)})), 1
        )  # never 0
        self.assertEqual(
            self.jobs({}, meminfo="MemAvailable:   3145728 kB\n"), 2
        )  # the host's 3 GB available
        with mock.patch.dict(os.environ, {"RKMON_JOBS": "7"}):
            self.assertEqual(steps.build_jobs(self.cg({}), "/nonexistent"), 7)

    def test_the_job_scripts_ask_steps_py_for_the_count(self):
        """The ctest of the review's checks and of the sanity run takes its -j from steps.py when the job runs."""
        from rkmon_lib import guard, review

        cfg = {
            "rocke_path": "rk",
            "repo": "ROCm/r",
            "paths": {"working_dir": "/nonexistent", "repo": "/nonexistent"},
        }
        plan = {
            "families": [],
            "tests": [],
            "build": True,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        scripts = {
            "checks": review.checks_script(cfg, 1, "a" * 40, "first-round", plan, None),
            "sanity": guard.sanity_script(
                cfg, "a" * 40, "/x/wtsanity", "/x/out", ["gemm"], True
            ),
        }
        for name, s in scripts.items():
            self.assertNotIn("-j16", s, name)
            m = re.search(r"ctest -j(\$\(.*?\|\| echo 1\))", s)
            self.assertIsNotNone(m, name)
            for env, want in (
                ({"RKMON_JOBS": "5"}, "5"),
                ({"PATH": "/nonexistent"}, "1"),
            ):
                p = subprocess.run(
                    ["/bin/bash", "-c", f"echo {m.group(1)}"],
                    capture_output=True,
                    text=True,
                    # $M the package's code, $T no installed worker tools: shell_vars sets both in a real job
                    env=dict(os.environ, M=str(ROOT), T="/nonexistent", **env),
                )
                self.assertEqual(p.stdout.strip(), want, (name, env))


class TestSafeUrl(unittest.TestCase):
    """An address is shown without a user, password, query or fragment, whatever its shape."""

    def test_shapes(self):
        self.assertEqual(
            config.safe_url("https://u:p@h.example.invalid:8443/v1?key=K#f"),
            "https://h.example.invalid:8443/v1",
        )
        self.assertEqual(
            config.safe_url("h.example.invalid:11434/api?key=K"),
            "h.example.invalid:11434/api",
        )  # no scheme
        self.assertEqual(
            config.safe_url(
                "http://h.example.invalid:" + "abc/v1"
            ),  # a port that is no number
            "http://h.example.invalid/v1",
        )  # bad port
        self.assertEqual(config.safe_url("?key=K"), "(not an address)")
        self.assertEqual(config.safe_url(""), "")

    def test_the_doctor_shows_the_reviewer_without_its_query(self):
        w = {
            "url": "http://127.0.0.1:11434/?key=SECRET",
            "model": "m",
            "backend": "ollama",
        }
        for presence in (
            {"answers": False, "error": "refused"},
            {"answers": True, "model_found": False},
        ):
            with mock.patch.object(
                doctor, "worker_running", lambda cfg: False
            ), mock.patch.object(doctor, "_presence", lambda rc, w: presence):
                c = doctor.model_check({"worker": w}, w)
            self.assertEqual(c.status, "FAIL")
            self.assertIn("127.0.0.1:11434", c.found)
            self.assertNotIn("SECRET", c.found)


class TestDoctorBuildChecks(unittest.TestCase):
    """Without Python's development headers CMake finds no Python3 and no native build ever runs."""

    def test_python_dev(self):
        ok = doctor.python_dev_check(
            "", run_=lambda cmd: (0, "/usr/include/python3.14 True\n")
        )
        self.assertEqual(ok.status, "PASS")
        bad = doctor.python_dev_check(
            "", run_=lambda cmd: (0, "/usr/include/python3.14 False\n")
        )
        self.assertEqual(bad.status, "FAIL")
        # the headers are the image's: the fix is an image of this package
        self.assertEqual(str(bad.fix), str(doctor.image_fix()))
        self.assertIn("it blocks the native build", bad.found)


class TestAgenticRefresh(Tmp):
    """Installed skill and agent files are refreshed; a file edited there is backed up once."""

    def test_install_keeps_them_current(self):
        rendered = self.root / "rendered"
        f = rendered / "skills" / agentic.SKILL / "SKILL.md"
        f.parent.mkdir(parents=True)
        f.write_text("v1\n")
        ws = self.root / "ws"
        agentic.install(rendered, ws)
        f.write_text("v2\n")
        lines = agentic.install(rendered, ws)
        dest = ws / ".claude" / "skills" / agentic.SKILL / "SKILL.md"
        self.assertEqual(dest.read_text(), "v2\n")
        self.assertFalse(
            any(x.startswith("back up") for x in lines), lines
        )  # as installed: no backup
        dest.write_text("edited\n")
        f.write_text("v3\n")
        lines = agentic.install(rendered, ws)
        self.assertEqual(sum(x.startswith("back up") for x in lines), 1)
        dest.write_text("edited\n")
        lines = agentic.install(rendered, ws)
        self.assertEqual(
            sum(x.startswith("back up") for x in lines), 0
        )  # the same edit: kept once already


if __name__ == "__main__":
    unittest.main()

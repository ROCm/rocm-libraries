# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The toolchain the checks run with is the live one, pinned: the image's ROCm release and its clang (one pin,
rkmon_lib/build_plan.py's, which the build gives the Dockerfile and the build template suggests), the venvs from
their locks (the image and rk provision alike), and rk doctor's checks of hipcc with the HIP headers and of a checks
venv without torch.
"""

import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import (
    build_plan,
    deploy,
    doctor,
    file_types,
    settings_file,
)  # noqa: E402
from rkmon_lib.settings_file import FileType  # noqa: E402
import test_build_plan as plans  # noqa: E402  (its build settings documents; its own tests run in their module)

DOCKER = ROOT / "docker"


class TestRocmPin(unittest.TestCase):
    def test_one_pin_everywhere(self):
        """The release and the clang the gates are tuned on, and the GPU targets ROCm's libraries are installed for and
        hipcc is checked with: rkmon_lib/build_plan.py's alone. The build gives each to the Dockerfile as a build
        argument (no ARG there has a default, so a build that is not given one stops), the build template suggests the
        release, by value and in its description, and the package's defaults name no clang: rk doctor compares with
        build_plan's commit."""
        dockerfile = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"(?m)^ARG \w+=.*$", dockerfile), [])
        for value in (
            build_plan.ROCM_RELEASE,
            build_plan.ROCM_CLANG_COMMIT,
            *build_plan.ROCM_LIBRARY_TARGETS,
            *build_plan.HIP_CHECK_TARGETS,
        ):
            self.assertNotIn(value, dockerfile)
        self.assertIn(
            '"${ROCM_CLANG_COMMIT?not given (rkmon build gives it)}"', dockerfile
        )  # the installer checks it
        args = plans.make_plan().args
        self.assertEqual(
            (
                args["ROCM_CLANG_COMMIT"],
                args["ROCM_LIBRARY_TARGETS"],
                args["HIP_CHECK_TARGETS"],
            ),
            (
                build_plan.ROCM_CLANG_COMMIT,
                ",".join(build_plan.ROCM_LIBRARY_TARGETS),
                ",".join(build_plan.HIP_CHECK_TARGETS),
            ),
        )
        defaults = (ROOT / "config.defaults.json").read_text(encoding="utf-8")
        self.assertNotIn(build_plan.ROCM_CLANG_COMMIT, defaults)
        version = file_types.BUILD.load()["properties"]["rocm"]["properties"]["version"]
        self.assertEqual(version["x-template"], build_plan.ROCM_RELEASE)
        self.assertIn(
            f'Keep "{build_plan.ROCM_RELEASE}"',
            settings_file.description(version, "rocm.version"),
        )

    def test_the_build_template_takes_the_release_from_its_one_home(self):
        """The build schema's file names no ROCm release: the build template's value and its description's take
        build_plan.ROCM_RELEASE, whatever it is (file_types.build_values)."""
        text = (ROOT / "schema" / "build_settings.schema.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(build_plan.ROCM_RELEASE, text)
        build = file_types.BUILD
        with mock.patch.object(build_plan, "ROCM_RELEASE", "9.8.7"):
            fresh = FileType(
                build.kind,
                build.version,
                build.schema,
                coherence=build.coherence,
                upgrades=build.upgrades,
                template_command=build.template_command,
                template_values=build.template_values,
            )
            written = settings_file.template(fresh)
        self.assertIn('Keep "9.8.7" unless your team needs another', written)
        self.assertRegex(written, r'(?m)^  version: "9\.8\.7"$')

    def test_the_installer_takes_a_release_and_checks_the_clang(self):
        """The installer's fourth argument is the pinned clang commit: a clang that names another stops the build
        (exit 4); the pinned one, or no pin, goes on."""
        install = (DOCKER / "install_rocm.sh").read_text(encoding="utf-8")
        p = subprocess.run(["sh", "-n", str(DOCKER / "install_rocm.sh")])
        self.assertEqual(p.returncode, 0)
        args = install[
            install.index("version=${1:-}") : install.index('case "$version" in')
        ]
        check = install[
            install.index('llvm_bin="$core/lib/llvm/bin"') : install.index(
                "# hipcc and the HIP headers"
            )
        ]
        with tempfile.TemporaryDirectory() as d:
            core = pathlib.Path(d) / "core-10.1"
            (core / "lib/llvm/bin").mkdir(parents=True)
            clang = core / "lib/llvm/bin/clang"
            clang.write_text(
                "#!/bin/sh\necho 'AMD clang version 22.0.0git "
                "(https://github.com/ROCm/llvm-project roc-10.1.0 abc123 aaaa1111)'\n"
            )
            clang.chmod(0o755)

            def run(*argv):
                return subprocess.run(
                    [
                        "sh",
                        "-c",
                        f"set -eu\n{args}core={core}\nname=rocm-installer-10.1.0-5.run\n"
                        f"{check}echo checked\n",
                        "install_rocm.sh",
                        *argv,
                    ],
                    capture_output=True,
                    text=True,
                )

            p = run("10.1.0", "", "gfx942", "bbbb2222")
            self.assertEqual(p.returncode, 4, p.stdout + p.stderr)
            self.assertIn("not the pinned compiler (bbbb2222)", p.stderr)
            self.assertNotIn("checked", p.stdout)
            for pin in ("aaaa1111", ""):
                p = run("10.1.0", "", "gfx942", pin)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertIn("checked", p.stdout)

    def test_the_installer_brings_the_hip_headers_and_hipcc_compiles_them(self):
        """The installer runs with the core-dev component (the HIP headers: its default, core, has none), and the
        build stops (exit 3) unless hipcc compiles <hip/hip_runtime.h> for each GPU target of its third argument, which
        the build gives from build_plan.HIP_CHECK_TARGETS, the targets rk doctor's check compiles for. A blank or
        malformed list stops it before any download (exit 2)."""
        install = (DOCKER / "install_rocm.sh").read_text(encoding="utf-8")
        self.assertIn("compo=core,core-dev\n", install)
        self.assertIn(
            'args="deps=install gfx=$gfx rocm compo=$compo gpu-access=user assumeyes"',
            install,
        )
        self.assertEqual(build_plan.HIP_CHECK_TARGETS, ("gfx942", "gfx950"))
        self.assertEqual(build_plan.ROCM_LIBRARY_TARGETS, ("gfx942",))
        targets = plans.make_plan().args["HIP_CHECK_TARGETS"]
        args = install[install.index("version=${1:-}") : install.index("url=$(")]
        snippet = install[
            install.index('hipcc="$core/bin/hipcc"') : install.index(
                "mkdir -p /etc/rkmon"
            )
        ]
        fake_hipcc = (
            "#!/bin/sh\n"
            'core=$(dirname "$(dirname "$0")")\n'
            '[ -f "$core/include/hip/hip_runtime.h" ] || '
            "{ echo \"probe.hip:1:10: fatal error: 'hip/hip_runtime.h' file not found\" >&2; exit 1; }\n"
            'while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift ;; '
            '--offload-arch=*) echo "$1" >> "$core/archs" ;; esac; shift; done\n'
            'echo obj > "$out"\n'
        )

        def run(core, hip_targets):
            return subprocess.run(
                [
                    "sh",
                    "-c",
                    f"set -eu\n{args}core={core}\nwork={core / 'work'}\ncompo=core,core-dev\n"
                    + snippet,
                    "install_rocm.sh",
                    "10.1.0",
                    "",
                    hip_targets,
                ],
                capture_output=True,
                text=True,
            )

        with tempfile.TemporaryDirectory() as d:
            for case, headers in (("with core-dev", True), ("core alone", False)):
                with self.subTest(case=case):
                    core = pathlib.Path(d) / case.replace(" ", "_")
                    (core / "bin").mkdir(parents=True)
                    (core / "work").mkdir()
                    (core / "bin" / "hipcc").write_text(fake_hipcc)
                    (core / "bin" / "hipcc").chmod(0o755)
                    if headers:
                        (core / "include" / "hip").mkdir(parents=True)
                        (core / "include" / "hip" / "hip_runtime.h").write_text("")
                    p = run(core, targets)
                    out = p.stdout + p.stderr
                    if headers:
                        self.assertEqual(p.returncode, 0, out)
                        self.assertIn(
                            "compiles <hip/hip_runtime.h> for gfx942 gfx950", out
                        )
                        self.assertEqual(
                            (core / "archs").read_text().split(),
                            ["--offload-arch=gfx942", "--offload-arch=gfx950"],
                        )
                    else:
                        self.assertEqual(p.returncode, 3, out)
                        self.assertIn(
                            "cannot compile <hip/hip_runtime.h> for gfx942", out
                        )
                        self.assertIn("'hip/hip_runtime.h' file not found", out)
            for given in ("", "gfx942;gfx950"):
                with self.subTest(given=given):
                    p = run(pathlib.Path(d) / "with_core-dev", given)
                    self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                    self.assertIn("is no list of GPU targets", p.stderr)

    def test_the_core_taken_is_the_one_the_installer_added(self):
        """The release's core folder is core-<major.minor> of its version, else the newest one its installer added,
        never merely the newest there: an image that already holds a release (the image test's rocm stage runs its
        fake installs in one) keeps its own apart."""
        install = (DOCKER / "install_rocm.sh").read_text(encoding="utf-8")
        start = install.index("before=$(ls -d /opt/rocm/core-*")
        snippet = install[start : install.index("reported=$(")]
        run_line = '(cd "$work" && sh "$HERE/masked.sh" package_mirror -- bash ./installer.run $args)'
        self.assertIn(run_line, snippet)
        cases = (
            (
                "another release is there",
                ["core-10.1"],
                "core-9.10",
                "9.10.0-2",
                "core-9.10",
            ),
            ("the first install", [], "core-10.1", "10.1.0-5", "core-10.1"),
            (
                "again over itself",
                ["core-9.10", "core-10.1"],
                "core-9.10",
                "9.10.0-2",
                "core-9.10",
            ),
            (
                "a folder named otherwise",
                ["core-10.1"],
                "core-11",
                "11.0.0-1",
                "core-11",
            ),
        )
        with tempfile.TemporaryDirectory() as d:
            for i, (case, there, adds, version, want) in enumerate(cases):
                with self.subTest(case=case):
                    top = pathlib.Path(d) / str(i)
                    top.mkdir()
                    for c in there:
                        (top / c).mkdir()
                    script = snippet.replace(
                        run_line, f"mkdir -p {top}/{adds}"
                    ).replace("/opt/rocm", str(top))
                    p = subprocess.run(
                        [
                            "sh",
                            "-c",
                            "set -eu"
                            + chr(10)
                            + f"version={version}"
                            + chr(10)
                            + f"release={version.split('-')[0]}"
                            + chr(10)
                            + script
                            + 'echo "$core"',
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(p.stdout.strip(), str(top / want), p.stderr)


class TestLocks(unittest.TestCase):
    def test_the_image_and_rk_provision_install_the_locks(self):
        dockerfile = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("requirements/checks.lock.txt", dockerfile)
        self.assertIn("requirements/gen.lock.txt", dockerfile)
        self.assertNotRegex(
            dockerfile, r"-r /opt/rkmon/build/requirements/(checks|gen)\.txt"
        )
        self.assertIn('! /opt/rkmon/venv/bin/python -c "import torch"', dockerfile)
        with tempfile.TemporaryDirectory() as d:
            cfg = {"paths": {"working_dir": d}}
            for which in ("checks", "gen"):
                pip = [
                    c
                    for c in deploy.venv_commands(cfg, which)
                    if c[1:4] == ["-m", "pip", "install"]
                ]
                self.assertTrue(pip[0][-1].endswith(f"requirements/{which}.lock.txt"))

    def test_each_lock_pins_every_line_and_carries_no_torch(self):
        for name, must in (
            (
                "checks.lock.txt",
                (
                    "pytest==9.1.1",
                    "numpy==2.5.4",
                    "ml_dtypes==0.6.0",
                    "pybind11==3.1.0",
                    "cmake==4.4.4",
                ),
            ),
            ("gen.lock.txt", ("Jinja2==3.1.6", "PyYAML==6.0.3")),
        ):
            with self.subTest(lock=name):
                lines = [
                    ln.strip()
                    for ln in (ROOT / "requirements" / name).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")
                ]
                self.assertTrue(
                    all(re.fullmatch(r"[A-Za-z0-9_.-]+==[\w.]+", ln) for ln in lines),
                    lines,
                )
                for m in must:
                    self.assertIn(m, lines)
                self.assertFalse([ln for ln in lines if ln.lower().startswith("torch")])


class TestDoctorRocm(unittest.TestCase):
    def test_hipcc_and_the_hip_headers(self):
        seen = []

        def run(cmd, timeout=20):
            seen.append(cmd)
            src = pathlib.Path(cmd[-1]).read_text()
            self.assertIn("#include <hip/hip_runtime.h>", src)
            self.assertIn("__global__", src)
            pathlib.Path(cmd[cmd.index("-o") + 1]).write_text("obj")
            return 0, ""

        ok = doctor.hip_check("", run_=run, hipcc="/opt/rocm/core-x/bin/hipcc")
        self.assertEqual(ok.status, "PASS")
        self.assertIn("gfx942, gfx950", ok.found)
        # One compile (-c, code generation) for each of the review's GPU targets.
        self.assertEqual(
            [c[:3] for c in seen],
            [
                ["/opt/rocm/core-x/bin/hipcc", "--offload-arch=gfx942", "-c"],
                ["/opt/rocm/core-x/bin/hipcc", "--offload-arch=gfx950", "-c"],
            ],
        )
        # A hipcc that exits 0 but writes no object compiled nothing.
        empty = doctor.hip_check(
            "", run_=lambda cmd, timeout=20: (0, ""), hipcc="/x/hipcc"
        )
        self.assertEqual(empty.status, "WARN")
        self.assertIn("for gfx942", empty.found)

        def only_942(cmd, timeout=20):
            if "--offload-arch=gfx942" in cmd:
                pathlib.Path(cmd[cmd.index("-o") + 1]).write_text("obj")
                return 0, ""
            return 1, "error: cannot find ROCm device library for gfx950"

        one = doctor.hip_check("", run_=only_942, hipcc="/x/hipcc")
        self.assertEqual(one.status, "WARN")
        self.assertIn("for gfx950 (exit 1)", one.found)
        self.assertIn("docker/install_rocm.sh installs them", str(one.fix))
        bad = doctor.hip_check(
            "",
            run_=lambda cmd, timeout=20: (
                1,
                "fatal error: 'hip/hip_runtime.h' file not found",
            ),
            hipcc="/x/hipcc",
        )
        self.assertEqual(bad.status, "WARN")
        self.assertIn("hip_runtime.h", bad.found)
        none = doctor.hip_check("", hipcc="")
        self.assertEqual(none.status, "WARN")
        self.assertIn("they skip", none.found)
        # Without hipcc the fix is an image of this package, which carries ROCm, wherever the doctor runs.
        self.assertEqual(str(none.fix), str(doctor.image_fix()))
        from unittest import mock

        with mock.patch.dict(doctor.os.environ, {"RKMON_CONTAINER": "1"}):
            boxed = doctor.hip_check("", hipcc="")
        self.assertIn("run an image built by this package", str(boxed.fix))
        # In the container, without the HIP headers: the image built again, named to the deployment by image.tag, then
        # the confirm.
        with mock.patch.dict(doctor.os.environ, {"RKMON_CONTAINER": "1"}):
            rebuilt = doctor.hip_check(
                "",
                run_=lambda cmd, timeout=20: (
                    1,
                    "fatal error: 'hip/hip_runtime.h' file not found",
                ),
                hipcc="/x/hipcc",
            )
            wrapper = doctor.wrapper_name()
        self.assertEqual(
            str(rebuilt.fix),
            "build the image again (QUICK_BUILD_GUIDE.md; docker/install_rocm.sh installs them), then set image.tag "
            f"in your deploy settings file to its tag, then {wrapper} confirm",
        )

    def test_hipcc_beside_the_llvm_tools(self):
        with tempfile.TemporaryDirectory() as d:
            core = pathlib.Path(d) / "core-10.1"
            (core / "lib" / "llvm" / "bin").mkdir(parents=True)
            (core / "bin").mkdir()
            (core / "bin" / "hipcc").write_text("#!/bin/sh\n")
            (core / "bin" / "hipcc").chmod(0o755)
            from unittest import mock

            with mock.patch.object(doctor.shutil, "which", lambda name: None):
                self.assertEqual(
                    doctor.hipcc_path(str(core / "lib" / "llvm" / "bin")),
                    str(core / "bin" / "hipcc"),
                )
                self.assertEqual(doctor.hipcc_path("/x"), "")

    def test_the_checks_venv_has_no_torch(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "bin").mkdir()
            (pathlib.Path(d) / "bin" / "python").write_text("")
            self.assertEqual(
                doctor.venv_torch_check(d, run_=lambda cmd: (1, "")).status, "PASS"
            )
            bad = doctor.venv_torch_check(d, run_=lambda cmd: (0, ""))
            self.assertEqual(bad.status, "FAIL")
            # the checks venv is the image's: its fix is an image of this package
            self.assertEqual(str(bad.fix), str(doctor.image_fix()))
        self.assertEqual(doctor.venv_torch_check("/nonexistent").status, "SKIP")


if __name__ == "__main__":
    unittest.main()

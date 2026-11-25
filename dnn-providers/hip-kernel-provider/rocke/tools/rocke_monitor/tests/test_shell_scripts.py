# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's scripts are Python (rocke's AGENTS.md: cross-platform, no .sh), but for a short list of shell files,
each with its reason: rk picks the code a command runs before Python starts, and the container's entry point and its
build-time helpers run only inside the Linux image. Every shell file left parses (`sh -n` for the POSIX ones, `bash -n`
for the bash ones), and rk is POSIX sh."""

import pathlib
import shutil
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
POSIX = ("rk",)
# The only shell files the package keeps, and why.
EXEMPT = {
    "rk": "the monitor's command: picks the code a command runs (rkmon_lib/code_volume.py) before Python starts",
    "docker/entrypoint.sh": "the container's entry point, inside the Linux image only",
    "docker/install_rocm.sh": "a build step of the Linux image (docker/Dockerfile)",
    "docker/install_atlassian.sh": "a build step of the Linux image (docker/Dockerfile)",
    "docker/install_sdk.sh": "a build step of the Linux image (docker/Dockerfile)",
    "docker/pkgs.sh": "the Linux image's package-manager helper (docker/Dockerfile), before any Python exists",
    "docker/param.sh": "a build step's download location, its build secret (docker/Dockerfile), before any Python "
    "exists",
    "docker/fetch.sh": "the download of the Linux image's build steps (docker/install_rocm.sh, "
    "docker/install_atlassian.sh, docker/install_sdk.sh), its address on curl's standard input",
    "docker/masked.sh": "a build step's tool, each download location in its lines shown by its build secret's name "
    "(docker/Dockerfile), before any Python exists",
}
SKIP = {".git", "__pycache__", "state", "dist", ".pytest_cache"}


def package_files():
    for f in sorted(ROOT.rglob("*")):
        if f.is_file() and not SKIP.intersection(f.relative_to(ROOT).parts):
            yield f


def first_line(f: pathlib.Path) -> bytes:
    try:
        with f.open("rb") as fh:
            return fh.readline()
    except OSError:
        return b""


def shell_scripts():
    for f in package_files():
        first = first_line(f)
        if first.startswith(b"#!") and (b"/sh" in first or b"bash" in first):
            yield f, (b"bash" in first)


class TestShellScripts(unittest.TestCase):
    def test_every_script_parses(self):
        found = list(shell_scripts())
        self.assertGreaterEqual(len(found), len(EXEMPT))
        for f, is_bash in found:
            shell = shutil.which("bash" if is_bash else "sh")
            with self.subTest(script=str(f.relative_to(ROOT))):
                p = subprocess.run(
                    [shell, "-n", str(f)], capture_output=True, text=True
                )
                self.assertEqual(p.returncode, 0, p.stderr)

    def test_the_bare_machine_entry_points_are_posix_sh(self):
        for name in POSIX:
            with self.subTest(script=name):
                first = (ROOT / name).read_text(encoding="utf-8").splitlines()[0]
                self.assertEqual(first, "#!/bin/sh")
                dash = shutil.which("dash")
                if dash:
                    p = subprocess.run(
                        [dash, "-n", str(ROOT / name)], capture_output=True, text=True
                    )
                    self.assertEqual(p.returncode, 0, p.stderr)

    def test_the_only_shell_files_are_the_exempt_ones(self):
        """A shell file (by its #! line, or by a .sh name) outside the exemption list is a script to write in Python."""
        shell = {f.relative_to(ROOT).as_posix() for f, _bash in shell_scripts()}
        shell |= {
            f.relative_to(ROOT).as_posix() for f in package_files() if f.suffix == ".sh"
        }
        self.assertEqual(
            sorted(shell - set(EXEMPT)), [], "shell files outside the exemption list"
        )
        self.assertEqual(
            sorted(set(EXEMPT) - shell),
            [],
            "exempt files that are gone: drop them from EXEMPT",
        )


if __name__ == "__main__":
    unittest.main()

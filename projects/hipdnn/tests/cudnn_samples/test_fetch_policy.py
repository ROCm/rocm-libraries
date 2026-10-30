#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures this harness with -DALLOW_FETCH_DEPS=OFF and checks each fetch fails with E28.

Runs as a ctest case. Stub hipdnn_frontend and Catch2 packages stand in for the real ones, the
compiler is forced so no probe runs, and network access goes through a dead proxy, so a fetch
that slipped past the policy fails quickly instead of cloning.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
DEAD_PROXY = "http://127.0.0.1:9"

ARGS = argparse.Namespace()


class FetchPolicyTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.frontend = self.root / "hipdnn_frontend"
        self.frontend.mkdir()
        (self.frontend / "hipdnn_frontendConfig.cmake").write_text(
            "set(hipdnn_frontend_cudnn_compatibility_FOUND ON)\n"
        )
        self.catch2 = self.root / "catch2"
        self.catch2.mkdir()
        (self.catch2 / "Catch2Config.cmake").write_text("\n")
        (self.catch2 / "Catch2ConfigVersion.cmake").write_text(
            "set(PACKAGE_VERSION 3.13.0)\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
        )

    def tearDown(self):
        self._dir.cleanup()

    def configure(self, *extra: str) -> str:
        command = [
            ARGS.cmake,
            "-S",
            str(HARNESS),
            "-B",
            str(self.root / "build"),
            "-G",
            ARGS.generator,
            f"-DCMAKE_CXX_COMPILER={ARGS.cxx}",
            "-DCMAKE_CXX_COMPILER_FORCED=TRUE",
            "-DCMAKE_CXX_COMPILER_ID_RUN=TRUE",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dhipdnn_frontend_DIR={self.frontend}",
            "-DALLOW_FETCH_DEPS=OFF",
            *extra,
        ]
        if ARGS.make_program:
            command.append(f"-DCMAKE_MAKE_PROGRAM={ARGS.make_program}")
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env[name] = DEAD_PROXY
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=300
        )
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        return output

    def test_missing_catch2_is_e28(self):
        output = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_Catch2=TRUE")
        self.assertIn("E28 FETCH: Catch2", output)

    def test_missing_corpus_is_e28(self):
        output = self.configure(f"-DCatch2_DIR={self.catch2}")
        self.assertIn("E28 FETCH: no pinned corpus", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--make-program", default="")
    ARGS, rest = parser.parse_known_args(namespace=ARGS)
    unittest.main(argv=[sys.argv[0], *rest])

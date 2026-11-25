# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package folder the deploy path runs from, which holds no worker/: every module docker/rkmon loads by name imports
in the tests' deploy folder (tests/deploy_folder.py), which holds none either, so no module the deploy path imports
reads a worker module as it loads (rkmon_lib/paths.py worker_module).
"""

import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402


class TestDeployFolder(unittest.TestCase):
    def test_every_module_rkmon_loads_imports_with_no_worker_folder(self):
        """Every module docker/rkmon loads by name (its `lib`) imports in the tests' deploy folder, which holds no
        worker/, as the package folder the deploy path runs from holds none: no module the deploy path imports reads a
        worker module as it loads (rkmon_lib/paths.py worker_module)."""
        text = (ROOT / "docker" / "rkmon").read_text(encoding="utf-8")
        names = sorted(set(re.findall(r'\blib\("(\w+)"\)', text)))
        self.assertIn("control", names)
        code = (
            "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
            "[importlib.import_module('rkmon_lib.' + n) for n in sys.argv[2:]]"
        )
        with tempfile.TemporaryDirectory() as d:
            pkg = deploy_folder.copy(pathlib.Path(d) / "pkg")
            self.assertFalse((pkg / "worker").exists())
            p = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code, str(pkg), *names],
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check SDK initialization order and failures without requiring native libraries."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TestRuntimeInitialization(unittest.TestCase):
    def run_import(self, sdk_source, sdk_version="10.3.0a20261123"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "origami"
            package.mkdir()
            source = Path(__file__).resolve().parents[1] / "src" / "origami"
            if not source.is_dir():
                spec = importlib.util.find_spec("origami")
                if spec is None or spec.origin is None:
                    raise RuntimeError("Cannot locate the installed Origami package")
                source = Path(spec.origin).parent
            for name in ["__init__.py"]:
                shutil.copyfile(source / name, package / name)
            from unittest.mock import patch

            build_spec = importlib.util.spec_from_file_location(
                "build_metadata",
                Path(__file__).resolve().parents[1] / "build_metadata.py",
            )
            build_metadata = importlib.util.module_from_spec(build_spec)
            build_spec.loader.exec_module(build_metadata)
            with patch.dict(os.environ, {}, clear=True):
                if sdk_version is not None:
                    os.environ["ROCM_SDK_VERSION"] = sdk_version
                build_metadata.write_initializer(package / "_rocm_init.py")
            (package / "_pyorigami.py").write_text(
                'print("EXTENSION_REACHED", flush=True)\n'
                'raise ImportError("extension sentinel")\n'
            )
            (root / "rocm_sdk.py").write_text(sdk_source)
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("LD_PRELOAD", None)
            env["PYTHONPATH"] = str(root)
            # This setting must not cause unrelated library preloads.
            env["ROCM_SDK_PRELOAD_LIBRARIES"] = "unrelated_library"
            return subprocess.run(
                [sys.executable, "-c", "import origami"],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_sdk_initializes_before_extension(self):
        result = self.run_import(
            "import json\n"
            "def initialize_process(**kwargs):\n"
            "    print(json.dumps(kwargs), flush=True)\n"
        )
        self.assertNotEqual(result.returncode, 0)
        lines = result.stdout.splitlines()
        self.assertEqual(
            json.loads(lines[0]),
            {
                "preload_shortnames": ["amdhip64", "origami"],
                "env_override": False,
                "check_version": "10.3.0a20261123",
                "fail_on_version_mismatch": True,
            },
        )
        self.assertEqual(lines[1:], ["EXTENSION_REACHED"])
        self.assertIn("extension sentinel", result.stderr)

    def test_absent_sdk_uses_system_loader(self):
        result = self.run_import(
            'raise ModuleNotFoundError("SDK absent", name="rocm_sdk")\n',
            sdk_version=None,
        )
        self.assertEqual(result.stdout.splitlines(), ["EXTENSION_REACHED"])
        self.assertIn("extension sentinel", result.stderr)

    def test_sdk_build_requires_sdk(self):
        result = self.run_import(
            'raise ModuleNotFoundError("SDK absent", name="rocm_sdk")\n'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EXTENSION_REACHED", result.stdout)
        self.assertIn("SDK absent", result.stderr)

    def test_broken_sdk_import_is_not_ignored(self):
        result = self.run_import(
            'raise ModuleNotFoundError("SDK dependency absent", name="sdk_dependency")\n'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EXTENSION_REACHED", result.stdout)
        self.assertIn("SDK dependency absent", result.stderr)

    def test_sdk_runtime_failure_stops_extension_import(self):
        result = self.run_import(
            "def initialize_process(**kwargs):\n"
            '    raise FileNotFoundError("native Origami missing from SDK")\n'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EXTENSION_REACHED", result.stdout)
        self.assertIn("native Origami missing from SDK", result.stderr)


if __name__ == "__main__":
    unittest.main()

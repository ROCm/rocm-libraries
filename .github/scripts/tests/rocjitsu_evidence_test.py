# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EvidenceTest(unittest.TestCase):
    def test_xdist_retains_passing_subprocess_output_and_metadata_before_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test = root / "test_example.py"
            test.write_text(
                """
import shutil, subprocess, sys
from pathlib import Path
import pytest
def _call_helper_in_subprocess(module, func, config, output_dir, artifact_dir, tensile_args):
    script = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(exist_ok=True); (p/'ClientParameters.ini').write_text('init-seed=7'+chr(10)); print('helper verdict: PASSED')"
    subprocess.run([sys.executable, '-c', script, output_dir], check=True)
@pytest.mark.parametrize('name', ['one', 'two'])
def test_roundtrip(name, tmp_path):
    output = tmp_path / name
    _call_helper_in_subprocess('test_config_build', '_build', 'config', str(output), str(tmp_path), [])
    shutil.rmtree(output)
    _call_helper_in_subprocess('test_config_run', '_run', 'config', str(output), str(tmp_path), [])
    shutil.rmtree(output)
def test_skip():
    pytest.skip('unsupported mode')
"""
            )
            scripts = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test),
                    "-p",
                    "rocjitsu_evidence",
                    "-p",
                    "xdist.plugin",
                    "-n",
                    "2",
                    "--capture=fd",
                    "--rocjitsu-evidence-dir",
                    str(root / "evidence"),
                    "-k",
                    "not excluded",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                env=os.environ
                | dict(
                    PYTHONPATH=str(scripts),
                    PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            items = list((root / "evidence").glob("*/item.json"))
            self.assertEqual(len(items), 3)
            passing = []
            for path in items:
                item = json.loads(path.read_text())
                if "test_skip" in item["nodeid"]:
                    self.assertIn("unsupported mode", item["reports"]["call"]["detail"])
                    continue
                passing.append(item)
                self.assertEqual([p["name"] for p in item["phases"]], ["build", "run"])
                self.assertTrue(all(p["completed"] for p in item["phases"]))
                self.assertTrue(all(p["wall_seconds"] > 0 for p in item["phases"]))
                self.assertEqual(
                    (path.parent / "call-stdout.log")
                    .read_text()
                    .count("helper verdict: PASSED"),
                    2,
                )
                for phase in ("build", "run"):
                    self.assertEqual(
                        (path.parent / phase / "ClientParameters.ini").read_text(),
                        "init-seed=7\n",
                    )
            self.assertEqual(len(passing), 2)
            collections = list((root / "evidence").glob("collection-gw*.json"))
            self.assertEqual(len(collections), 2)
            self.assertTrue(
                all(
                    len(json.loads(p.read_text())["selected"]) == 3 for p in collections
                )
            )

    def test_rejects_capture_mode_that_drops_success_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "rocjitsu_evidence",
                    "-s",
                    "--rocjitsu-evidence-dir",
                    temporary,
                    "--collect-only",
                ],
                env=os.environ
                | dict(PYTHONPATH=str(scripts), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --capture=fd", result.stderr)


if __name__ == "__main__":
    unittest.main()

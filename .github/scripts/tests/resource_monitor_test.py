"""Tests for the Linux CI resource monitor."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "resource_monitor.py"
sys.path.insert(0, str(SCRIPT.parent))
import resource_monitor  # noqa: E402


class CgroupCountersTest(unittest.TestCase):
    def test_cgroup_v2_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpu.stat").write_text("nr_periods 10\nusage_usec 2500000\n")
            (root / "memory.current").write_text("10485760\n")
            counters = resource_monitor.CgroupCounters(root)
            self.assertEqual(counters.read(), (2.5, 10485760))

    def test_cgroup_v1_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpuacct").mkdir()
            (root / "memory").mkdir()
            (root / "cpuacct/cpuacct.usage").write_text("2500000000\n")
            (root / "memory/memory.usage_in_bytes").write_text("10485760\n")
            counters = resource_monitor.CgroupCounters(root)
            self.assertEqual(counters.read(), (2.5, 10485760))


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc is required")
class MonitorExecutionTest(unittest.TestCase):
    def test_records_ram_and_preserves_command_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resources.json"
            grandchild = (
                "import time; data=bytearray(16*1024*1024); "
                "time.sleep(.25); raise SystemExit(7)"
            )
            parent = (
                "import subprocess, sys; "
                f"raise SystemExit(subprocess.run([sys.executable, '-c', {grandchild!r}]).returncode)"
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--scope",
                "process",
                "--phase",
                "test",
                "--output",
                str(output),
                "--interval",
                "0.05",
                "--",
                sys.executable,
                "-c",
                parent,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            summary = json.loads(output.read_text())
            self.assertEqual(summary["exit_code"], 7)
            self.assertGreater(summary["peak_memory_bytes"], 8 * 1024 * 1024)
            self.assertGreater(summary["sample_count"], 2)
            self.assertTrue(output.with_suffix(".csv").is_file())


if __name__ == "__main__":
    unittest.main()

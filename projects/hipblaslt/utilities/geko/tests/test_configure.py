# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for configure.py CLI interface.

Usage:
    python3 -m pytest tests/test_configure.py \\
        --hipblaslt-path /path/to/hipblaslt \\
        --workload /path/to/workload.yaml \\
        [--hw gfx942]

    ``--hw`` sets ``configure.py --architecture`` (default: gfx950).
"""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    """Run configure.py against a real workload YAML and hipBLASLt repo.

    Usage:
        python3 -m pytest tests/test_configure.py \\
            --hipblaslt-path ~/rocm-libraries/projects/hipblaslt \\
            --workload workload.yaml
    """

    def test_full_cli_run(self, hipblaslt_path, workload_path, hw_arch, tmp_path):
        """Full configure CLI completes (uses conftest CLI options)."""
        if hipblaslt_path is None or workload_path is None:
            msg = "Skipped: requires --hipblaslt-path and --workload CLI options"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)

        hip = Path(hipblaslt_path)
        workload = Path(workload_path)
        if not hip.is_dir():
            msg = f"Skipped: hipBLASLt path not found: {hip}"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)
        if not workload.is_file():
            msg = f"Skipped: workload file not found: {workload}"
            warnings.warn(msg, stacklevel=1)
            pytest.skip(msg)

        workdir = tmp_path / "workdir"
        run_cwd = str(tmp_path.resolve())
        configure_py = str((ROOT / "scripts" / "configure.py").resolve())
        result = subprocess.run(
            [
                sys.executable,
                configure_py,
                str(workload.resolve()),
                "--hipblaslt",
                str(hip.resolve()),
                "--workdir",
                str(workdir.resolve()),
                "--architecture",
                hw_arch,
            ],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"configure.py exited with code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
        assert workdir.is_dir()
        generated = list(workdir.rglob("*"))
        assert len(generated) > 0, "Expected output files but workdir is empty"

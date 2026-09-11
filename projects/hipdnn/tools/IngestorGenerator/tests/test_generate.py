# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI subprocess tests for generate.py.

Exit-code contract (frozen in the phase-2 shared contract):
exit 0 success; exit 1 on ConfigError or a render failure; exit 2 from
argparse on a bad flag. --dry-run must not create the output dir.
--force is required to overwrite an existing non-empty output directory.
"""

import subprocess
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).parent.parent
SCALE_ADD_CONFIG = TOOL_ROOT / "configs" / "scale_add.yaml"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(TOOL_ROOT / "generate.py"), *args],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )


class TestExitCodes:
    def test_success_exits_zero(self, tmp_path):
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(tmp_path / "out")
        )
        assert result.returncode == 0, result.stderr

    def test_missing_config_file_exits_one(self, tmp_path):
        result = run_cli(
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_config_error_exits_one(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("engine:\n  name: unscoped\n")
        result = run_cli(
            "--config", str(bad_config), "--output-dir", str(tmp_path / "out")
        )
        assert result.returncode == 1
        assert "Config error" in result.stderr

    def test_bad_flag_exits_two(self, tmp_path):
        result = run_cli("--nonexistent-flag", "value")
        assert result.returncode == 2

    def test_missing_required_flag_exits_two(self):
        result = run_cli("--config", str(SCALE_ADD_CONFIG))
        assert result.returncode == 2


class TestDryRun:
    def test_dry_run_does_not_create_output_dir(self, tmp_path):
        output_dir = tmp_path / "out"
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        )
        assert result.returncode == 0, result.stderr
        assert not output_dir.exists()

    def test_dry_run_lists_expected_files(self, tmp_path):
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        )
        assert "descriptors/scale_add/scale_add.kmd.json" in result.stdout
        assert "packs/ScaleAddNative.cpp" in result.stdout


class TestForceOverwrite:
    def test_existing_nonempty_dir_without_force_exits_one(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("hand-authored content")

        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 1
        assert "--force" in result.stderr
        # The pre-existing file must survive untouched.
        assert (output_dir / "existing_file.txt").read_text() == "hand-authored content"

    def test_existing_nonempty_dir_with_force_succeeds(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("stale")

        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(output_dir),
            "--force",
        )
        assert result.returncode == 0, result.stderr

    def test_existing_empty_dir_without_force_succeeds(self, tmp_path):
        """--force only guards a NON-empty directory."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 0, result.stderr


class TestGeneratedOutput:
    def test_success_reports_generated_file_count(self, tmp_path):
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(tmp_path / "out")
        )
        assert "Generated 15 files" in result.stdout

    def test_success_writes_files_to_disk(self, tmp_path):
        output_dir = tmp_path / "out"
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 0, result.stderr
        assert (
            output_dir / "descriptors" / "scale_add" / "scale_add.kmd.json"
        ).exists()
        assert (output_dir / "packs" / "ScaleAddNative.cpp").exists()

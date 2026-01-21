# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rrperf.compare module with focus on resource tracking."""

import io
import sys
from pathlib import Path

from rrperf.compare import compare

repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(repo_dir / "scripts" / "lib"))

FILE_DIR = Path(__file__).parent.resolve()


class TestResourceMarkdown:
    def test_any_increase_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_increase_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "- SGPR: 102 -> 104 (+2) | VGPR: 206 -> 205 (-1)" in result

    def test_decrease_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_decrease_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "+ VGPR: 206 -> 199 (-7) | LDS: 131072 -> 131070 bytes (-2)" in result

    def test_same_dir(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1"  # same dir
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result_lower = result_io.getvalue().lower()
        assert "sgpr" not in result_lower
        assert "vgpr" not in result_lower


class TestHTML:
    def test_decrease_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_decrease_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="html",
            output=result_io,
        )
        result = result_io.getvalue()
        # These are the GPR and LDS value pairs from samples
        expected_values = """
                    <td> 102 </td>
                    <td> 102 </td>
                    <td> 206 </td>
                    <td> 199 </td>
                    <td> 256 </td>
                    <td> 256 </td>
                    <td> 131,072 </td>
                    <td> 131,070 </td>
        """
        for line in expected_values.splitlines():
            line = line.strip()
            if line:
                assert line in result

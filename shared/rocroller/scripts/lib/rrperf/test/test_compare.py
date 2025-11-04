################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

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

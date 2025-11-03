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
from pathlib import Path
import sys

repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(repo_dir / "scripts" / "lib"))

from rrperf.compare import compare

FILE_DIR = Path(__file__).parent.resolve()


def test_resource_change_detection():
    result_io = io.StringIO()
    samples_dir = FILE_DIR / "samples"
    original_dir = samples_dir / "1"
    modified_dir = samples_dir / "1_modified_resource_usage"
    assert original_dir.exists()
    assert modified_dir.exists()
    compare(
        directories=[str(original_dir), str(modified_dir)],
        format="resource_md",
        output=result_io,
    )
    result = result_io.getvalue()
    assert "SGPR: 102 -> 104 (+2) | VGPR: 206 -> 205 (-1)" in result


def test_no_resource_change_detection():
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
    result = result_io.getvalue().lower()
    assert "sgpr" not in result
    assert "vgpr" not in result

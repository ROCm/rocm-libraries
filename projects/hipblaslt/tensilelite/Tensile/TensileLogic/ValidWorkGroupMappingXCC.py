################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
ValidWorkGroupMappingXCC
---
For logic files under a CU-variant directory (e.g. gfx942_38cu), ensures
WorkGroupMappingXCC is -1 or a power-of-two that divides the directory's
CU count. Prevents solutions that would fail WorkgroupMappingXCCCheck at
runtime (e.g. ROCM-2963: XCC=4 with 38 CUs).
"""

import re
from pathlib import Path

# Per-file failure count so we print one full message + one "... N more" per file (not one per solution)
_xcc_failures_by_file = {}


def reset_reported_failures() -> None:
    """Clear per-file failure count so the next run does not reuse old state (e.g. when running in serial)."""
    _xcc_failures_by_file.clear()


def _report_xcc_failure(filepath: Path, solution: dict, detail: str) -> None:
    key = str(filepath)
    count = _xcc_failures_by_file.get(key, 0) + 1
    _xcc_failures_by_file[key] = count
    if count == 1:
        print(f"Error: {detail} (file: {filepath}, index: {solution.get('SolutionIndex', '?')})")
    elif count == 2:
        print(f"  ... (more solutions in this file)")


def _cu_count_from_path(filepath: Path) -> int:
    """Extract CU count from any path component matching *_Ncu (e.g. gfx942_38cu -> 38)."""
    for part in filepath.parts:
        match = re.search(r"_(\d+)cu$", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _validateWorkGroupMappingXCC(solution: dict, filepath: Path) -> bool:
    try:
        cu_count = _cu_count_from_path(filepath)
        if cu_count <= 0:
            return True  # Not a CU-variant directory; skip this check

        xcc = solution.get("WorkGroupMappingXCC", -1)
        if xcc == -1:
            return True

        if xcc <= 0:
            _report_xcc_failure(filepath, solution, f"WorkGroupMappingXCC must be -1 or positive (WorkGroupMappingXCC={xcc})")
            return False

        if (xcc & (xcc - 1)) != 0:
            _report_xcc_failure(filepath, solution, f"WorkGroupMappingXCC must be -1 or a power of two (WorkGroupMappingXCC={xcc})")
            return False

        if cu_count % xcc != 0:
            _report_xcc_failure(filepath, solution, f"WorkGroupMappingXCC={xcc} must divide CU count {cu_count}")
            return False

        return True
    except Exception as e:
        print(f"Error: ValidWorkGroupMappingXCC failed: {e} (file: {filepath})")
        return False

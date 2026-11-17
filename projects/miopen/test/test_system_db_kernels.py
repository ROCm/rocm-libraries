#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################
"""Structural sanity checks for the RDNA system databases in src/kernels/."""

from pathlib import Path

# This file lives at projects/miopen/test/; the DBs live at projects/miopen/src/kernels/.
MIOPEN_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = MIOPEN_ROOT / "src" / "kernels"

FIND_SUFFIX = ".HIP.fdb.txt.bz2"
PERF_SUFFIX = ".db.txt.bz2"


def _rdna_find_dbs():
    return sorted(
        p
        for p in KERNELS_DIR.glob("gfx1*" + FIND_SUFFIX)
        if p.name.startswith(("gfx11", "gfx12"))
    )


def test_rdna_find_perf_paired():
    """Each RDNA find DB must have a matching perf DB (no orphan halves)."""
    missing = []
    for fdb in _rdna_find_dbs():
        base = fdb.name[: -len(FIND_SUFFIX)]
        if not (KERNELS_DIR / (base + PERF_SUFFIX)).is_file():
            missing.append(fdb.name)
    assert not missing, "RDNA find DBs with no matching perf DB: " + ", ".join(missing)


if __name__ == "__main__":
    test_rdna_find_perf_paired()
    print("system DB kernel checks passed")

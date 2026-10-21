# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx12-generic FP16 WMMA attention independently qualified on gfx1201."""

from ...implementations import wmma
from ...contract import Case

NAME = "gfx1201"
COMPILER_TARGET = "gfx12-generic"
CASES = wmma.CASES
CASE_BY_ID = {case.id: case for case in CASES}


def prepare(case: Case, library_root: str) -> wmma.Prepared:
    """Compile the family kernel while retaining this processor's qualification."""
    return wmma.prepare(case, library_root, COMPILER_TARGET)


launch = wmma.launch
exported_kernel = wmma.exported_kernel

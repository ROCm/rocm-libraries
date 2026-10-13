# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1151 standalone FP16 WMMA attention reference cohort."""

from .. import wmma
from ...contract import Case

NAME = "gfx1151"
FAMILIES = (NAME,)
CASES = wmma.CASES
CASE_BY_ID = {case.id: case for case in CASES}


def prepare(case: Case, library_root: str) -> wmma.Prepared:
    """Select this target's standalone WMMA kernel for qualification."""
    return wmma.prepare(case, library_root, NAME)


launch = wmma.launch
exported_kernel = wmma.exported_kernel

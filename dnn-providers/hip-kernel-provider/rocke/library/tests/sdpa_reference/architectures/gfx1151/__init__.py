# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1151 standalone FP16 WMMA attention reference cohort."""

from reference_common.cases import index_cases

from ...implementations import wmma
from ...contract import Case

NAME = "gfx1151"
CASES = wmma.CASES
CASE_BY_ID = index_cases(CASES, label="SDPA")


def prepare(case: Case, library_root: str) -> wmma.Prepared:
    """Select this target's standalone WMMA kernel for qualification."""
    return wmma.prepare(case, library_root, NAME)


launch = wmma.launch
exported_kernel = wmma.exported_kernel

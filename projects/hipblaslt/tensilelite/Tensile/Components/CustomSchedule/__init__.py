################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

"""CustomSchedule package.

Public façade for the per-arch / per-schedule custom main-loop schedule
registry. External callers (KernelWriter, Solution, tests) import from this
module; internals live in `shared`, `dispatch`, and `gfx950/`.

The internal `_SCHEDULE_REGISTRY` and `_SCHEDULE_METADATA` lists are leaked
through this façade because `test_CustomSchedule_LayoutAutoDetection.py`
mutates them (save/restore around tests). They are re-exported as live
bindings to the same module-level objects in `dispatch.py` — never
deepcopied — so the test fixture continues to work.
"""

# Order matters: dispatch must be imported BEFORE the gfx950 subpackage so the
# registry lists exist when each per-schedule module's `@RegisterSchedule`
# decorator runs at import time.

from .shared import (
    CMSKernelInfo,
    ScheduleInfo,
    SyncSchedule,
    TileConfig,
    is16bit,
    is8bit,
    isMixed,
    isNN,
    isNT,
    isTF32,
    isTN,
    isTT,
)
from .dispatch import (
    RegisterSchedule,
    ScheduleMatchStatus,
    _SCHEDULE_METADATA,
    _SCHEDULE_REGISTRY,
    customMainLoopSchedule,
    get_available_dtypes,
    get_available_layouts,
    get_cms_kernel_info_objects,
    hasCustomSchedule,
    query_cms_kernels,
)

# Side-effect import: runs every per-schedule module's @RegisterSchedule.
from . import gfx950  # noqa: F401

__all__ = [
    # shared
    "CMSKernelInfo",
    "ScheduleInfo",
    "SyncSchedule",
    "TileConfig",
    "is16bit",
    "is8bit",
    "isMixed",
    "isNN",
    "isNT",
    "isTF32",
    "isTN",
    "isTT",
    # dispatch
    "RegisterSchedule",
    "ScheduleMatchStatus",
    "_SCHEDULE_METADATA",
    "_SCHEDULE_REGISTRY",
    "customMainLoopSchedule",
    "get_available_dtypes",
    "get_available_layouts",
    "get_cms_kernel_info_objects",
    "hasCustomSchedule",
    "query_cms_kernels",
]

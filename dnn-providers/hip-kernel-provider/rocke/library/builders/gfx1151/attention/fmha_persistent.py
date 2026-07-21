# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Back-compat shim: the persistent-grid (work-queue) transposed-QK variant now
lives in ``kernels.gfx1151.wmma_fmha_swapqk_persistent`` (a long-sequence
production candidate). Re-exported here so ``pers_tune`` keeps importing from
this path."""

from __future__ import annotations

from kernels.gfx1151.wmma_fmha_swapqk_persistent import (  # noqa: F401
    PersistentCfg,
    build_wmma_fmha_persistent,
    num_work_items,
    persistent_grid,
)

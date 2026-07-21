# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Back-compat shim: the multi-wave (occupancy) WMMA FMHA variant now lives in
``kernels.gfx1151.wmma_fmha_multiwave`` (a long-sequence production candidate).
Re-exported here so ``mw_tune`` / ``ck_parity_probe`` keep importing from this
path."""

from __future__ import annotations

from kernels.gfx1151.wmma_fmha_multiwave import (  # noqa: F401
    MultiWaveCfg,
    build_wmma_fmha_multiwave,
    multiwave_grid,
)

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 SCALE/SCALE16 matrix format selectors.

Mirrored by scaled_wmma_internal.h. The packed matrix ABI uses sixteen i32
words for every format; FP6 and FP4 use twelve and eight meaningful words.
"""

from __future__ import annotations

MATRIX_FORMATS = {"fp8": 0, "bf8": 1, "fp6": 2, "bf6": 3, "fp4": 4}
SCALED_WMMA_OPS = {
    f"{mode}_f32_16x16x128_{a}_{b}": (mode == "wmma_scale16", fa, fb)
    for mode in ("wmma_scale", "wmma_scale16")
    for a, fa in MATRIX_FORMATS.items()
    for b, fb in MATRIX_FORMATS.items()
}

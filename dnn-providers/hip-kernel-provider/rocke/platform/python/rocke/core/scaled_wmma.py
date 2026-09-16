# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 SCALE/SCALE16 matrix and scale format selectors.

Mirrored by scaled_wmma_internal.h. The packed matrix ABI uses sixteen i32
words for every format; FP6 and FP4 use twelve and eight meaningful words.
"""

from __future__ import annotations

MATRIX_FORMATS = {"fp8": 0, "bf8": 1, "fp6": 2, "bf6": 3, "fp4": 4}
SCALE_FORMATS = {"e8m0": 0, "i8": 0, "e5m3": 1, "e4m3": 2}
SCALED_WMMA_OPS = {
    f"{mode}_f32_16x16x128_{a}_{b}": (mode == "wmma_scale16", fa, fb)
    for mode in ("wmma_scale", "wmma_scale16")
    for a, fa in MATRIX_FORMATS.items()
    for b, fb in MATRIX_FORMATS.items()
}


def scale_formats(
    fmt_a: int, fmt_b: int, dtype_a: str = "e8m0", dtype_b: str = "e8m0"
) -> tuple[int, int]:
    """Validate a documented matrix/scale combination and return selectors."""
    if dtype_a not in SCALE_FORMATS or dtype_b not in SCALE_FORMATS:
        raise ValueError("scaled WMMA scale types must be e8m0, e5m3, or e4m3")
    sa, sb = SCALE_FORMATS[dtype_a], SCALE_FORMATS[dtype_b]
    if (fmt_a != 4 and sa != 0) or (fmt_b != 4 and sb != 0):
        raise ValueError("scaled WMMA e5m3/e4m3 scales require an FP4 operand")
    if fmt_a == fmt_b == 4 and sa != sb:
        raise ValueError("scaled WMMA FP4 x FP4 requires matching scale formats")
    return sa, sb

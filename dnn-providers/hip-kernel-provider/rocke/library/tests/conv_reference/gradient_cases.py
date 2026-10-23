# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bounded initial 2-D gradient cohorts for gfx942 and gfx950."""

from .contract import Case

COMMON_CASES = tuple(
    Case(name, dtype, **geometry)
    for name, geometry in (
        ("padded", {}),
        ("pointwise", {"Y": 1, "X": 1, "pH": 0, "pW": 0}),
        ("stride2", {"Hi": 9, "Wi": 11, "sH": 2, "sW": 2}),
        ("dilation2", {"dH": 2, "dW": 2, "pH": 2, "pW": 2}),
        ("groups2", {"groups": 2}),
        ("odd-channels", {"C": 33, "K": 33}),
    )
    for dtype in ("fp16", "bf16")
)
DGRAD_CASES = COMMON_CASES
WGRAD_CASES = COMMON_CASES + tuple(
    Case("depthwise", dtype, C=64, K=64, groups=64) for dtype in ("fp16", "bf16")
)

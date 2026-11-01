# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent rounding references of the hipTensor kernel tests (pure Python, no GPU): one
rounding to nearest even into f16 / bf16 with exact rational arithmetic, and f64 probe values
next to the rounding boundaries of those formats."""

from __future__ import annotations

import math
import random

#: (significand bits, smallest normal exponent, largest exponent) of the 16-bit formats
FORMATS = {"f16": (11, -14, 15), "bf16": (8, -126, 127)}


def round_rne(x: float, dt: str) -> float:
    """``x`` rounded once to nearest even in the 16-bit format ``dt`` (subnormals included,
    overflow to infinity), with exact rational arithmetic: an independent reference."""
    from fractions import Fraction

    p, emin, emax = FORMATS[dt]
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x
    sign = -1.0 if x < 0 else 1.0
    a = Fraction(abs(x))
    q = max(math.frexp(abs(x))[1] - 1, emin) - (p - 1)  # exponent of the last bit
    v = a / Fraction(2) ** q
    n = v.numerator // v.denominator
    r = v - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    out = n * Fraction(2) ** q
    if out > (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax:
        return sign * math.inf
    return sign * float(out)


def rounding_probes(dt: str):
    """f64 values next to the rounding boundaries of ``dt``: midpoints between two neighbours,
    and values a tiny relative distance (far below an f32 ulp) above and below them, in the
    subnormal, normal and overflow ranges, both signs; plus zeros, infinities and NaN.
    """
    p, emin, emax = FORMATS[dt]
    rng = random.Random(p * 31 + emax)
    out = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 1e300, -1e300]
    for e in (emin - p, emin - 3, emin, -7, -1, 0, 1, 9, emax - 1, emax):
        ulp = 2.0 ** (max(e, emin) - (p - 1))
        for _ in range(12):
            lead = 2.0**e if e >= emin else 0.0
            base = lead + rng.randrange(1 << (p - 1)) * ulp
            mid = base + ulp / 2
            # offsets (in ulps of dt) below half an f32 ulp, and one above as a control
            for d in (
                -(2.0**-12),
                -(2.0**-30),
                -(2.0**-38),
                0.0,
                2.0**-38,
                2.0**-30,
                2.0**-12,
            ):
                v = mid + ulp * d
                out += [v, -v]
    top = (2 - 2.0 ** (1 - p)) * 2.0**emax  # the largest finite value
    half = 2.0 ** (emax - p)  # half its ulp
    for v in (
        top,
        top + half,
        top + half * (1 - 2.0**-30),
        top + half * (1 + 2.0**-30),
    ):
        out += [v, -v]
    return out

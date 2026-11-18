# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fp8 / bf8 element encodings, and which one each target uses.

Two families of 8-bit float encodings exist, with the same bit layouts:

* OCP: E4M3 (bias 7, largest finite 448, NaN at ``0x7f`` / ``0xff``, no
  infinities) and E5M2 (bias 15, largest finite 57344, IEEE-like infinities
  and NaNs).
* FNUZ: the same layouts with the bias one higher, no infinities and no
  negative zero; ``0x80`` is the one NaN. Largest finite 240 (E4M3) and 57344
  (E5M2).

A target's f32 -> fp8 / bf8 conversions produce one family and its matrix
cores read the same one: FNUZ on gfx942, OCP on gfx950 and gfx1250. The
conversions do not saturate, so a kernel that quantizes must keep its values
within :attr:`Fp8ElementFormat.largest_value` of the target's encoding
(:func:`fp8_element_format`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BF8_E5M2_FNUZ",
    "BF8_E5M2_OCP",
    "FP8_E4M3_FNUZ",
    "FP8_E4M3_OCP",
    "Fp8ElementFormat",
    "fp8_element_format",
]


@dataclass(frozen=True)
class Fp8ElementFormat:
    """An fp8 / bf8 element encoding: ``name`` says which one (for example
    ``"E4M3 FNUZ"``) and ``largest_value`` is its largest finite magnitude."""

    name: str
    largest_value: float


FP8_E4M3_OCP = Fp8ElementFormat("OCP E4M3", 448.0)
BF8_E5M2_OCP = Fp8ElementFormat("OCP E5M2", 57344.0)
FP8_E4M3_FNUZ = Fp8ElementFormat("E4M3 FNUZ", 240.0)
BF8_E5M2_FNUZ = Fp8ElementFormat("E5M2 FNUZ", 57344.0)

# The fp8 / bf8 dtypes an encoding is looked up for.
_FP8_DTYPES = ("bf8e5m2", "fp8e4m3")

# The encoding each target's conversions produce and its matrix cores read.
_FP8_ELEMENT_FORMATS = {
    "gfx942": {"fp8e4m3": FP8_E4M3_FNUZ, "bf8e5m2": BF8_E5M2_FNUZ},
    "gfx950": {"fp8e4m3": FP8_E4M3_OCP, "bf8e5m2": BF8_E5M2_OCP},
    "gfx1250": {"fp8e4m3": FP8_E4M3_OCP, "bf8e5m2": BF8_E5M2_OCP},
}


def fp8_element_format(dtype: str, arch: str) -> Fp8ElementFormat:
    """The encoding ``arch`` uses for ``dtype`` (``"fp8e4m3"`` or
    ``"bf8e5m2"``): what its f32 -> fp8 / bf8 conversions produce and its
    matrix cores read. FNUZ on gfx942, OCP on gfx950 and gfx1250.

    Raises ``ValueError`` for any other dtype or arch.
    """
    if dtype not in _FP8_DTYPES:
        raise ValueError(
            f"fp8_element_format failed - dtype={dtype!r}, expected one of "
            f"{_FP8_DTYPES}"
        )
    if arch not in _FP8_ELEMENT_FORMATS:
        raise ValueError(
            f"fp8_element_format failed - arch={arch!r}, expected one of "
            f"{tuple(_FP8_ELEMENT_FORMATS)}"
        )
    return _FP8_ELEMENT_FORMATS[arch][dtype]

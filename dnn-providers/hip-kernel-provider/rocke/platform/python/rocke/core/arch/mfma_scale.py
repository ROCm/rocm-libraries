# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled MFMA: 32 logical values and one K32 E8M0 scale per lane."""

from ..dtypes import dtype_info
from ..storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from .target import ArchTarget, _op_id_family

MATRIX_FORMATS = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
INTRINSIC_KEY = "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32"


def gfx950_scaled_mfma(op_id):
    if _op_id_family().get(op_id.removeprefix("tile.")) != "mfma_scaled":
        return None
    atom = ArchTarget.from_gfx("gfx950").mma.by_op_id(op_id.removeprefix("tile."))
    return atom if atom is not None and atom.family == "mfma_scaled" else None


def matrix_layout(dtype):
    if dtype not in MATRIX_FORMATS:
        raise ValueError("unsupported scaled MFMA matrix dtype")
    return MatrixFragmentLayout(
        FragmentPacking(BitPacking(dtype_info(dtype).encoded_bits), 32, 32, 8),
        16 if MATRIX_FORMATS[dtype] < 2 else 32,
        2,
        32,
    )


def scaled_mfma_error(op_id, operands, results=None):
    if gfx950_scaled_mfma(op_id) is None:
        return None
    if [v.type.name for v in operands] != [
        "vec<i32x8>",
        "vec<i32x8>",
        "vec<f32x16>",
        "i32",
        "i32",
    ]:
        return "scaled MFMA requires two vec<i32x8> payloads, vec<f32x16> accumulator and two i32 E8M0 scales"
    if results is not None and [v.type.name for v in results] != ["vec<f32x16>"]:
        return "scaled MFMA result must be vec<f32x16>"
    return None


# --- K=128 f8f6f4 MFMA (16x16x128): format selectors and operand width -------

DEFAULT_MATRIX_FORMAT = "fp8e4m3"
# Every operand of the 16x16x128 call is 256 bits per lane (<8 x i32>). fp8 and
# bf8 data fill all 8 dwords. fp6 data fills the low 6 and fp4 data the low 4;
# the instruction ignores the remaining dwords. This matches the 32x32x64 family.
F8F6F4_16X16X128_OPERAND_BITS = 256


def _type_bits(t):
    """Encoded bit width of an IR type, or ``None`` when it is not known."""
    count = getattr(t, "count", None)
    elem = getattr(t, "elem", None)
    if count is not None and elem is not None:
        bits = _type_bits(elem)
        return None if bits is None else count * bits
    try:
        return dtype_info(t.name).encoded_bits
    except (ValueError, AttributeError):
        return None


def f8f6f4_16x16x128_selectors(a_format, b_format):
    """Return ``(cbsz, blgp)`` for the K=128 f8f6f4 MFMA.

    ``a_format`` and ``b_format`` must be known format names, and they must be
    equal in this version. Raises ``ValueError`` otherwise.
    """
    for role, fmt in (("a_format", a_format), ("b_format", b_format)):
        if fmt not in MATRIX_FORMATS:
            raise ValueError(
                f"unknown {role} {fmt!r}; use one of {', '.join(MATRIX_FORMATS)}"
            )
    if a_format != b_format:
        raise ValueError(
            "a_format and b_format must name the same format in this version; "
            f"got {a_format!r} and {b_format!r}"
        )
    return MATRIX_FORMATS[a_format], MATRIX_FORMATS[b_format]


def f8f6f4_16x16x128_operand_error(a, b):
    """Return an error text when A or B is not 256 bits per lane, else ``None``."""
    for role, v in (("A", a), ("B", b)):
        bits = _type_bits(v.type)
        if bits != F8F6F4_16X16X128_OPERAND_BITS:
            got = "an unknown width" if bits is None else f"{bits} bits"
            return (
                f"the {role} operand of the 16x16x128 f8f6f4 MFMA must have "
                f"{F8F6F4_16X16X128_OPERAND_BITS} bits per lane (pad fp6 and fp4 "
                f"data to 8 dwords); got {got}"
            )
    return None
